import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load dataset
data = pd.read_csv('../dataset/process/tweets_convert_cleaned.tsv', sep='\t')


X = data['text'].values
emo = data['emo'].fillna('').values
y = data['sentiment'].values

# Split data into train, validation, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp, emo_train, emo_temp = train_test_split(X, y, emo, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, emo_val, emo_test = train_test_split(X_temp, y_temp, emo_temp, test_size=0.5, random_state=42)

print(f'Train size: {len(X_train)}')
print(f'Validation size: {len(X_val)}')
print(f'Test size: {len(X_test)}')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, texts, emos, labels, tokenizer, max_len):
        self.texts = texts
        self.emos = emos
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        emo = str(self.emos[item]) if pd.notna(self.emos[item]) else ''
        label = self.labels[item]

        text_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        emo_encoding = self.tokenizer.encode_plus(
            emo,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'emo': emo,
            'input_ids_text': text_encoding['input_ids'].flatten(),
            'attention_mask_text': text_encoding['attention_mask'].flatten(),
            'input_ids_emo': emo_encoding['input_ids'].flatten(),
            'attention_mask_emo': emo_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

MAX_LEN = 160
BATCH_SIZE = 16

train_dataset = SentimentDataset(X_train, emo_train, y_train, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(X_val, emo_val, y_val, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(X_test, emo_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define multi-input BERT model
class MultiInputBERT(nn.Module):
    def __init__(self, bert_model_name, num_labels, hidden_dim=768):
        super(MultiInputBERT, self).__init__()
        self.bert_text = BertModel.from_pretrained(bert_model_name)
        self.bert_emo = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids_text, attention_mask_text, input_ids_emo, attention_mask_emo):
        text_outputs = self.bert_text(input_ids_text, attention_mask=attention_mask_text)
        emo_outputs = self.bert_emo(input_ids_emo, attention_mask=attention_mask_emo)

        text_pooled_output = text_outputs.pooler_output
        emo_pooled_output = emo_outputs.pooler_output

        combined_output = torch.cat((text_pooled_output, emo_pooled_output), dim=1)
        logits = self.classifier(combined_output)
        return logits

model = MultiInputBERT(bert_model_name='bert-base-uncased', num_labels=3)
model = model.to(device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define loss function
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = 0
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids_text = d["input_ids_text"].to(device)
        attention_mask_text = d["attention_mask_text"].to(device)
        input_ids_emo = d["input_ids_emo"].to(device)
        attention_mask_emo = d["attention_mask_emo"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids_text=input_ids_text,
            attention_mask_text=attention_mask_text,
            input_ids_emo=input_ids_emo,
            attention_mask_emo=attention_mask_emo
        )

        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, losses / n_examples

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = 0
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids_text = d["input_ids_text"].to(device)
            attention_mask_text = d["attention_mask_text"].to(device)
            input_ids_emo = d["input_ids_emo"].to(device)
            attention_mask_emo = d["attention_mask_emo"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids_text=input_ids_text,
                attention_mask_text=attention_mask_text,
                input_ids_emo=input_ids_emo,
                attention_mask_emo=attention_mask_emo
            )

            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses += loss.item()

    return correct_predictions.double() / n_examples, losses / n_examples

EPOCHS = 5

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(X_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(X_val)
    )

    print(f'Validation loss {val_loss} accuracy {val_acc}')

# Save the model
torch.save(model.state_dict(), 'multi_input_bert_model.bin')

# Evaluate the model
model.load_state_dict(torch.load('multi_input_bert_model.bin'))
model = model.to(device)

y_pred = []
y_true = []

model = model.eval()

with torch.no_grad():
    for d in test_loader:
        input_ids_text = d["input_ids_text"].to(device)
        attention_mask_text = d["attention_mask_text"].to(device)
        input_ids_emo = d["input_ids_emo"].to(device)
        attention_mask_emo = d["attention_mask_emo"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids_text=input_ids_text,
            attention_mask_text=attention_mask_text,
            input_ids_emo=input_ids_emo,
            attention_mask_emo=attention_mask_emo
        )

        _, preds = torch.max(outputs, dim=1)

        y_pred.extend(preds)
        y_true.extend(labels)

y_pred = torch.stack(y_pred).cpu()
y_true = torch.stack(y_true).cpu()

print('Accuracy:', accuracy_score(y_true, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['negative', 'neutral', 'positive'])
    ax.yaxis.set_ticklabels(['negative', 'neutral', 'positive'])

plot_confusion_matrix(y_true, y_pred)
plt.show()