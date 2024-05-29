import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords 
from collections import Counter
import re
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 加載並清理數據
data = pd.read_csv('../dataset/process/en-2020-01-merged-cleaned-without-emoji.tsv', sep='\t')
data = data.dropna()

X = data['text'].values
y = data['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'Train size: {len(X_train)}')
print(f'Test size: {len(X_test)}')

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r"\d", '', s)
    return s

def tokenize_and_pad(x_train, x_val):
    word_list = []
    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
    
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    def tokenize(sentences):
        final_list = []
        for sent in sentences:
            final_list.append(torch.tensor([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                            if preprocess_string(word) in onehot_dict.keys()]))
        return final_list
    
    tokenized_train = tokenize(x_train)
    tokenized_test = tokenize(x_val)
    
    max_len = max(max(len(seq) for seq in tokenized_train), max(len(seq) for seq in tokenized_test))
    
    final_list_train = pad_sequence(tokenized_train, batch_first=True, padding_value=0)
    final_list_test = pad_sequence(tokenized_test, batch_first=True, padding_value=0)
    
    return final_list_train, final_list_test, onehot_dict

X_train_tk, X_test_tk, onehot_dict = tokenize_and_pad(X_train, X_test)

print(f'X_train_tk shape: {X_train_tk.shape}')
print(f'X_test_tk shape: {X_test_tk.shape}')
print(f'Onehot_dict length: {len(onehot_dict)}')

train_data = TensorDataset(torch.from_numpy(X_train_tk.numpy()), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test_tk.numpy()), torch.from_numpy(y_test))
batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# 加載 GloVe 向量
def load_glove_embeddings(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        glove_vectors = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

glove_file = '../dataset/glove_twitter_27B/glove.twitter.27B.200d.txt'
glove_vectors = load_glove_embeddings(glove_file)

embedding_dim = 200
vocab_size = len(onehot_dict) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, index in onehot_dict.items():
    if word in glove_vectors:
        embedding_matrix[index] = glove_vectors[word]

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

no_layers = 3
hidden_dim = 256

model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)
model.to(device)

lr = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item() / len(label)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for i, batch in enumerate(tqdm(iterator, desc="Training")):
        optimizer.zero_grad()
        tweet, labels = batch
        labels = labels.type(torch.FloatTensor)
        tweet, labels = tweet.to(device), labels.to(device)
        hidden = model.init_hidden(labels.shape[0])
        output, hidden = model(tweet, hidden)
        loss = criterion(output.squeeze(), labels)
        accura = acc(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accura
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Evaluating")):
            tweet, labels = batch
            labels = labels.type(torch.FloatTensor)
            tweet, labels = tweet.to(device), labels.to(device)
            hidden = model.init_hidden(labels.shape[0])
            output, hidden = model(tweet, hidden)
            loss = criterion(output.squeeze(), labels.squeeze())
            accura = acc(output, labels)
            epoch_loss += loss.item()
            epoch_acc += accura
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 10
best_valid_loss = float('inf')

train_epoch_losses = []
train_epoch_accs = []
val_epoch_losses = []
val_epoch_accs = []

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)
    train_epoch_losses.append(train_loss)
    train_epoch_accs.append(train_acc)
    val_epoch_losses.append(valid_loss)
    val_epoch_accs.append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    print(f'Epoch: {epoch}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_epoch_losses, label="training")
plt.plot(val_epoch_losses, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])

model.load_state_dict(torch.load('saved_weights.pt'))
model.eval()

y_pred_list = []
y_true_list = []

with torch.no_grad():
    for i, (tweet, labels) in enumerate(test_loader):
        tweet, labels = tweet.to(device), labels.to(device)
        hidden = model.init_hidden(labels.shape[0])
        output, hidden = model(tweet, hidden)
        y_pred_tag = torch.round(output)
        y_pred_list.append(y_pred_tag.cpu().numpy())
        y_true_list.append(labels.cpu().numpy())

# Flatten the lists and ensure they are integers
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_true_list = [item for sublist in y_true_list for item in sublist]

# Binarize predictions
y_pred_list = [1 if i > 0.5 else 0 for i in y_pred_list]

plot_confusion_matrix(y_true_list, y_pred_list)
plt.show()
plot_confusion_matrix(y_true_list, y_pred_list)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true_list, y_pred_list)
print(f'Accuracy: {accuracy*100:.2f}%')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

model.load_state_dict(torch.load('saved_weights.pt'))
model.eval()

y_pred_list = []
y_true_list = []

with torch.no_grad():
    for i, (tweet, labels) in enumerate(test_loader):
        tweet, labels = tweet.to(device), labels.to(device)
        hidden = model.init_hidden(labels.shape[0])
        output, hidden = model(tweet, hidden)
        y_pred_list.append(output.cpu().numpy())
        y_true_list.append(labels.cpu().numpy())

y_pred_list = [item for sublist in y_pred_list for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
y_true_list = [item for sublist in y_true_list for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]

fpr, tpr, _ = roc_curve(y_true_list, y_pred_list)
roc_auc = roc_auc_score(y_true_list, y_pred_list)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

