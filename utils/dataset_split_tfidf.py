import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.sparse

def load_and_split_data():
    # read sparse matrix
    X = scipy.sparse.load_npz('../dataset/process/tfidf_sparse.npz')
    data = pd.read_csv('../dataset/process/en-2020-01-merged-cleaned-without-emoji-tfidf.tsv', sep='\t')

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
