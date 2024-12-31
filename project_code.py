#!/home/jmalec/projects/def-ilie/jmalec/myenv3/bin/python
import os
import warnings
import re
from sklearn import svm
from Bio import SeqIO
from scipy import stats
import blosum as bl
import numpy as np
import pandas as pd
import torch
import sentencepiece
import shutil
from tqdm import tqdm
from transformers import T5Model
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #always going to be GPU


#code to initialize protT5
def ProtT5_initialize():

    ProtT5 = T5EncoderModel.from_pretrained("/home/jmalec/projects/def-ilie/jmalec/protT5", local_files_only=True)
    ProtT5.half()
    ProtT5 = ProtT5.to(device)
    ProtT5 = ProtT5.eval()
    ProtT5_tokenizer = T5Tokenizer.from_pretrained("/home/jmalec/projects/def-ilie/jmalec/protT5", do_lower_case=False)

    return ProtT5, ProtT5_tokenizer


def get_embs_T5(T5, tokenizer, sequences, n):
  sequence_examples = sequences[:n]

  # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
  sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

  # tokenize sequences and pad up to the longest sequence in the batch
  ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  # generate embeddings
  with torch.no_grad():
      embedding_repr = T5(input_ids=input_ids, attention_mask=attention_mask)

  last_layer_repr = embedding_repr.last_hidden_state
  final_embs = []
  for i in range(len(last_layer_repr)):
    final_embs.append(last_layer_repr[i , :len(sequences[i])])

  return final_embs


####################################################################################################################################################################################
#for processing the input CSVs
def parse_dataframe(data, model, tokenizer):

    examples_seqs = []
    embeddings = []
    labels = []
    concatenated_embeddings = None
    label_all = ""

    # Step 1: Extract sequences and labels from DataFrame
    for index, row in data.iterrows():
        seq = row["seq"]
        labels8 = row["sst3"]
        examples_seqs.append((seq, labels8))

    # Step 2: Generate embeddings and collect labels
    for exampleSeq in examples_seqs:
        seq = exampleSeq[0]
        label = exampleSeq[1]

        # Generate embeddings (assuming get_embs_T5 outputs a single embedding array)
        emb = get_embs_T5(model, tokenizer, [seq], 1)[0].cpu().numpy()

        if concatenated_embeddings is None:  # Handle the first case
            concatenated_embeddings = emb
        else:
            concatenated_embeddings = np.concatenate((concatenated_embeddings, emb), axis=0)

        label_all = label_all + label

    return concatenated_embeddings, label_all
####################################################################################################################################################################################

def plot_learning_curves(model, X_train, X_test, y_train, y_test, interval=100):
    train_accuracies, val_accuracies = [], []

    # Loop through the data with increasing training set size in intervals of 'interval'
    for m in range(interval, len(X_train) + 1, interval):
        # Only fit the model if there are at least two unique classes in the current subset
        if len(np.unique(y_train[:m])) > 1:  # Ensure there are multiple classes
            model.fit(X_train[:m], y_train[:m])  # Train the model on a subset
            y_train_predict = model.predict(X_train[:m])  # Predict on the training set
            y_val_predict = model.predict(X_test)  # Predict on the validation set

            # Calculate and store training and validation accuracies
            train_accuracies.append(accuracy_score(y_train[:m], y_train_predict))
            val_accuracies.append(accuracy_score(y_test, y_val_predict))
        else:
            # Append NaN for the case where there's only one class in the training subset
            train_accuracies.append(np.nan)
            val_accuracies.append(np.nan)

    # Plot the learning curves
    plt.plot(range(interval, len(X_train) + 1, interval), train_accuracies, "r-+", linewidth=2, label="Training Set")
    plt.plot(range(interval, len(X_train) + 1, interval), val_accuracies, "b-", linewidth=3, label="Validation Set")
    plt.legend(loc="lower right", fontsize=14)
    plt.xlabel("Dataset Size", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    plt.yticks(np.linspace(0, 1, 11))
    plt.grid(True)

####################################################################################################################################################################################
def shuffle_data(X, y):
    #ensure consistent shuffling for features and labels
    assert len(X) == len(y), "Features and labels must have the same length"
    indices = np.arange(len(X))  #create an array of indices
    np.random.shuffle(indices)  #shuffle the indices
    return X[indices], [y[i] for i in indices]
####################################################################################################################################################################################

def main():
    
    #embeddings
    emb_train = np.loadtxt('train_embeddings.csv', delimiter=',')
    emb_train = emb_train[1:] #remove header
    emb_test = np.loadtxt('test_embeddings.csv', delimiter=',')
    emb_test = emb_test[1:] #remove header

    #labels
    train_list_label = []
    test_list_label = []

    #open the file in read mode
    with open('train_labels.txt', 'r') as file:
        #iterate through each line in the file
        for line in file:
            #strip any trailing whitespace (like newlines) and add the character to the list
            train_list_label.append(line.strip())

    with open('test_labels.txt', 'r') as file:
        #iterate through each line in the file
        for line in file:
            #strip any trailing whitespace (like newlines) and add the character to the list
            test_list_label.append(line.strip())

    X_train = emb_train
    X_test = emb_test
    y_train = train_list_label
    y_test= test_list_label

    #shuffle
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test) 

    #increase sizes here
    X_train = X_train[:1000]
    X_test = X_test[:1000]
    y_train = y_train[:1000]
    y_test = y_test[:1000]

    #option 1) for learning curve
    svm_model = svm.SVC(kernel='sigmoid', C=1, gamma=0.01, degree = 2, decision_function_shape='ovr') #changes
    plot_learning_curves(svm_model, X_train, X_test, y_train, y_test)
    # plt.show()
    plt.savefig("plot.png") 

    #option 2)for training without learning curve
    svm_model = svm.SVC(kernel='sigmoid', C=1, gamma=0.01, degree = 2, decision_function_shape='ovo') #changes
    svm_model.fit(X_train, y_train)
    #make predictions on the test set
    y_pred = svm_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    #option 3) for training with grid seach set up
    #make changes as needed
    param_grid = {
    'C': [0.9, 0.95, 1, 1.05, 1.1], 
    'gamma': ['scale', 0.5, 0.1, 0.01, 0.001],  
    'degree': [2,3,4]  
    }
    svm_model = SVC(kernel='rbf') #make changes to the kernel
    #find the best combination of parameters
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    #make predictions using the best model
    y_pred = best_model.predict(X_test)
    #print results
    print("Best Parameters:", best_params)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()