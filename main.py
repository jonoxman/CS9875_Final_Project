from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from protT5 import ProtT5_initialize, get_embs_T5
from preprocessing import shuffle_data, parse_dataframe
from model_inspection import plot_learning_curves

def main():
    '''Main function to execute the workflow of loading data, generating embeddings, 
    training an SVM model, and evaluating its performance.'''
    
    model, tokenizer = ProtT5_initialize()
    #embeddings
    emb_train, seqs_train = parse_dataframe(open('train.csv'), model, tokenizer) 
    emb_test, seqs_test = parse_dataframe(open('test.csv'), model, tokenizer)
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
    '''svm_model = svm.SVC(kernel='sigmoid', C=1, gamma=0.01, degree = 2, decision_function_shape='ovr') #changes
    plot_learning_curves(svm_model, X_train, X_test, y_train, y_test)
    # plt.show()
    plt.savefig("plot.png") 
    '''
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