import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def plot_learning_curves(model, X_train, X_test, y_train, y_test, interval=100):
    '''Plot learning curves for training and validation sets.
    Parameters:
    model: The machine learning model to be trained and evaluated
    X_train: Training features
    X_test: Validation features
    y_train: Training labels
    y_test: Validation labels
    interval: The step size for increasing the training set size (default is 100)'''
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
