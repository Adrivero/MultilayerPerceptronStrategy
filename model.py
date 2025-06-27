import numpy as np
from MLPerceptron import MultilayerPerceptron
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
from preprocessing_hyperparamOptimization import *


def predict(x_test):
    with open("best_hyperparams.pkl","rb") as f:
        best_params = pickle.load(f)


    print(best_params)
    
    # Train final model using the best parameters
    best_params = best_params
    final_hidden_layers = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
    final_learning_rate = best_params["learning_rate"]
    final_epochs = best_params["epochs"]

    final_model = MultilayerPerceptron(
        input_size=X_train.shape[1],
        hidden_layers=final_hidden_layers,
        output_size=y_train.shape[1]
    )

    # Train the final model
    for epoch in range(final_epochs):
        preds = final_model.propagate_forward(X_train.T)
        grads = final_model.propagate_backward(X_train.T, y_train.T)
        final_model.update_parameters(grads, final_learning_rate)

    # Evaluate on the test set
    return final_model.propagate_forward(x_test)

    


if __name__ == "__main__":
    x_test = X_test.reshape(1,-1)
    test_preds = predict(x_test=x_test)  
    # Plotting predictions
    plt.plot(y_test, label="Actual")
    plt.plot(test_preds.T, label="Predicted")
    plt.ylabel("Closing Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.legend()
    plt.show()

