import numpy as np
import optuna
import yfinance as yf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from MLPerceptron import MultilayerPerceptron
import pickle

# Data
ticker = "NVDA"
data = yf.download(ticker, period="1y")
closing_prices = data["Close"].values.reshape(-1, 1)

# X: today's closing price, y: tomorrow's closing price
X = closing_prices[:-1]
y = closing_prices[1:]

# Train-test split (no shuffling to preserve time series order)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# standardization 
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Targets should be column vectors 
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

if __name__ == "__main__":
    # Further split training set for internal validation during hyperparameter tuning
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    # Optimization using BO 
    def objective(trial):
        # Sample hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 10)
        hidden_layers_neurons = [trial.suggest_int(f"n_units_l{i}", 5, 50) for i in range(n_layers)]
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 500, 2000)

        # Initialize model with sampled architecture
        model = MultilayerPerceptron(
            input_size=X_train_sub.shape[1],
            hidden_layers=hidden_layers_neurons,
            output_size=y_train_sub.shape[1]
        )

        # Train the model
        for epoch in range(epochs):
            preds = model.propagate_forward(X_train_sub.T)
            grads = model.propagate_backward(X_train_sub.T, y_train_sub.T)
            model.update_parameters(grads, learning_rate)

        # Validate on the hold-out validation set
        val_preds = model.propagate_forward(X_val.T)
        val_loss = np.mean((val_preds - y_val.T) ** 2)
        return val_loss


    # Hyperparameter optimization 
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)


    # Print the best hyperparameters found
    print("Best hyperparameters found:")
    with open("best_hyperparams.pkl", "wb") as f:
        pickle.dump(study.best_params,f)

