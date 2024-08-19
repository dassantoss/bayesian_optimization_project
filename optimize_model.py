#!/usr/bin/env python3
"""
Script to optimize a machine learning model using Bayesian Optimization
and GPyOpt.
"""
import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


class BayesianOptimizationModel:
    """Class to perform Bayesian Optimization on a machine learning model."""

    def __init__(self):
        """
        Initializes the BayesianOptimizationModel class.

        Defines the search space for the hyperparameters.
        """
        self.bounds = [
            {'name': 'lr', 'type': 'continuous', 'domain': (0.0001, 0.1)},
            {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128,
                                                             256)},
            {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
            {'name': 'l2', 'type': 'continuous', 'domain': (0.0001, 0.01)},
            {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64,
                                                                  128)}
        ]

    def build_model(self, lr, units, dropout, l2_reg, input_shape):
        """
        Builds and compiles a Keras Sequential model.

        Args:
            lr (float): Learning rate.
            units (int): Number of units in the hidden layer.
            dropout (float): Dropout rate.
            l2_reg (float): L2 regularization weight.
            input_shape (int): Shape of the input data.

        Returns:
            model: Compiled Keras model.
        """
        model = Sequential([
            Dense(units, activation='relu', kernel_regularizer=l2(l2_reg),
                  input_shape=(input_shape,)),
            Dropout(dropout),
            Dense(1, activation='linear')
        ])

        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def objective_function(self, X):
        """
        Objective function to minimize during Bayesian Optimization.

        Args:
            X (numpy.ndarray): Array of hyperparameter values to evaluate.

        Returns:
            validation_loss (float): The validation loss of the model.
        """
        lr = float(X[:, 0])
        units = int(X[:, 1])
        dropout = float(X[:, 2])
        l2_reg = float(X[:, 3])
        batch_size = int(X[:, 4])

        input_shape = 10  # Replace with the actual input shape of your data

        model = self.build_model(lr, units, dropout, l2_reg, input_shape)

        # Example of training with dummy data, replace with your actual
        # training and validation data
        X_train = np.random.rand(1000, input_shape)
        y_train = np.random.rand(1000, 1)
        X_val = np.random.rand(200, input_shape)
        y_val = np.random.rand(200, 1)

        early_stopping = \
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=5,
                                             restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping], verbose=0)

        validation_loss = history.history['val_loss'][-1]

        self.save_best_model(model, lr, units, dropout, l2_reg, batch_size)

        return validation_loss

    def save_best_model(self, model, lr, units, dropout, l2_reg, batch_size):
        """
        Saves the current best model checkpoint.

        Args:
            model: Trained Keras model.
            lr (float): Learning rate used in the model.
            units (int): Number of units in the hidden layer.
            dropout (float): Dropout rate.
            l2_reg (float): L2 regularization weight.
            batch_size (int): Batch size.
        """
        filename = (
            f"model_checkpoints/model_lr{lr}_units{units}_dropout{dropout}_"
            f"l2{l2_reg}_batch{batch_size}.h5"
        )
        model.save(filename)

    def run_optimization(self):
        """
        Runs the Bayesian Optimization process.

        Saves the best hyperparameters, the corresponding performance, and
        generates the convergence plot.
        """
        opt = GPyOpt.methods.BayesianOptimization(f=self.objective_function,
                                                  domain=self.bounds,
                                                  acquisition_type='EI',
                                                  exact_feval=True)
        opt.run_optimization(max_iter=30)

        best_parameters = opt.X[np.argmin(opt.Y)]
        best_performance = np.min(opt.Y)

        print(f"Best hyperparameters found: {best_parameters}")
        print(f"Best performance: {best_performance}")

        with open('bayes_opt.txt', 'w') as f:
            f.write(
                f"Best Hyperparameters:\n"
                f"Learning Rate: {best_parameters[0]}\n"
                f"Units: {best_parameters[1]}\n"
                f"Dropout: {best_parameters[2]}\n"
                f"L2: {best_parameters[3]}\n"
                f"Batch Size: {best_parameters[4]}\n"
            )
            f.write(f"Best Metric: {best_performance}\n")

        plt.plot(opt.Y_best)
        plt.xlabel('Iteration')
        plt.ylabel('Best Metric')
        plt.title('Convergence Plot')
        plt.savefig('convergence_plot.png')
        plt.show()


if __name__ == "__main__":
    optimizer = BayesianOptimizationModel()
    optimizer.run_optimization()
