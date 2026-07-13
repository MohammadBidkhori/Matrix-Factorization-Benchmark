# In the name of Allah
import numpy as np
import matplotlib.pyplot as plt

class MatrixFactorization:
    """
    Empirical Benchmark of Optimization Techniques for Matrix Factorization:
    Alternating Least Squares (ALS) vs. Stochastic Gradient Descent (SGD)
    """
    def __init__(self, Matrix, epochs=100, min_value=1, max_value=5, learning_rate=0.01, num_features=3, lambda_reg=0.1):
        self.Matrix = Matrix
        self.num_users, self.num_items = Matrix.shape
        self.num_features = num_features
        self.lambda_reg = lambda_reg  # L2 Regularization penalty

        # Initialize QMatrix (Users) and PMatrix (Items) with random uniform values
        np.random.seed(42) # Seed for reproducibility
        self.QMatrix = np.random.uniform(min_value, max_value, size=(self.num_users, self.num_features))
        self.PMatrix = np.random.uniform(min_value, max_value, size=(self.num_features, self.num_items))

        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Lists to store the loss history for plotting
        self.loss_history = []

    def train_SGD(self):
        """
        Optimizes the latent features using Stochastic Gradient Descent.
        Time Complexity per epoch: O(N * features) where N is non-zero elements.
        """
        self.loss_history = [] # Reset loss
        for epoch in range(self.epochs):
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if self.Matrix[i, j] != 0:
                        # 1. Compute prediction and error
                        prediction = np.dot(self.QMatrix[i, :], self.PMatrix[:, j])
                        error = self.Matrix[i, j] - prediction
                        
                        # 2. Update latent factors simultaneously (with regularization)
                        Q_i_temp = self.QMatrix[i, :].copy()
                        self.QMatrix[i, :] += self.learning_rate * (error * self.PMatrix[:, j] - self.lambda_reg * self.QMatrix[i, :])
                        self.PMatrix[:, j] += self.learning_rate * (error * Q_i_temp - self.lambda_reg * self.PMatrix[:, j])
                        
            self.loss_history.append(self._compute_loss())
        return self.QMatrix, self.PMatrix, self.loss_history

    def train_ALS(self):
        """
        Optimizes the latent features using Alternating Least Squares.
        Converts the non-convex optimization into a series of convex linear least squares problems.
        Uses np.linalg.solve for numerical stability instead of direct matrix inversion.
        """
        self.loss_history = [] # Reset loss
        
        # Identity matrix for L2 Regularization (lambda * I)
        I_Q = self.lambda_reg * np.eye(self.num_features)
        I_P = self.lambda_reg * np.eye(self.num_features)

        for epoch in range(self.epochs):
            
            # Step 1: Fix PMatrix (Items), Optimize QMatrix (Users)
            for i in range(self.num_users):
                # Only consider items that have been rated by user i
                rated_items_idx = np.where(self.Matrix[i, :] > 0)[0]
                if len(rated_items_idx) > 0:
                    P_subset = self.PMatrix[:, rated_items_idx] # Shape: (features, num_rated)
                    R_i = self.Matrix[i, rated_items_idx]       # Shape: (num_rated,)
                    
                    # Formula: Q_i = R_i * P^T * (P * P^T + lambda*I)^-1
                    # A * x = b  ==>  x = solve(A, b)
                    A = np.dot(P_subset, P_subset.T) + I_Q
                    b = np.dot(P_subset, R_i.T)
                    
                    self.QMatrix[i, :] = np.linalg.solve(A, b)

            # Step 2: Fix QMatrix (Users), Optimize PMatrix (Items)
            for j in range(self.num_items):
                # Only consider users who have rated item j
                rated_users_idx = np.where(self.Matrix[:, j] > 0)[0]
                if len(rated_users_idx) > 0:
                    Q_subset = self.QMatrix[rated_users_idx, :] # Shape: (num_rated, features)
                    R_j = self.Matrix[rated_users_idx, j]       # Shape: (num_rated,)
                    
                    # Formula: P_j = (Q^T * Q + lambda*I)^-1 * Q^T * R_j
                    A = np.dot(Q_subset.T, Q_subset) + I_P
                    b = np.dot(Q_subset.T, R_j)
                    
                    self.PMatrix[:, j] = np.linalg.solve(A, b)

            self.loss_history.append(self._compute_loss())
        return self.QMatrix, self.PMatrix, self.loss_history

    def _compute_loss(self):
        """
        Computes the Mean Squared Error (MSE) only for the observed ratings.
        """
        loss = 0
        count = 0
        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.Matrix[i, j] != 0:
                    prediction = np.dot(self.QMatrix[i, :], self.PMatrix[:, j])
                    loss += (self.Matrix[i, j] - prediction) ** 2
                    count += 1
        return loss / count if count > 0 else 0


# ==========================================
# Benchmark Execution & Visualization
# ==========================================

# 1. Dataset setup
Matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# 2. Run SGD Model
print("Running SGD Optimization...")
model_sgd = MatrixFactorization(Matrix, learning_rate=0.01, epochs=200, num_features=2, lambda_reg=0.05)
_, _, loss_sgd = model_sgd.train_SGD()

# 3. Run ALS Model (Notice it needs way fewer epochs to converge)
print("Running ALS Optimization...")
# Resetting the model to ensure fair comparison from scratch
model_als = MatrixFactorization(Matrix, learning_rate=0.01, epochs=20, num_features=2, lambda_reg=0.05)
_, _, loss_als = model_als.train_ALS()

# 4. Plotting the Convergence (The "Winning Plot")
plt.figure(figsize=(10, 5))

# We plot SGD on a secondary x-axis or just limit the view, as SGD takes more epochs
plt.plot(range(1, len(loss_sgd) + 1), loss_sgd, label='SGD Convergence', color='blue', linestyle='--')
plt.plot(range(1, len(loss_als) + 1), loss_als, label='ALS Convergence', color='red', linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Convergence Speed: Alternating Least Squares vs Stochastic Gradient Descent')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(0, max(len(loss_als), len(loss_sgd) // 4)) # Zoom in to see the ALS drop clearly

# Show the plot
plt.tight_layout()
plt.show()

# 5. Final Display
print("\n--- Final Predictions (ALS) ---")
final_prediction = np.dot(model_als.QMatrix, model_als.PMatrix)
print(np.round(final_prediction, decimals=1))
