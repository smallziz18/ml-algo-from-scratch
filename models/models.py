import numpy as np

# =============================
# 1. LINEAR REGRESSION
# =============================
class MyLinearRegression:
    """
    Simple Linear Regression using the Normal Equation.
    Solves w in X^T X w = X^T y.
    """
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the linear model.
        Parameters:
            X : np.ndarray, shape (n_samples, n_features)
            y : np.ndarray, shape (n_samples,)
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.linalg.solve(X.T @ X, X.T @ y)
        return self

    def predict(self, X):
        """
        Predict using the linear model.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.w


# =============================
# 2. RIDGE REGRESSION
# =============================
class MyRidge:
    """
    Ridge Regression (L2 regularization).
    Solves w in (X^T X + alpha*I) w = X^T y.
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.w = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_features = X.shape[1]
        self.w = np.linalg.solve(X.T @ X + self.alpha * np.eye(n_features), X.T @ y)
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.w


# =============================
# 3. LOGISTIC REGRESSION
# =============================
def sigmoid(z):
    """
    Compute the sigmoid function with clipping for numerical stability.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class MyLogisticRegression:
    """
    Logistic Regression using Gradient Descent.
    """
    def __init__(self, lr=0.01, n_iter=1000, fit_intercept=True):
        self.w = None
        self.lr = lr
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            y_pred = sigmoid(X @ self.w)
            gradient = X.T @ (y_pred - y) / X.shape[0]
            self.w -= self.lr * gradient
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return (sigmoid(X @ self.w) >= 0.5).astype(int)


# =============================
# 4. K-MEANS
# =============================
class MyKMeans:
    """
    K-Means clustering algorithm.
    """
    def __init__(self, n_clusters=3, max_iter=100, n_init=10, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.centers = None

    def fit(self, X):
        best_inertia = np.inf
        best_centers = None

        for _ in range(self.n_init):
            # Random initialization
            idx = np.random.choice(len(X), self.n_clusters, replace=False)
            centers = X[idx]
            np.zeros(len(X), dtype=int)
            for _ in range(self.max_iter):
                distances = np.linalg.norm(X[:, None] - centers, axis=2)
                labels = np.argmin(distances, axis=1)

                new_centers = np.array([
                    X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                    for k in range(self.n_clusters)
                ])

                # Check for convergence
                if np.all(np.linalg.norm(new_centers - centers, axis=1) < self.tol):
                    break
                centers = new_centers

            # Ensure labels correspond to the final centers before computing inertia
            labels = np.argmin(np.linalg.norm(X[:, None] - centers, axis=2), axis=1)

            # Compute inertia (sum of squared distances)
            inertia = np.sum((X - centers[labels])**2)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers

        self.centers = best_centers
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centers, axis=2)
        return np.argmin(distances, axis=1)


# =============================
# 5. PCA
# =============================
class MyPCA:
    """
    Principal Component Analysis for dimensionality reduction.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components].T
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components.T + self.mean


# =============================
# 6. K-NEAREST NEIGHBORS
# =============================
class MyKNN:
    """
    K-Nearest Neighbors classifier.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = distances.argsort()[:self.k]
            k_labels = self.y_train[k_idx]
            predictions.append(np.bincount(k_labels).argmax())
        return np.array(predictions)
