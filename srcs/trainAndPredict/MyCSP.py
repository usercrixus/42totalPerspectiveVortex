from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.linalg import eigh

class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        # Ensure shapes and labels
        X = np.asarray(X)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim != 3:
            raise ValueError(f"X must be (trials (labels try), channels (electrods), samples (frequency * time)), got {X.shape}")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP implementation supports exactly 2 classes.")

        covs = []
        for c in classes:
            Xc = X[y == c]
            cov = np.mean([self._cov(trial) for trial in Xc], axis=0)
            covs.append(cov)

        C1, C2 = covs # average cov of each channel for the label [n_channel][n_channel] (symetric)
        # Generalized eigenvalue problem
        # eigvecs → columns are the corresponding eigenvectors (spatial filters)
        # eigvals → array of eigenvalues (λ values, one per filter)
        # Large λλ → high variance for class 1, low for class 2
        # Small λλ → high variance for class 2, low for class 1
        eigvals, eigvecs = eigh(C1, C1 + C2) 

        # Sort by eigenvalues descending
        ix = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, ix]

        # Keep first and last components
        self.n_components = min(self.n_components, eigvecs.shape[1])
        half = self.n_components // 2
        self.filters_ = np.hstack([eigvecs[:, :half], eigvecs[:, -half:]]) # shape [n_filter][n_channel]
        return self

    def transform(self, X):
        X = np.asarray(X)
        feats = []
        for trial in X:
            projected = self.filters_.T @ trial # one filter per channel matrix [n_filter][n_sample]
            var = np.var(projected, axis=1) # vector [n_filter] representing the variance of each sequence
            # This vector says:
            # CSP component 1’s signal had variance ≈ 0.916 for this trial
            # CSP component 2’s signal had variance ≈ 0.555 for this trial
            # Those variances are the features that tell the classifier:
            # For left-hand imagery → certain CSP components have high variance
            # For right-hand imagery → those same components have low variance, and vice versa.
            feats.append(np.log(var / np.sum(var)))
        return np.array(feats) # [n_trial][n_filter]

    def _cov(self, X):
        X = X - X.mean(axis=1, keepdims=True)
        return (X @ X.T) / np.trace(X @ X.T)
