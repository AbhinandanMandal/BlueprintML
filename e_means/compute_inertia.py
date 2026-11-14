
"""
Calculating the final objective function value (Sum of Square Errors)
for all three methods. Testing whether e-means matches or exceeds the 
clustring quality of k-means++ while using fraction of epochs
"""
import numpy as np


def compute_innertia(X, labels, centroids):
    assigned_centroids = centroids[labels]
    return np.sum((X - assigned_centroids)**2)
