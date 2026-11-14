
"""
Initial Requirements
!pip install faiss-cpu

For e-means visit: 
# Source: https://github.com/facebookresearch/faiss/tree/main/contrib/e_means
# Arxiv: https://arxiv.org/pdf/2611.09501v1#page=3.18
"""

import os
import time
import numpy as np
import pandas as pd
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from e_means.compute_inertia import compute_innertia

output_dir = "/results"

# We'll be testing k-means, k-means++, e-means on varying dimension, clusters and number of samples
D = [64, 128, 256]
K = [10, 20, 30, 40, 50]
N_samples = [200_000, 400_000, 600_000, 800_000, 1_000_000]


def experimentation(D, K, N_samples):
    results = []  # For storing results
    for n_samples in N_samples:
        for d in D:
            for k in K:
                X, _ = make_blobs(n_samples=n_samples, n_features=d,
                                  centers=k, cluster_std=0.60, random_state=42)
                X_float = X.astype("float32")  # For E-Means input

                print("\nRunning Standard K-Means (Random Init)...")
                start_time = time.time()
                kmeans_rand = KMeans(n_clusters=k, init="random",
                                     n_init=10, random_state=42)
                kmeans_rand.fit(X)
                time_rand = time.time() - start_time
                inertia_rand = kmeans_rand.inertia_
                print(
                    f"K-Means | Time: {time_rand:.4f} seconds | Inertia: {inertia_rand:,.2f}")
                results.append({
                    "n_samples": n_samples,
                    "dimension": d,
                    "k_clusters": k,
                    "algorithm": "K-Means",
                    "time_seconds": time_rand,
                    "inertia": inertia_rand
                })

                print("\nRunning K-Means++")
                start_time = time.time()
                kmeans_pp = KMeans(n_clusters=k, init="k-means++",
                                   n_init=10, random_state=42)
                kmeans_pp.fit(X)
                time_pp = time.time() - start_time
                inertia_pp = kmeans_pp.inertia_
                print(
                    f"K-Means++ | Time: {time_pp:.4f} seconds | Inertia: {inertia_pp:,.2f}")
                results.append({
                    "n_samples": n_samples,
                    "dimension": d,
                    "k_clusters": k,
                    "algorithm": "K-Means++",
                    "time_seconds": time_pp,
                    "inertia": inertia_pp
                })

                print("\nRunning E-Means...")
                start_time = time.time()
                em = NumpyEmeans(d, k)
                centroids_em, labels_em = em.train(X_float)
                time_em = time.time() - start_time
                inertia_em = compute_innertia(X_float, labels_em, centroids_em)
                print(
                    f"E-Means | Time: {time_em:.4f} seconds | Innertia: {inertia_em:,.2f}")
                results.append({
                    "n_samples": n_samples,
                    "dimension": d,
                    "k_clusters": k,
                    "algorithm": "E-Means",
                    "time_seconds": time_em,
                    "inertia": inertia_em
                })

                print("\n" + "="*30)
                print(
                    f"SUMMARY for {n_samples} samples, {d} dimensions, {k} clusters")
                print("="*30)
                print(
                    f"K-Means (Random) | Time : {time_rand:.4f} s | Innertia : {inertia_rand:,.2f}")
                print(
                    f"K-Means++ | Time :       {time_pp:.4f} s | Innertia : {inertia_pp:,.2f}")
                print(
                    f"E-Means | Time :         {time_em:.4f} s | Innertia : {inertia_em:,.2f}")

    # Saving all the results into .csv file
    os.makedirs(output_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    csv_filename = os.path.join(
        output_dir, f"clock_reconstruction_results_{d}.csv")
    df_results.to_csv(csv_filename, index=False)
    print(f"\nBenchmark results successfully saved to {csv_filename}!")


if __name__ == "__main__":
    experimentation(D, K, N_samples)
