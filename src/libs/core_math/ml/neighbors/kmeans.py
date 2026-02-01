import random
from typing import Literal, Callable, Iterable
import math
import numpy as np
from libs.core_math.ml.distances import squared_euclidean_distance, \
                                        cosine_distance, \
                                        manhattan_distance, \
                                        euclidean_distance

DistanceFn = Callable[[Iterable, Iterable], float]

class KMeans:
    """
    K-Means clustering algorithm.

    Objective:
        Minimize the Within-Cluster Sum of Squares (WCSS):

            J = Σ_{j=1}^{k} Σ_{x_i ∈ C_j} || x_i - μ_j ||²

    where:
        μ_j = centroid of cluster j
        C_j = set of points assigned to cluster j
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: Literal["random", "k-means++"] = "random",
        distance: Literal["squared_euclidean", "cosine", "manhattan", "euclidian"] = "squared_euclidean",
        n_init: int = 10,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters (k).

        max_iter : int
            Maximum number of iterations allowed for a single run.

        tol : float
            Convergence threshold based on centroid displacement:

                || μ^(t) - μ^(t-1) || < tol

        init : str
            Initialization strategy.
            Options typically include:
                - "random"
                - "k-means++"

        n_init : int
            Number of independent runs with different centroid seeds.
            The best solution is selected using the lowest inertia.

        random_state : int | None
            Seed for reproducibility.
        """

        # Hyperparameters
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.distance = distance
    
        # Learned attributes (set during fit)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

        registry: dict[str, DistanceFn] = {
            "squared_euclidean": squared_euclidean_distance,
            "cosine": cosine_distance,
            "manhattan": manhattan_distance,
            "euclidean": euclidean_distance,
        }

        try:
            self._distance_fn: DistanceFn = registry[self.distance]
        except:
            raise ValueError(f"Unkown distance metric: {self.distance}")

    def _initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids.

        Random initialization:
            Select k samples from X.

        K-Means++ initialization:
            1. Choose first centroid uniformly at random.
            2. For each remaining centroid, sample with probability:

                p(x) = D(x)² / Σ D(x)²

            where D(x) is the distance from x to the nearest centroid.
        """

        n_samples = X.shape[0]
        k = self.n_clusters
        
        if self.init == "random":
            
            chosen = random.sample(range(n_samples), k)
            return X[chosen]
        
        elif self.init == "k-means++":
            
            first_idx = random.randrange(n_samples)
            chosen_indices = {first_idx}
            centroids = [X[first_idx]]

            distances = [float("inf")] * n_samples

            while len(centroids) < k:
                last_centroid = centroids[-1]

                for i in range(n_samples):
                    
                    if i in chosen_indices:
                        distances[i] = 0
                        continue

                    d = self._distance_fn(X[i], last_centroid)
                    if d < distances[i]:
                        distances[i] = d
                
                S = math.fsum(distances)

                if S == 0:
                    remaining = [i for i in range(n_samples) if i not in chosen_indices]
                    next_idx = random.choice(remaining)
                    chosen_indices.add(next_idx)
                    centroids.append(X[next_idx])
                    continue
                
                p = [d / S for d in distances]
                u = random.random()
                cum = 0
                next_idx = None

                for i in range(n_samples):

                    cum += p[i]

                    if cum >= u:
                        next_idx = i
                        break
                
                chosen_indices.add(next_idx)
                centroids.append(X[next_idx])

            return centroids

    def _assign_clusters(self, X, centroids):
        """
        Assignment step.

        For each point x_i, assign the closest centroid:

            c_i = argmin_j || x_i - μ_j ||²

        Produces the label vector:
            labels ∈ {0, ..., k-1}
        """
        pass

    def _update_centroids(self, X, labels):
        """
        Update step.

        Recompute each centroid as the mean of its assigned points:

            μ_j = (1 / |C_j|) Σ_{x_i ∈ C_j} x_i

        Handle empty clusters appropriately.
        """
        pass

    def _compute_inertia(self, X, centroids, labels):
        """
        Compute the objective function (WCSS):

            inertia = Σ || x_i - μ_{c_i} ||²

        Used to select the best run among n_init trials.
        """
        pass

    def _has_converged(self, old_centroids, new_centroids):
        """
        Check convergence using centroid displacement:

            max(|| μ_new - μ_old ||) < tol
        """
        pass

    def fit(self, X):
        """
        Train the K-Means model.

        Algorithm:
            Repeat for n_init runs:
                1. Initialize centroids
                2. Alternate:
                    a) Assignment step
                    b) Update step
                3. Stop when convergence or max_iter is reached
                4. Compute inertia

            Keep the run with the lowest inertia.
        """
        pass

    def predict(self, X):
        """
        Assign clusters to new samples.

        Rule:
            c_i = argmin_j || x_i - μ_j ||²

        Requires fitted centroids.
        """
        pass

    def fit_predict(self, X):
        """
        Equivalent to calling:

            fit(X)
            return labels_

        Provided for convenience.
        """
        pass
