import random
from typing import Literal, Callable, Iterable
import math
import numpy as np
from ..distances import squared_euclidean_distance, \
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

        n_samples: int = X.shape[0]
        k: int = self.n_clusters
        
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
                
                if next_idx is None:
                    remaining = [i for i in range(n_samples) if i not in chosen_indices]
                    next_idx = random.choice(remaining)

                chosen_indices.add(next_idx)
                centroids.append(X[next_idx])

            return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assignment step.

        For each point x_i, assign the closest centroid:

            c_i = argmin_j || x_i - μ_j ||²

        Produces the label vector:
            labels ∈ {0, ..., k-1}
        """
        n_samples: int = X.shape[0]
        k: int = centroids.shape[0]
        labels = np.ones(n_samples, dtype=int) * (-1) 
        
        for i in range(n_samples):
            d_min = float("inf")

            for j in range(k):

                d = self._distance_fn(X[i], centroids[j])

                if d < d_min:
                    d_min = d
                    labels[i] = int(j)

        return labels            

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update step.

        Recompute each centroid as the mean of its assigned points:

            μ_j = (1 / |C_j|) Σ_{x_i ∈ C_j} x_i

        Handle empty clusters appropriately.
        """
        n_samples: int = X.shape[0]
        n_features: int = X.shape[1]
        k: int = self.n_clusters
        new_centroids = np.zeros((k, n_features))
        counts = np.zeros(k, dtype=int)

        for i in range(n_samples):

            cluster = int(labels[i])

            new_centroids[cluster] += X[i]
            counts[cluster] += 1
        
        for cluster in range(k):

            if counts[cluster] == 0:
                random_x = random.randint(0, n_samples-1)
                new_centroids[cluster] = X[random_x]

            else:
                new_centroids[cluster] = new_centroids[cluster]/counts[cluster]
        
        return new_centroids

    def _compute_inertia(self, X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the objective function (WCSS):

            inertia = Σ || x_i - μ_{c_i} ||²

        Used to select the best run among n_init trials.
        """
        n_samples: int = X.shape[0]
        error: float = 0.0

        for i in range(n_samples):

            ci = int(labels[i])
            centroid = centroids[ci]
            error += self._distance_fn(X[i], centroid)

        return error

    def _has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """
        Check convergence using centroid displacement:

            max(|| μ_new - μ_old ||) < tol
        """
        k = self.n_clusters
        max_delta = 0.0

        for i in range(k):
            delta = math.sqrt(self._distance_fn(old_centroids[i], new_centroids[i]))
            if delta > max_delta:
                max_delta = delta
        
        return max_delta < self.tol

    def fit(self, X: np.ndarray):
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
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if X.shape[0] < 1:
            raise ValueError("X must have at least 1 sample")
        if self.n_clusters > X.shape[0]:
            raise ValueError("The number of clusters can the greater than the number of samples")

        best_inertia = float("inf")
        best_n_iter = 0
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids = np.asarray(self._initialize_centroids(X))
            iterations = 0

            while True:
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                converge = self._has_converged(centroids, new_centroids)

                iterations += 1

                if converge or iterations >= self.max_iter:
                    inertia = self._compute_inertia(X, new_centroids, labels)

                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_n_iter = iterations
                        best_centroids = new_centroids
                        best_labels = labels
                    break

                centroids = new_centroids

        self.cluster_centers_ = np.asarray(best_centroids)
        self.labels_ = np.asarray(best_labels, dtype=int)
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign clusters to new samples.

        Rule:
            c_i = argmin_j || x_i - μ_j ||²

        Requires fitted centroids.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted")
        if X.shape[0] < 1:
            raise ValueError("X must contain at least one sample")
        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("The number of features is different from the number of trained features")
        
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X: np.ndarray):
        """
        Equivalent to calling:

            fit(X)
            return labels_

        Provided for convenience.
        """

        self.fit(X)

        return self.labels_


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    # Criando 3 clusters bem separados
    cluster_1 = rng.normal(loc=(0, 0), scale=0.5, size=(100, 2))
    cluster_2 = rng.normal(loc=(5, 5), scale=0.5, size=(100, 2))
    cluster_3 = rng.normal(loc=(10, 0), scale=0.5, size=(100, 2))

    X = np.vstack([cluster_1, cluster_2, cluster_3])

    kmeans = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42
    )

    print("Fitting model...")
    kmeans.fit(X)

    print("\nCentroids:")
    print(kmeans.cluster_centers_)

    print("\nInertia:")
    print(kmeans.inertia_)

    print("\nIterations:")
    print(kmeans.n_iter_)

    new_points = np.array([
        [0, 0],
        [5, 5],
        [10, 0],
        [8, 1]
    ])

    preds = kmeans.predict(new_points)

    print("\nNew points predictions:")
    for point, label in zip(new_points, preds):
        print(f"Point {point} -> Cluster {label}")
