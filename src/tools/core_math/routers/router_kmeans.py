from fastapi import APIRouter, HTTPException
import numpy as np

from core_math.ml.neighbors.kmeans import KMeans
from ..models.schema_kmeans import KmeansInput, KmeansOutput

router = APIRouter(prefix="/kmeans", tags=["kmeans"])


@router.post("/fit", response_model=KmeansOutput)
def run_kmeans(payload: KmeansInput) -> KmeansOutput:
    distance = "euclidean" if payload.distance == "euclidian" else payload.distance

    try:
        model = KMeans(
            n_clusters=payload.n_clusters,
            max_iter=payload.max_iter,
            tol=payload.tol,
            init=payload.init,
            distance=distance,
            n_init=payload.n_init,
            random_state=payload.random_state,
        )

        X = np.asarray(payload.data, dtype=float)
        model.fit(X)

        return KmeansOutput(
            centroids=model.cluster_centers_.tolist(),
            labels=model.labels_.tolist(),
            inertia=model.inertia_,
            n_iter=model.n_iter_,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

