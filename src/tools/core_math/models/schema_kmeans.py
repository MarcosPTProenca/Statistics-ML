from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

class KmeansInput(BaseModel):
    data: list[list[float]]
    n_clusters: int = Field(..., ge=1)
    max_iter: int = Field(300, ge=1)
    tol: float = Field(1e-4, gt=0)
    init: Literal["random", "k-means++"] = "random"
    distance: Literal["squared_euclidean", "cosine", "manhattan", "euclidean", "euclidian"] = "squared_euclidean"
    n_init: int = Field(10, ge=1)
    random_state: int | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("data must not be empty")

        first_len = None
        cleaned: list[list[float]] = []

        for row in value:
            if row is None:
                raise ValueError("data rows must not be null")

            row_list = list(row)

            if len(row_list) == 0:
                raise ValueError("data rows must not be empty")

            if first_len is None:
                first_len = len(row_list)
            elif len(row_list) != first_len:
                raise ValueError("data rows must have the same length")

            for item in row_list:
                if not isinstance(item, (int, float)):
                    raise ValueError("data must contain only numeric values")

            cleaned.append(row_list)

        return cleaned

    @model_validator(mode="after")
    def validate_clusters(self) -> "KmeansInput":
        if self.n_clusters > len(self.data):
            raise ValueError("n_clusters cannot be greater than the number of samples")
        return self

class KmeansOutput(BaseModel):
    centroids: list[list[float]]
    labels: list[int]
    inertia: float
    n_iter: int
    
