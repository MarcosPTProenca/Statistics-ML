import pytest 

from src.core_math.ml.distances import euclidean_distance, manhattan_distance, cosine_distance

EUCLIDEAN_VECTORS = {
    5.0:  ((0.0, 0.0), (3.0, 4.0)),
    0.0:  ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    2.0:  ((1.0, 0.0), (-1.0, 0.0)),
    2.23606797749979: ((1.0, 2.0), (2.0, 4.0)),
}

SQUARED_EUCLIDEAN_VECTORS = {
    25.0: ((0.0, 0.0), (3.0, 4.0)),
    0.0:  ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    4.0:  ((1.0, 0.0), (-1.0, 0.0)),
    5.0:  ((1.0, 2.0), (2.0, 4.0)),
}

MANHATTAN_VECTORS = {
    7.0: ((0.0, 0.0), (3.0, 4.0)),
    0.0: ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    2.0: ((1.0, 0.0), (-1.0, 0.0)),
    3.0: ((1.0, 2.0), (2.0, 4.0)),
}

COSINE_VECTORS = {
    1.0: ((1.0, 0.0), (0.0, 1.0)),
    0.0: ((1.0, 2.0), (2.0, 4.0)),
    2.0: ((1.0, 0.0), (-1.0, 0.0)),
}

COSINE_RAISES_ZERO_NORM = (
    ((0.0, 0.0), (3.0, 4.0)),
    ((), ()),
)

def test_euclidean():
    for expected, (x, y) in EUCLIDEAN_VECTORS.items():
        assert euclidean_distance(x, y) == expected

def test_cosine():
    for expected, (x, y) in COSINE_VECTORS.items():
        assert cosine_distance(x, y) == pytest.approx(expected)
    
    for (x, y) in COSINE_RAISES_ZERO_NORM:
        with pytest.raises(ZeroDivisionError):
            cosine_distance(x, y)

def test_manhattan():
    for expected, (x, y) in MANHATTAN_VECTORS.items():
        assert manhattan_distance(x, y) == expected
