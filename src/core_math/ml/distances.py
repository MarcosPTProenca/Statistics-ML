import numpy as np
import math
from typing import Iterable

def euclidean_distance(x: Iterable, y: Iterable) -> float:
    """
    Compute the Euclidean distance between two vectors.

    d(x, y) = sqrt( Σ (xi - yi)^2 )
    
    Args:
        x: Iterable
        y: Iterable
    
    Return:
        euclidean distance (float)
    """     
    return math.sqrt(math.fsum((xi-yi)**2 for xi, yi in zip(x,y)))

def squared_euclidean_distance(x: Iterable, y: Iterable) -> float:
    """
    Compute the Squared Euclidean distance between two vectors.

    d(x, y) = Σ (xi - yi)^2
    
    Args:
        x: Iterable
        y: Iterable
    
    Return:
        Squared euclidean distance (float)
    """   
    return math.fsum((xi-yi)**2 for xi, yi in zip(x,y))

def cosine_distance(x: Iterable, y: Iterable) -> float:
    """
    Compute the Cosine distance between two vectors.

    Cosine distance = 1 - (Σ xi*yi) / (sqrt(Σ xi^2) * sqrt(Σ yi^2))

    Args:
        x: Iterable
        y: Iterable

    Returns:
        Cosine distance (float)
    """
    return 1 - (math.fsum(xi*yi for xi, yi in zip(x,y))
                / (math.sqrt(math.fsum(xi**2 for xi in x))
                    * math.sqrt(math.fsum(yi**2 for yi in y))))

def manhattan_distance(x: Iterable, y: Iterable) -> float:
    """
    Compute the Manhattan distance between two vectors.

    Manhattan distance = Σ |xi - yi|

    Args:
        x: Iterable
        y: Iterable

    Returns:
        Manhattan distance (float)
    """
    return math.fsum(abs(xi-yi) for xi, yi in zip(x,y))    

if __name__ == '__main__':

    x = np.array([1, 2, 3, 4])
    y = np.array([1, 3, 5, 4])

    print("squared euclidean", squared_euclidean_distance(x, y))
    print("euclidean", euclidean_distance(x, y))
    print("Cosine", cosine_distance(x, y))
    print("Manhattan", manhattan_distance(x, y))