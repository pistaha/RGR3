from dataclasses import dataclass

import numpy as np


@dataclass
class ModelResult:
    name: str
    equation: str
    y_hat: np.ndarray
    residuals: np.ndarray
    rss: float
    r2: float
    approx_error: float
    coefficients: dict
