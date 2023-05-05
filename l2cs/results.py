from dataclasses import dataclass
import numpy as np

@dataclass
class GazeResultContainer:

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
