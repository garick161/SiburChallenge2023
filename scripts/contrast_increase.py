import cv2
import numpy as np


def contrast_increase(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    frame_new = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame_new