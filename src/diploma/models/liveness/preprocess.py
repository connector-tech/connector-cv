import cv2
import numpy as np


def preprocess(
    img: np.ndarray,
    rgb: bool = False,
    imsize: int = 300,
    normalize: bool = True,
    mean=np.array([0.485, 0.456, 0.406]),
    std=np.array([0.229, 0.224, 0.225]),
) -> np.ndarray:
    """
    Preprocess an input image for neural network inference.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        rgb (bool, optional): Whether the image is in RGB format. Defaults to False.
        imsize (int, optional): Size to which the input image is resized.
            Defaults to 300.
        normalize (bool, optional): Whether to normalize the input image.
            Defaults to True.
        mean (np.ndarray, optional): Mean value for normalization.
            Defaults to np.array([0.485, 0.456, 0.406]).
        std (np.ndarray, optional): Standard deviation values for normalizations.
            Defaults to np.array([0.229, 0.224, 0.225]).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    if not rgb:
        img = img[:, :, ::-1]

    img = cv2.resize(img, (imsize, imsize))
    if normalize:
        img = img - img.min()
        img = img / img.max()
        img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))[None, ...]
    return img
