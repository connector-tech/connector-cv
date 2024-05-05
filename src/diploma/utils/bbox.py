from typing import List, Literal, Tuple, Union

import cv2
import numpy as np


def apply_margin(
    img_shape: tuple,
    box: List[float],
    margin: Union[int, float],
    margin_type: Literal["pixelwise", "percentage"] = "pixelwise",
) -> List[int]:
    """
    Applies margin to bbox to create bbox wider.
    New bbox is then clipped by provided img shape.

    Args:
        img_shape (tuple): shape of the original image (height, width).
        box (List[float]): [x1, y1, x2, y2].
        margin (Union[int, float]): margin value.
                                    If margin_type is 'pixelwise', this is in pixels.
                                    If margin type is in 'percentage', this is a float.
                                    (e.g., 0.1 for 10%)
        margin_type (str): Either 'pixelwise or 'percentage'. Defaults to "pixelwise".

    Returns:
        List[int]: New bbox with margin.
    """

    if margin_type == "pixelwise":
        dx = dy = margin
    elif margin_type == "percentage":
        dx = int((box[2] - box[0]) * margin)
        dy = int((box[3] - box[1]) * margin)
    else:
        raise ValueError(
            "Invalid margin_type. It should be either 'pixelwise' or 'percentage'."
        )
    return [
        int(max(box[0] - dx, 0)),
        int(max(box[1] - dy, 0)),
        int(min(box[2] + dx, img_shape.shape[1])),
        int(min(box[3] + dy, img_shape.shape[0])),
    ]


def preprocess_bbox(bbox: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding box coordinates to be within the valid frame boundaries.
    Between 0 and frame shape.

    Args:
        bbox (np.ndarray): Bounding box coordinates. Shape (n, 4) where n is
            the number of boxes.
        frame_size (Tuple[int, int]): Size of the frame in the format (height, width).

    Returns:
        np.ndarray: Bounding box coordinates after clipping.
    """
    bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, frame_size[1]).astype(int)
    bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, frame_size[0]).astype(int)
    return bbox


def area(bbox: np.ndarray) -> np.ndarray:
    """
    Computes area of bounding box.

    Args:
        bbox (np.ndarray): Numpy array with shape [N, 4] holding N boxes.

    Returns:
        np.ndarray: A numpy array with shape [N*1] representing box areas.
    """
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def crop_by_coordinates(
    image: Union[str, np.ndarray], coordinates: Union[List[float], np.ndarray]
) -> np.ndarray:
    """Crop a region from an image based on provided coordinates.

    Args:
        image (Union[str, np.ndarray]): Path to the image file or image array.
        coordinates (Union[List[float], np.ndarray]): Coordinates [x1, y1, x2, y2]
            of the cropping region.

    Returns:
        np.ndarray: Cropped region.
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    if isinstance(coordinates, str):
        coordinates = np.array(coordinates)

    x1, y1, x2, y2 = coordinates.astype(int)
    cropped_region = image[y1:y2, x1:x2]
    return cropped_region
