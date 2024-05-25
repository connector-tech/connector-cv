from typing import Iterable, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from code.utils import apply_margin


def draw_landmarks(
    frame: np.ndarray,
    landmarks: List[List[float]],
    override: bool = False,
    text: Optional[str] = None,
    color: Optional[Iterable[int]] = (0, 0, 255),
    radius: int = 1,
    thickness: float = 3,
) -> np.ndarray:
    """
    Draw landmarks on an image.
    Landmarks are [left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner].
    Each landmark is drawn as circle.

    Args:
        frame (np.ndarray): Original frame.
        landmarks (List[List[float]]): List of landmarks.
        override (bool, optional): Save changes on an image. Defaults to False.
        text (Optional[str], optional): Adds text on image. Defaults to None.
        color (Optional[Iterable[int]], optional): Color of landmarks. Defaults to (0, 0, 255).
        radius (int, optional): Circle radius. Defaults to 1.
        thickness (float, optional): Thickness of circle. Defaults to 3.

    Returns:
        np.ndarray: Image with landmarks drawn.
    """
    work_frame = frame
    if not override:
        work_frame = frame.copy()
    if text:
        cv2.putText(
            work_frame,
            text,
            (int(landmarks[0][0][0]), int(landmarks[0][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
    for landmark in landmarks:
        for i in landmark:
            work_frame = cv2.circle(
                work_frame,
                (i[0].astype(int), i[1].astype(int)),
                radius=radius,
                color=color,
                thickness=thickness,
            )

    return work_frame


def draw_rectangles(
    frame: np.ndarray,
    bboxes: List[List[float]],
    margin: Optional[int] = None,
    override: bool = False,
    color: Optional[Iterable[int]] = (0, 0, 255),
    thickness: float = 1,
) -> np.ndarray:
    """
    Draws bbox on image. Bbox points are [x1, y1, x2, y2].

    Args:
        frame (np.ndarray): Original frame.
        bboxes (List[List[float]]): [x1, y1, x2, y2].
        margin (Optional[int], optional): Applies margin to bbox. Defaults to None.
        override (bool, optional): Saves changes on an image. Defaults to False.
        color (Optional[Iterable[int]], optional): Color of rectangle. Defaults to (0, 0, 255).
        thickness (float, optional): Thickness of rectangle. Defaults to 3.

    Returns:
        np.ndarray: Image with bbox drawn.
    """
    work_frame = frame
    if not override:
        work_frame = frame.copy()
    for box in bboxes:
        b = list(map(int, box))
        if margin is not None:
            b = apply_margin(work_frame, b, margin)
        cv2.rectangle(work_frame, (b[0], b[1]), (b[2], b[3]), color, thickness)
    return work_frame


def draw_images(
    images: List[Union[str, np.ndarray]],
    image_num: int = 64,
    start_index: Optional[int] = 0,
    figsize: Union[int, Tuple[int, int]] = 10,
    read_images: bool = True,
    spacing: Tuple[float, float] = (0.2, 0.2),
) -> None:
    """
    Draws list of images in one matplotlib figure.

    Args:
        images (List[Union[str, np.ndarray]]): A list of image paths or numpy arrays.
        image_num (int, optional): Number of images to display. Defaults to 64.
        start_index (Optional[int], optional): Index of the first image to display. Defaults to 0.
        figsize (Union[int, Tuple[int, int]], optional): Figure size. Defaults to 10.
        read_images (bool, optional): Whether to read images from paths. Defaults to True.
        spacing (Tuple[float, float], optional): Vertical and horizontal spacing between
            subplots. Defaults to (0.2, 0.2).
    """
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    plt.figure(figsize=figsize)

    num_images = len(images)
    image_num = min(image_num, num_images - start_index)

    rows = int(np.ceil(np.sqrt(image_num)))
    cols = int(np.ceil(np.sqrt(image_num)))

    for i in range(image_num):
        plt.subplots_adjust(hspace=spacing[0], wspace=spacing[1])
        plt.subplot(rows, cols, i + 1)
        if read_images:
            image = cv2.imread(images[i + start_index])[..., ::-1]
        else:
            image = images[i + start_index][..., ::-1]
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel(i + start_index)
