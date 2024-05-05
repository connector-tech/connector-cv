from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.linalg import norm
from skimage import transform
from sklearn.preprocessing import normalize

from diploma.utils.bbox import apply_margin


def cosine_simularity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    normalize_vectors: bool = True,
    return_distance: bool = False,
) -> Tuple[float, float]:
    """
    Calculate the cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): The first embedding.
        embedding2 (np.ndarray): The second embedding.
        normalize_vectors (bool, optional): Whether to normalize the embeddings before
            calculation. Default is True.
        return_distance (bool, optional): Whether to calculate and return the Euclidean
            distance between the embeddings. Default is False.

    Returns:
        Tuple[float, float]: A tuple containing the similarity and distance scores.
            If return_distance is False, only the similarity score is returned.
    """
    if normalize_vectors:
        embedding1 = normalize(embedding1, axis=1)
        embedding2 = normalize(embedding2, axis=1)

    similarity = np.dot(embedding1, embedding2.T)

    if return_distance:
        distance = np.sum(np.square(embedding1 - embedding2))
        return similarity, distance

    return similarity


def align_face(
    original_img: np.ndarray,
    bbox: Iterable[Union[float, int]],
    landmarks: Iterable[Union[float, int]],
    image_size: Iterable[int] = (112, 112),
    method: str = "similar",
    margin: Optional[int] = None,
) -> np.ndarray:
    """
    Takes output of face detector (bounding box and landmark points for face)
    as input and generates aligned face images.

    Args:
        original_img (np.ndarray): Original frame from camera.
        bbox (Iterable[Union[float, int]]): Bounding box of face to align.
        landmarks (Iterable[Union[float, int]]): Landmarks of face to align.
        image_size (Iterable[int], optional): Output image size of aligned face.
            Defaults to (112, 112).
        method (str, optional): Name of transform to apply. Similar or affine.
            Defaults to "simular".
        margin (Optional[int], optional): Margin in pixels. Defaults to None.

    Returns:
        np.ndarray: Aligned face image.
    """
    assert len(landmarks) == 5 and all(
        len(point) == 2 for point in landmarks
    ), "Incorrect shape for landmarks"
    assert len(bbox) == 4, "Incorrect shape for bbox"
    if margin is None:
        margin = image_size[0]
    bigger_box = apply_margin(original_img, bbox, margin)
    img = original_img[
        int(bigger_box[1]) : int(bigger_box[3]),
        int(bigger_box[0]) : int(bigger_box[2]),
        :,
    ].copy()

    cropped_landmark = np.array(landmarks).copy()
    cropped_landmark[:, 0] = cropped_landmark[:, 0] - bigger_box[0]
    cropped_landmark[:, 1] = cropped_landmark[:, 1] - bigger_box[1]

    tform = (
        transform.AffineTransform()
        if method == "affine"
        else transform.SimilarityTransform()
    )
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.729904, 92.2041],
        ],
        dtype=np.int64,
    )

    x_scale = image_size[1] / 112
    y_scale = image_size[0] / 112
    src[:, 0] = x_scale * src[:, 0]
    src[:, 1] = y_scale * src[:, 1]

    tform.estimate(cropped_landmark.astype("int64"), src)
    ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)

    return (ndimage * 255).astype(np.uint8)


def estimate_yaw(landmarks: List[np.ndarray]) -> float:
    """
    Estimate the yaw of a face using facial landmarks.

    Args:
        landmark (List[np.ndarray]): List of facial landmarks as Numpy arrays.

    Returns:
        float: Estimated yaw value.
    """
    eye_left = np.array(landmarks[0])
    eye_right = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    mouth_left = np.array(landmarks[3])
    mouth_right = np.array(landmarks[4])

    vectA = mouth_left - eye_left
    vectB = nose - eye_left
    vectC = nose - eye_right
    vectD = mouth_right - eye_right

    unitA = vectA / norm(vectA)
    unitB = vectB / norm(vectB)
    unitC = vectC / norm(vectC)
    unitD = vectD / norm(vectD)

    AcrossB = np.clip(np.cross(unitA, unitB), -1.0, 1.0)
    CcrossD = np.clip(np.cross(unitC, unitD), -1.0, 1.0)
    AdotB = np.clip(np.dot(unitA, unitB), -1.0, 1.0)
    CdotD = np.clip(np.dot(unitC, unitD), -1.0, 1.0)

    alpha_sin = np.arcsin(AcrossB)
    theta_sin = np.arcsin(CcrossD)

    alpha_cos = np.arccos(AdotB)
    theta_cos = np.arccos(CdotD)

    if alpha_sin >= 0:
        return -1.0

    if theta_sin >= 0:
        return +1.0

    ratio_sin_alpha_theta = alpha_sin / theta_sin

    ratio_sin_theta_alpha = theta_sin / alpha_sin

    ratio = 1.0

    if alpha_cos >= theta_cos:
        ratio = ratio_sin_theta_alpha
        return 1 - ratio

    if alpha_cos < theta_cos:
        ratio = ratio_sin_alpha_theta
        return -(1 - ratio)


def estimate_pitch(landmarks: List[np.ndarray]) -> float:
    """
    Estimate the pitch of a face using facial landmarks.

    Args:
        landmarks (List[np.ndarray]): List of facial landmarks as Numpy arrays.

    Returns:
        float: Estimated pitch value.
    """
    eye_left = np.array(landmarks[0])
    eye_right = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    mouth_left = np.array(landmarks[3])
    mouth_right = np.array(landmarks[4])

    A = minimum_distance(eye_left, eye_right, nose)
    B = minimum_distance(mouth_left, mouth_right, nose)
    ratio = (A - B) / (A + B)

    return ratio


def minimum_distance(
    pointU: np.ndarray, pointV: np.ndarray, pointP: np.ndarray
) -> float:
    """
    Find the minimum distance between a point and a line segment defined by two points.

    Args:
        pointU (np.ndarray): First point of the line segment.
        pointV (np.ndarray): Second point of the line segment.
        pointP (np.ndarray): Point for which the distance is calculated.

    Returns:
        float: Minimum distance between the point and the line segment.
    """
    length2 = (pointU[0] - pointV[0]) ** 2 + (pointU[1] - pointV[1]) ** 2

    t = np.clip(np.dot(pointP - pointU, pointV - pointU) / length2, 0.0, 1.0)
    projectionPUV = pointU + t * (pointV - pointU)

    return np.linalg.norm(pointP - projectionPUV)


def estimate_blur(image: np.ndarray) -> float:
    """
    This function estimates blur using Laplacian filter.

    Args:
        image (np.ndarray): A np.ndarray BGR image.

    Returns:
        float: A value indicating the level of blur (higher for sharper images)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

    return laplacian
