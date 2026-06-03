"""Core functions for appearance-based gaze data normalization.

The functions in this module are reusable library utilities only: they do not
load datasets, read calibration files, detect landmarks, or run a preprocessing
pipeline. Callers provide an undistorted image, camera parameters, landmarks or
head pose, and optional gaze target data.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FaceNormalizationResult:
    image: np.ndarray
    head_pose: np.ndarray
    gaze_vector: np.ndarray | None
    warp: np.ndarray
    rotation_matrix: np.ndarray
    face_center: np.ndarray


def draw_gaze(image_in: np.ndarray, pitchyaw: np.ndarray, thickness: int = 2,
              color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Draw a 2D gaze arrow on an image.

    Args:
        image_in: Input image, grayscale or BGR.
        pitchyaw: Gaze angle as ``[pitch, yaw]`` in radians.
        thickness: Arrow thickness.
        color: BGR arrow color.

    Returns:
        Image with the gaze arrow drawn.
    """
    image_out = image_in
    h, w = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))

    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(int)),
        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image_out


def estimateHeadPose(landmarks: np.ndarray, face_model: np.ndarray, camera: np.ndarray,
                     distortion: np.ndarray | None, iterate: bool = True):
    """Estimate head pose from 2D landmarks and a 3D face model.

    Args:
        landmarks: 2D image landmarks with shape ``(N, 1, 2)`` or compatible.
        face_model: Matching 3D landmarks with shape ``(N, 1, 3)`` or compatible.
        camera: Camera intrinsic matrix.
        distortion: Camera distortion coefficients. Use zeros/``None`` for
            already-undistorted points.
        iterate: If True, refine the EPNP estimate with iterative solvePnP.

    Returns:
        ``(rvec, tvec)`` head rotation and translation vectors.
    """
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP
    )
    if not ret:
        raise RuntimeError("cv2.solvePnP failed to estimate an initial head pose")

    if iterate:
        ret, rvec, tvec = cv2.solvePnP(
            face_model, landmarks, camera, distortion, rvec, tvec, True
        )
        if not ret:
            raise RuntimeError("cv2.solvePnP failed during iterative refinement")

    return rvec, tvec


def normalizeData(img: np.ndarray, face: np.ndarray, hr: np.ndarray, ht: np.ndarray,
                  gc: np.ndarray, cam: np.ndarray):
    """Normalize right and left eye crops for appearance-based gaze estimation.

    This is the original eye-level normalization API. The generic ``face`` model
    is expected to contain six landmarks ordered as four eye corners and two
    mouth corners. The returned list contains right eye first, then left eye.

    Returns:
        ``[[eye_image, head_pose, gaze_vector], ...]``.
    """
    focal_norm = 960
    distance_norm = 600
    roiSize = (60, 36)

    img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ht = ht.reshape((3, 1))
    gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face) + ht
    re = 0.5 * (Fc[:, 0] + Fc[:, 1]).reshape((3, 1))
    le = 0.5 * (Fc[:, 2] + Fc[:, 3]).reshape((3, 1))

    data = []
    for et in [re, le]:
        distance = np.linalg.norm(et)
        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
        img_warped = cv2.warpPerspective(img_u, W, roiSize)
        img_warped = cv2.equalizeHist(img_warped)

        hR_norm = np.dot(R, hR)
        hr_norm = cv2.Rodrigues(hR_norm)[0]

        gc_normalized = gc - et
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        data.append([img_warped, hr_norm, gc_normalized])

    return data


def normalizeFaceData(img: np.ndarray, face: np.ndarray, hr: np.ndarray, ht: np.ndarray,
                      gc: np.ndarray, cam: np.ndarray, roiSize: tuple[int, int] = (224, 224),
                      focal_norm: float = 700, distance_norm: float = 600,
                      grayscale: bool = False, equalize: bool = False):
    """Normalize/rectify a full-face image crop.

    This uses the same normalization idea as :func:`normalizeData` but uses a
    single face center instead of separate eye centers.

    Args:
        img: Undistorted input image.
        face: Generic 3D face model with shape ``(3, N)``.
        hr: Head rotation vector from :func:`estimateHeadPose`.
        ht: Head translation vector from :func:`estimateHeadPose`.
        gc: 3D gaze target position in camera coordinates.
        cam: Original camera matrix.
        roiSize: Output face crop size as ``(width, height)``.
        focal_norm: Focal length of the normalized camera. Increase for a
            tighter crop, decrease for a looser crop.
        distance_norm: Normalized distance between face center and camera.
        grayscale: If True, output a grayscale crop. Otherwise keep BGR color.
        equalize: If True and ``grayscale`` is True, apply histogram equalization.

    Returns:
        ``[img_warped, hr_norm, gc_normalized, W]`` where ``W`` is the
        perspective transform used by ``cv2.warpPerspective``.
    """
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else img

    ht = ht.reshape((3, 1))
    gc = gc.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]

    Fc = np.dot(hR, face) + ht
    face_center = np.mean(Fc, axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img_in, W, roiSize)

    if grayscale and equalize:
        img_warped = cv2.equalizeHist(img_warped)

    hR_norm = np.dot(R, hR)
    hr_norm = cv2.Rodrigues(hR_norm)[0]

    gc_normalized = gc - face_center
    gc_normalized = np.dot(R, gc_normalized)
    gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

    return [img_warped, hr_norm, gc_normalized, W]


def normalizeFaceDataWithGeometry(
    img: np.ndarray,
    face: np.ndarray,
    hr: np.ndarray,
    ht: np.ndarray,
    cam: np.ndarray,
    gc: np.ndarray | None = None,
    roiSize: tuple[int, int] = (224, 224),
    focal_norm: float = 700,
    distance_norm: float = 600,
    grayscale: bool = False,
    equalize: bool = False,
) -> FaceNormalizationResult:
    """Normalize a full-face crop and return geometry needed for projection.

    The returned ``rotation_matrix`` maps camera coordinates into the normalized
    face coordinate system. ``face_center`` is the ray origin in camera
    coordinates, both matching the conventions used by ``map_to_display.py``.
    """
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if grayscale else img

    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]

    Fc = np.dot(hR, face) + ht
    face_center = np.mean(Fc, axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    if distance <= 1e-6:
        raise RuntimeError("Face center is too close to the camera origin.")
    z_scale = distance_norm / distance

    cam_norm = np.array([
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down_norm = np.linalg.norm(down)
    if down_norm <= 1e-6:
        raise RuntimeError("Unable to build normalized face coordinate frame.")
    down /= down_norm
    right = np.cross(down, forward)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-6:
        raise RuntimeError("Unable to build normalized face coordinate frame.")
    right /= right_norm
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img_in, W, roiSize)

    if grayscale and equalize:
        img_warped = cv2.equalizeHist(img_warped)

    hR_norm = np.dot(R, hR)
    hr_norm = cv2.Rodrigues(hR_norm)[0]

    gc_normalized = None
    if gc is not None:
        gc = gc.reshape((3, 1))
        gc_normalized = gc - face_center
        gc_normalized = np.dot(R, gc_normalized)
        gc_norm = np.linalg.norm(gc_normalized)
        if gc_norm > 1e-6:
            gc_normalized = gc_normalized / gc_norm

    return FaceNormalizationResult(
        image=img_warped,
        head_pose=hr_norm,
        gaze_vector=gc_normalized,
        warp=W,
        rotation_matrix=R,
        face_center=face_center,
    )


# PEP 8 aliases for new code. The original camelCase names remain exported for
# compatibility with existing code from the paper/repository.
estimate_head_pose = estimateHeadPose
normalize_eye_data = normalizeData
normalize_face_data = normalizeFaceData
normalize_face_data_with_geometry = normalizeFaceDataWithGeometry
