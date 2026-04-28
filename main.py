import cv2
import torch
from face_pos.camera import Camera
from face_pos.detector import FaceDetector
from face_pos.estimator import DistanceEstimator
from gaze_ui.AccuracyUi import GazeAccuracyUI
from gaze_ui.Screen import Screen
from GazeEstimation import Estimator
from IrisDetection import Detector
from l2cs import Pipeline
from ML_Projection.projector import Projector

iris_detector = Detector(
    mediapipe_model_path="assets/mediapipe/face_landmarker.task",
    iris_data_dir="IrisLibs/data",
    device="cpu",
)

estimator = Estimator(
    IrisDetector=iris_detector,
    screen_size=(1920, 1080),
)

point_xy = estimator.predict("input.jpg")
print(point_xy)


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam.")
        return

    gaze_pipeline = Pipeline(
        weights="models/L2CSNet_gaze360.pkl",
        arch="ResNet50",
        device=torch.device("cpu"),
    )

    h, w, _ = frame.shape

    camera = Camera(width=w, height=h, fov_degrees=60.0)
    detector = FaceDetector("models/face_landmarker_v2_with_blendshapes.task")
    estimator = DistanceEstimator(camera)
    screen = Screen("config.toml")
    accuracy_ui = GazeAccuracyUI(screen, points_path="points.csv")
    projector = Projector("ML_Projection/projector_checkpoint.pt")

    print("ML projection pipeline running. Press Q in the accuracy window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_points = detector.extract_key_landmarks(frame)
        if image_points is None:
            _, _, still_running = accuracy_ui.update(None)
            if not still_running:
                break
            continue

        face_pos_cam_mm = estimator.estimate_3d_position(image_points)
        if face_pos_cam_mm is None:
            _, _, still_running = accuracy_ui.update(None)
            if not still_running:
                break
            continue

        results = gaze_pipeline.step(frame)
        if results.pitch.shape[0] == 0 or results.yaw.shape[0] == 0:
            _, _, still_running = accuracy_ui.update(None)
            if not still_running:
                break
            continue

        pitch = float(results.pitch[0])
        yaw = float(results.yaw[0])

        projected_cam_mm = projector.predict(face_pos_cam_mm, pitch, yaw)
        projected_screen_mm = camera_mm_to_screen_mm(projected_cam_mm, screen)

        print(projected_cam_mm)

        _, _, still_running = accuracy_ui.update(projected_screen_mm)
        if not still_running:
            break

    accuracy_ui.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
