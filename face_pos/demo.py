import cv2
import numpy as np
from camera import Camera
from detector import FaceDetector
from estimator import DistanceEstimator

# --- PHYSICAL CALIBRATION (in cm) ---
# Where is the camera lens relative to the exact center of the screen?
# X: + is right, - is left
# Y: + is above screen center, - is below
# Z: + is protruding in front of the screen, - is behind


def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam.")
        return

    h, w, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    writer = cv2.VideoWriter(
        "demo.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    camera = Camera(width=w, height=h, fov_degrees=60.0)
    detector = FaceDetector("models/face_landmarker_v2_with_blendshapes.task")
    estimator = DistanceEstimator(camera)

    print("Pipeline running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_points = detector.extract_key_landmarks(frame)

        if image_points is not None:
            # 1. Get position relative to camera lens
            face_pos_cam = estimator.estimate_3d_position(image_points)

            if face_pos_cam is not None:
                # 3. Visualize results
                text_cam = f"Cam: X:{face_pos_cam[0]:.1f} Y:{face_pos_cam[1]:.1f} Z:{face_pos_cam[2]:.1f}"

                cv2.putText(
                    frame,
                    text_cam,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 150, 0),
                    2,
                )

                # Draw landmarks
                for point in image_points:
                    cv2.circle(
                        frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1
                    )

        writer.write(frame)
        cv2.imshow("Face 3D Position Pipeline", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
