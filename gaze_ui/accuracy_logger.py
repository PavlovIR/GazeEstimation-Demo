import json
import math
import os
import shutil
import time

import cv2


class AccuracyLogger:
    def __init__(self, base_dir, screen_cfg, calibration_path=None):
        self.base_dir = base_dir
        self.screen_cfg = screen_cfg
        self.run_dir = self._make_run_dir(base_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.frames_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.json_path = os.path.join(self.run_dir, "accuracy.json")
        self.records = []
        self.index = 0
        self._copy_calibration(calibration_path)

    def _make_run_dir(self, base_dir):
        os.makedirs(base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, timestamp)
        if not os.path.exists(run_dir):
            return run_dir
        suffix = 1
        while True:
            candidate = f"{run_dir}_{suffix:02d}"
            if not os.path.exists(candidate):
                return candidate
            suffix += 1

    def _clamp_bbox(self, bbox, frame_shape):
        if bbox is None or frame_shape is None:
            return None
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _copy_calibration(self, calibration_path):
        if not calibration_path:
            return
        if not os.path.isabs(calibration_path):
            calibration_path = os.path.join(os.getcwd(), calibration_path)
        if not os.path.exists(calibration_path):
            return
        dest = os.path.join(self.run_dir, os.path.basename(calibration_path))
        try:
            shutil.copy2(calibration_path, dest)
        except OSError:
            pass

    def _flush(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def log(
        self,
        frame,
        face_bbox,
        gaze_cm,
        target_cm,
        target_px,
        error_cm,
        error_deg,
        gaze_vec=None,
    ):
        if frame is None or face_bbox is None or gaze_cm is None:
            return None

        bbox = self._clamp_bbox(face_bbox, frame.shape)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox

        face_crop = frame[y1:y2, x1:x2]
        frame_name = f"frame_{self.index:06d}.jpg"
        frame_path = os.path.join(self.frames_dir, frame_name)

        cv2.imwrite(frame_path, face_crop)

        pred_px = None
        if self.screen_cfg is not None and gaze_cm is not None:
            try:
                pred_px = self.screen_cfg._gaze_to_point(gaze_cm)
            except Exception:
                pred_px = None

        if target_cm is None and target_px is not None and self.screen_cfg is not None:
            try:
                target_cm = self.screen_cfg._compute_gaze_3d(target_px[0], target_px[1])
            except Exception:
                target_cm = None

        derived_error_cm = error_cm
        if derived_error_cm is None and target_cm is not None and gaze_cm is not None:
            try:
                dx = float(gaze_cm[0]) - float(target_cm[0])
                dy = float(gaze_cm[1]) - float(target_cm[1])
                derived_error_cm = math.hypot(dx, dy)
            except Exception:
                derived_error_cm = None

        payload = {
            "frame": os.path.join("frames", frame_name),
            "index": self.index,
            "timestamp": time.time(),
            "error_cm": float(derived_error_cm)
            if derived_error_cm is not None
            else None,
            "error_deg": float(error_deg) if error_deg is not None else None,
            "gaze_vector": [float(v) for v in gaze_vec]
            if gaze_vec is not None
            else None,
            "target_cm": [float(v) for v in target_cm]
            if target_cm is not None
            else None,
            "target_px": [int(v) for v in target_px] if target_px is not None else None,
            "predicted_cm": [float(v) for v in gaze_cm],
            "predicted_px": [int(v) for v in pred_px] if pred_px is not None else None,
            "face_bbox_px": [int(v) for v in bbox],
        }
        self.records.append(payload)

        self.index += 1
        return frame_name

    def close(self):
        self._flush()
