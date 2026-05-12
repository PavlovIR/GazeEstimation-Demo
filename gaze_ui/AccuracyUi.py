import csv
import ctypes
import platform
import random
import time

import numpy as np


def set_dpi_awareness():
    if platform.system() == "Windows":
        # Windows 8.1 and 10+ (Per-Monitor DPI awareness)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            # Windows Vista, 7, and 8
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass  # Fallback for older systems


class GazeAccuracyUI:
    """Show a target dot and compute accuracy against predicted gaze."""

    def __init__(
        self, screen, hold_seconds=2, margin_px=80, points_path=None, cicle_num=5
    ):
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required for --pygame mode. Install with `pip install pygame`."
            ) from exc

        self.pygame = pygame
        self.screen_cfg = screen
        set_dpi_awareness()
        pygame.init()

        self.hold_seconds = hold_seconds
        self.margin_px = margin_px
        max_margin_x = max(0, (self.screen_cfg.width_px - 10) // 2)
        max_margin_y = max(0, (self.screen_cfg.height_px - 10) // 2)
        self.margin_px = min(self.margin_px, max_margin_x, max_margin_y)
        self.points = []
        self.point_index = 0
        self.completed_cycles = 0
        self.total_cycles = max(1, int(cicle_num))

        if points_path:
            with open(points_path, mode="r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                self.points = [
                    self.relative_to_px((row["x"], row["y"]))
                    for row in reader
                    if row.get("x") and row.get("y")
                ]

        self.screen = pygame.display.set_mode(
            (self.screen_cfg.width_px, self.screen_cfg.height_px), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Gaze target (press Q to exit)")
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

        self.target_px = None
        self.target_mm = None
        self.errors_mm = []
        self.last_target_ts = 0.0
        self.running = True
        self._new_target()

    def relative_to_px(self, point):
        return round(float(point[0]) * self.screen_cfg.width_px), round(
            float(point[1]) * self.screen_cfg.height_px
        )

    def _new_target(self, point=None):
        if point is not None:
            x, y = point
        elif self.points:
            x, y = self.points[self.point_index]
            self.point_index += 1
            if self.point_index >= len(self.points):
                self.point_index = 0
                self.completed_cycles += 1
        else:
            x = random.randint(
                self.margin_px, self.screen_cfg.width_px - self.margin_px
            )
            y = random.randint(
                self.margin_px, self.screen_cfg.height_px - self.margin_px
            )
        self.target_px = (x, y)
        self.target_mm = self.screen_cfg._px_to_mm(x, y)
        self.last_target_ts = time.time()

    def _draw(self, pred_px, error_mm, avg_error, status=None):
        pygame = self.pygame
        self.screen.fill((10, 10, 10))
        pygame.draw.circle(self.screen, (80, 180, 255), self.target_px, 20)

        lines = [
            f"Target (px): {self.target_px[0]}, {self.target_px[1]}",
            f"Target (cm from cam): {self.target_mm[0]:.2f}, {self.target_mm[1]:.2f}",
        ]

        if pred_px is not None:
            pygame.draw.circle(self.screen, (255, 90, 90), pred_px, 15)
            pygame.draw.line(self.screen, (180, 180, 180), self.target_px, pred_px, 1)
            lines.append(f"Pred (px): {pred_px[0]}, {pred_px[1]}")
        elif status:
            lines.append(status)

        if error_mm is not None:
            lines.append(f"Last error: {error_mm:.2f} mm")
        if avg_error is not None:
            lines.append(
                f"Mean error: {avg_error:.2f} mm over {len(self.errors_mm)} samples"
            )

        y = 10
        for line in lines:
            surf = self.font.render(line, True, (240, 240, 240))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 4

        pygame.display.flip()

    def update(self, predicted_mm):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.running = False

        pred_px = None
        error_mm = None
        avg_error = None
        if predicted_mm is not None:
            pred_px = self.screen_cfg._mm_to_px(predicted_mm)
            error_mm = float(
                np.linalg.norm(np.array(predicted_mm) - np.array(self.target_mm))
            )
            self.errors_mm.append(error_mm)
            avg_error = float(np.mean(self.errors_mm))

        now = time.time()
        if now - self.last_target_ts > self.hold_seconds:
            if self.points and self.completed_cycles >= self.total_cycles:
                self.running = False
            elif self.running:
                self._new_target()

        self._draw(pred_px, error_mm, avg_error)
        self.clock.tick(60)
        return error_mm, avg_error, self.running

    def update_px(self, predicted_px, status=None):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.running = False

        pred_px = None
        error_mm = None
        avg_error = None
        if predicted_px is not None:
            x_px, y_px = float(predicted_px[0]), float(predicted_px[1])
            pred_px = (
                int(np.clip(round(x_px), 0, self.screen_cfg.width_px - 1)),
                int(np.clip(round(y_px), 0, self.screen_cfg.height_px - 1)),
            )
            predicted_mm = self.screen_cfg._px_to_mm(*pred_px)
            error_mm = float(
                np.linalg.norm(np.array(predicted_mm) - np.array(self.target_mm))
            )
            self.errors_mm.append(error_mm)
            avg_error = float(np.mean(self.errors_mm))

        now = time.time()
        if now - self.last_target_ts > self.hold_seconds:
            if self.points and self.completed_cycles >= self.total_cycles:
                self.running = False
            elif self.running:
                self._new_target()

        self._draw(pred_px, error_mm, avg_error, status=status)
        self.clock.tick(60)
        return error_mm, avg_error, self.running

    def close(self):
        self.pygame.quit()
