import ctypes
import platform
import time


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


class GazeDemoUI:
    """Draw a 3x3 grid and highlight the cell containing the predicted gaze."""

    def __init__(
        self,
        screen,
        block_ratio=0.15,
        margin_ratio=0.08,
        gap_ratio=0.08,
        block_num=4,
    ):
        try:
            import pygame
        except ImportError as exc:
            raise SystemExit(
                "pygame is required for the demo UI. Install with `pip install pygame`."
            ) from exc

        self.pygame = pygame
        set_dpi_awareness()
        pygame.init()

        self.screen_cfg = screen
        self.block_ratio = float(block_ratio)
        self.margin_ratio = float(margin_ratio)
        self.gap_ratio = float(gap_ratio)
        self.block_num = int(block_num)

        self.screen = pygame.display.set_mode(
            (self.screen_cfg.width_px, self.screen_cfg.height_px), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Gaze demo grid (press ESC to exit)")
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

        self.rects = self._build_grid()
        self.running = True
        self.last_highlight = None
        self.last_update_ts = time.time()
        self.video_margin_px = 24
        self.video_width_px = max(160, int(round(self.screen_cfg.width_px * 0.18)))
        self.video_height_px = int(round(self.video_width_px * 9 / 16))

    def _build_grid(self):
        w = int(self.screen_cfg.width_px)
        h = int(self.screen_cfg.height_px)
        block_w = int(round(self.block_ratio * w))
        block_h = int(round(self.block_ratio * h))
        margin_x = int(round(self.margin_ratio * w))
        margin_y = int(round(self.margin_ratio * h))
        gap_x = int(round(self.gap_ratio * w))
        gap_y = int(round(self.gap_ratio * h))

        rects = []
        for row in range(self.block_num):
            for col in range(self.block_num):
                x = margin_x + col * (block_w + gap_x)
                y = margin_y + row * (block_h + gap_y)
                rects.append(self.pygame.Rect(x, y, block_w, block_h))
        return rects

    def _draw_camera_frame(self, camera_frame):
        if camera_frame is None:
            return

        pygame = self.pygame
        if camera_frame.ndim != 3 or camera_frame.shape[2] != 3:
            return

        source_h, source_w = camera_frame.shape[:2]
        target_w = self.video_width_px
        target_h = min(
            self.video_height_px,
            max(1, self.screen_cfg.height_px - self.video_margin_px * 2),
        )
        scale = min(target_w / source_w, target_h / source_h)
        preview_w = max(1, int(round(source_w * scale)))
        preview_h = max(1, int(round(source_h * scale)))
        x = self.screen_cfg.width_px - preview_w - self.video_margin_px
        y = self.screen_cfg.height_px - preview_h - self.video_margin_px

        surface = pygame.image.frombuffer(
            camera_frame.tobytes(), (source_w, source_h), "RGB"
        )
        surface = pygame.transform.smoothscale(surface, (preview_w, preview_h))
        self.screen.blit(surface, (x, y))
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(x, y, preview_w, preview_h),
            2,
        )

    def _draw(self, gaze_px, active_idx, camera_frame=None):
        pygame = self.pygame
        self.screen.fill((12, 12, 12))

        base_color = (60, 60, 60)
        active_color = (80, 180, 255)
        border_color = (140, 140, 140)

        for idx, rect in enumerate(self.rects):
            color = active_color if idx == active_idx else base_color
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, border_color, rect, 2)

        if gaze_px is not None:
            pygame.draw.circle(self.screen, (255, 90, 90), gaze_px, 6)

        self._draw_camera_frame(camera_frame)
        pygame.display.flip()

    def update(self, predicted_cm, camera_frame=None):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False

        gaze_px = None
        active_idx = None
        if predicted_cm is not None:
            gaze_px = self.screen_cfg._gaze_to_point(predicted_cm)
            for idx, rect in enumerate(self.rects):
                if rect.collidepoint(gaze_px):
                    active_idx = idx
                    break

        self._draw(gaze_px, active_idx, camera_frame)
        self.clock.tick(60)
        self.last_update_ts = time.time()
        self.last_highlight = active_idx
        return active_idx, self.running

    def capture_frame_rgb(self):
        frame = self.pygame.surfarray.array3d(self.screen)
        return frame.swapaxes(0, 1).copy()

    def close(self):
        self.pygame.quit()
