import time


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
        pygame.init()

        self.screen_cfg = screen
        self.block_ratio = float(block_ratio)
        self.margin_ratio = float(margin_ratio)
        self.gap_ratio = float(gap_ratio)
        self.block_num = int(block_num)

        self.screen = pygame.display.set_mode(
            (self.screen_cfg.width_px, self.screen_cfg.height_px)
        )
        pygame.display.set_caption("Gaze demo grid (press ESC to exit)")
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()

        self.rects = self._build_grid()
        self.running = True
        self.last_highlight = None
        self.last_update_ts = time.time()

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

    def _draw(self, gaze_px, active_idx):
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

        pygame.display.flip()

    def update(self, predicted_cm):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

        gaze_px = None
        active_idx = None
        if predicted_cm is not None:
            gaze_px = self.screen_cfg._gaze_to_point(predicted_cm)
            for idx, rect in enumerate(self.rects):
                if rect.collidepoint(gaze_px):
                    active_idx = idx
                    break

        self._draw(gaze_px, active_idx)
        self.clock.tick(60)
        self.last_update_ts = time.time()
        self.last_highlight = active_idx
        return active_idx, self.running

    def close(self):
        self.pygame.quit()
