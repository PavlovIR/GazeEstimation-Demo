import torch
import torch.nn as nn


class ItrackerImageModel(nn.Module):
    """Shared image tower used by the iTracker eyes pathway."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class FaceImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class FaceGridModel(nn.Module):
    def __init__(self, grid_size=25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(grid_size * grid_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ITrackerModel(nn.Module):
    """GazeCapture iTracker model.

    The forward output is ``(x_cm, y_cm)`` in the same camera-centimeter
    coordinate space used by ``labelDotXCam`` and ``labelDotYCam``.
    """

    def __init__(self, grid_size=25):
        super().__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel(grid_size=grid_size)
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, faces, eyes_left, eyes_right, face_grids):
        x_eye_left = self.eyeModel(eyes_left)
        x_eye_right = self.eyeModel(eyes_right)
        x_eyes = torch.cat((x_eye_left, x_eye_right), 1)
        x_eyes = self.eyesFC(x_eyes)

        x_face = self.faceModel(faces)
        x_grid = self.gridModel(face_grids)

        x = torch.cat((x_eyes, x_face, x_grid), 1)
        return self.fc(x)
