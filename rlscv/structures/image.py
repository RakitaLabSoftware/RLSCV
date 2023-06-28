from enum import Enum
import math
from typing import Self
import torch
import numpy as np
import torchvision.transforms.v2.functional as T
import cv2
import os


class ImageFormat(Enum):
    CHW = "CHW"
    HWC = "HWC"
    BCHW = "BCHW"


class RLSImage:
    def __init__(
        self, data: torch.Tensor, state: ImageFormat = ImageFormat.CHW
    ) -> None:
        super().__init__()
        self._data = data
        self._state = state

    def clone(self):
        cloned = self._data.clone()
        cur_state = self._state
        return RLSImage(cloned, cur_state)

    def reshape(self, h: int, w: int) -> "RLSImage":
        self._data = T.resize(self._data, [h, w])
        return self

    def rotate(self, angle: float) -> None:
        # FIXME ROTATE FIX
        _, h, w = self._data.shape[:2]
        diagonal: int = int(math.sqrt(h ^ 2 + w ^ 2))
        self._data = T.pad(self._data, [diagonal, diagonal])
        self._data = T.rotate(
            self._data, angle, T.InterpolationMode.BILINEAR, expand=True
        )

    def hflip(self) -> None:
        self._data = T.hflip(self._data)

    def vflip(self) -> None:
        self._data = T.vflip(self._data)

    def __array__(self):
        return self._data.detach().numpy()

    @classmethod
    def load(cls, path: str) -> Self:
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return cls(torch.Tensor(np.array(data) / 255.0).permute(2, 0, 1))  # type: ignore

    @property
    def __call__(self) -> torch.Tensor:
        return self._data

    def to_chw(self) -> "RLSImage":
        """
        Convert to [channels, height, width]
        """
        if self._data is None:
            raise KeyError()
        if self._state == ImageFormat.HWC:
            data = self._data.permute(1, 2, 0).clone()
        elif self._state == ImageFormat.BCHW:
            data = self._data.squeeze(0).clone()
        else:
            data = self._data.clone()
        state = ImageFormat.CHW
        return RLSImage(data, state)

    def to_bchw(self) -> "RLSImage":
        """
        Convert to [batch, channels, height, width]
        """
        if self._data is None:
            raise KeyError()
        if self._state == ImageFormat.CHW:
            data = self._data.unsqueeze(0).clone()
        elif self._state == ImageFormat.HWC:
            data = self._data.permute(2, 0, 1).unsqueeze(0).clone()
        else:
            data = self._data.clone()
        state = ImageFormat.BCHW
        return RLSImage(data, state)

    def to_hwc(self) -> "RLSImage":
        """
        Convert to [height, width, channels]
        """
        if self._data is None:
            raise KeyError()
        if self._state == ImageFormat.BCHW:
            data = self._data.squeeze(0).clone()
        elif self._state == ImageFormat.CHW:
            data = self._data.permute(1, 2, 0).clone()
        else:
            data = self._data.clone()
        state = ImageFormat.HWC
        return RLSImage(data, state)

    def to_numpy(self) -> np.ndarray:
        # TODO add dtype
        if isinstance(self._data, torch.Tensor):
            self._data.detach().cpu().numpy()
        return np.asarray(self._data)

    def save(self, path: str) -> None:
        if not os.path.isdir(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        cv2.imwrite(path, self.to_numpy())
