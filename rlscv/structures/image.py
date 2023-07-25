import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Self

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as T
from torch import Tensor
from PIL import Image


class ImageFormat(Enum):
    CHW = "CHW"
    HWC = "HWC"
    BCHW = "BCHW"


class RLSImage(Tensor):
    _state = ImageFormat.CHW

    def rotate(self, angle: float) -> None:
        # FIXME ROTATE FIX
        h, w = self.data.shape[:2]
        diagonal: int = int(math.sqrt(h ^ 2 + w ^ 2))
        self.data = T.pad(self.data, [diagonal, diagonal])
        self.data = T.rotate(self, angle, T.InterpolationMode.BILINEAR, expand=True)

    def hflip_(self) -> None:
        self.data = T.hflip(self.data)

    def vflip_(self) -> None:
        self.data = T.vflip(self.data)

    @classmethod
    def load(cls, path: str) -> Self:
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return cls(torch.Tensor(np.array(data) / 255.0).permute(2, 0, 1))  # type: ignore

    def to_chw(self) -> "RLSImage":
        """
        Convert to [channels, height, width]
        """
        if self.data is None:
            raise KeyError()
        if self._state == ImageFormat.HWC:
            data = self.data.permute(1, 2, 0).clone()
        elif self._state == ImageFormat.BCHW:
            data = self.data.squeeze(0).clone()
        else:
            data = self.data.clone()
        state = ImageFormat.CHW
        image = RLSImage(data)
        image._state = state
        return image

    def to_bchw(self) -> "RLSImage":
        """
        Convert to [batch, channels, height, width]
        """
        if self.data is None:
            raise KeyError()
        if self._state == ImageFormat.CHW:
            data = self.data.unsqueeze(0).clone()
        elif self._state == ImageFormat.HWC:
            data = self.data.permute(2, 0, 1).unsqueeze(0).clone()
        else:
            data = self.data.clone()
        state = ImageFormat.BCHW
        image = RLSImage(data)
        image._state = state
        return image

    def to_hwc(self) -> "RLSImage":
        """
        Convert to [height, width, channels]
        """
        if self.data is None:
            raise KeyError()
        if self._state == ImageFormat.BCHW:
            data = self.data.squeeze(0).clone()
        elif self._state == ImageFormat.CHW:
            data = self.data.permute(1, 2, 0).clone()
        else:
            data = self.data.clone()
        state = ImageFormat.HWC
        image = RLSImage(data)
        image._state = state
        return image

    def to_numpy(self) -> np.ndarray:
        if isinstance(self.data, torch.Tensor):
            self.data.detach().cpu().numpy()
        return np.asarray(self.data)

    def save(self, path: str) -> None:
        if not os.path.isdir(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        img = Image.fromarray(self.to_numpy()[0])
        img = img.convert("RGB")
        img.save(path)
