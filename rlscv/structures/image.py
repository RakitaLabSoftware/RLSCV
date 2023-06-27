from enum import Enum
from typing import Self
import torch
import numpy as np
import torchvision.transforms.functional as T
import cv2


class ImageFormat(Enum):
    CHW = "CHW"
    HWC = "HWC"
    BCHW = "BCHW"


class RLSImage(torch.Tensor):
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

    def reshape(self, h, w):
        self._data = T.resize(self._data, [h, w])
        return self

    def rotate(self, angle):
        self._data = T.rotate(self._data, angle, expand=True)

    def hflip(self):
        self._data = T.hflip(self._data)

    def vflip(self):
        self._data = T.vflip(self._data)

    def __array__(self):
        return self._data.detach().numpy()

    @classmethod
    def load(cls, path: str) -> Self:
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return cls(torch.Tensor(np.array(data) / 255.0).permute(2, 0, 1))  # type: ignore

    @property
    def __call__(self) -> torch.Tensor | np.ndarray:
        return self._data

    def to_chw(self):
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

    def to_bchw(self):
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

    def to_hwc(self):
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

    def to_numpy(self):
        if isinstance(self._data, torch.Tensor):
            self._data.detach().cpu().numpy()
        return np.asarray(self._data)
