from enum import Enum
from typing import Literal, Self
import torch
import numpy as np
import torchvision.transforms.functional as T
import cv2


class ImageFormat(Enum):
    CHW = "CHW"
    HWC = "HWC"
    BCHW = "BCHW"


class RLSImage:
    def __init__(
        self, data: torch.Tensor, state: ImageFormat = ImageFormat.CHW
    ) -> None:
        self._data = data
        self._state = state

    def clone(self):
        cloned = self._data.clone()
        cur_state = self._state
        return RLSImage(cloned, cur_state)

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

    @property
    def size(self):
        return self._data.size()

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

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(self._data, *args, **kwargs)
        return RLSImage(result)

    @classmethod
    def load(cls, path: str) -> Self:
        data = cv2.imread(path)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return cls(torch.Tensor(np.array(data) / 255.0).permute(2, 0, 1))  # type: ignore

    @property
    def __call__(self) -> torch.Tensor | np.ndarray:
        return self._data

    def to_chw(self):
        if self._data is None:
            raise KeyError()
        elif self._state == ImageFormat.HWC:
            self._data = self._data.permute(1, 2, 0)
        elif self._state == ImageFormat.BCHW:
            self._data = self._data.squeeze(0)
        self._state = ImageFormat.CHW
        return self

    def to_bchw(self):
        if self._data is None:
            raise KeyError()
        if self._state == ImageFormat.CHW:
            self._data = self._data.unsqueeze(0)
        elif self._state == ImageFormat.HWC:
            self._data = self._data.permute(2, 0, 1).unsqueeze(0)
        self._state = ImageFormat.BCHW
        return self

    def to_hwc(self):
        if self._data is None:
            raise KeyError()
        if self._state == ImageFormat.BCHW:
            self._data = self._data.squeeze(0)
        elif self._state == ImageFormat.CHW:
            self._data = self._data.permute(1, 2, 0)
        self._state = ImageFormat.HWC
        return self

    def to_numpy(self):
        if isinstance(self._data, torch.Tensor):
            self._data.detach().cpu().numpy()
        return np.asarray(self._data)
