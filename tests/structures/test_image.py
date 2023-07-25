import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
from rlsvision import RLSImage
from rlsvision.structures.image import ImageFormat


def test_load():
    # Test loading an image
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    tr = torch.Tensor()
    assert isinstance(img, RLSImage)
    assert img.shape == (3, 224, 224)
    assert img.max() <= 1.0
    assert img.min() >= 0.0


def test_reshape():
    # Test resizing an image
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img = resize(img, [100, 100])
    assert img.shape == (3, 100, 100)


def test_rotate():
    # Test rotating an image
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img.rotate(90)
    # assert img.shape == (3, 224, 224)


def test_hflip():
    # Test horizontal flipping of an image
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img.hflip_()
    # assert np.array_equal(img.to_numpy(), np.fliplr(img.to_numpy()))


def test_vflip():
    # Test vertical flipping of an image
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img.vflip_()
    # assert np.array_equal(img.detach().cpu().numpy(), np.flipud(img.to_numpy()))


def test_to_chw():
    # Test converting an image to CHW format
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img_chw = img.to_chw()
    assert img_chw.shape == (3, 224, 224)
    assert img_chw._state == ImageFormat.CHW


def test_to_bchw():
    # Test converting an image to BCHW format
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img_bchw = img.to_bchw()
    assert img_bchw.shape == (1, 3, 224, 224)
    assert img_bchw._state == ImageFormat.BCHW


def test_to_hwc():
    # Test converting an image to HWC format
    img_path = "tests/test_data/test_image.jpg"
    img = RLSImage.load(img_path)
    img_hwc = img.to_hwc()
    assert img_hwc.shape == (224, 224, 3)
    assert img_hwc._state == ImageFormat.HWC


# def test_save():
#     # Test saving an image
#     img_path = "tests/test_data/test_image.jpg"
#     img = RLSImage.load(img_path)
#     img.save("tests/test_data/test_save.jpg")
#     img_saved = ToTensor()(Image.open("tests/test_data/test_save.jpg"))
#     print(img.shape, img_saved.shape)
#     assert np.array_equal(img_saved.numpy(), img.to_numpy())
