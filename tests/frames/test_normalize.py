import numpy as np

from fgc_detector.frames.normalize import normalize

CANONICAL = (1920, 1080)


def _image(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_already_canonical_is_returned_unchanged():
    image = _image(1920, 1080)
    result = normalize(image, CANONICAL)
    assert result is image


def test_smaller_16_9_is_upscaled_to_canonical():
    result = normalize(_image(1280, 720), CANONICAL)
    assert result is not None
    assert result.shape == (1080, 1920, 3)


def test_larger_16_9_is_downscaled_to_canonical():
    result = normalize(_image(3840, 2160), CANONICAL)
    assert result is not None
    assert result.shape == (1080, 1920, 3)


def test_wrong_aspect_ratio_is_rejected():
    # 4:3 capture — sampling fixed 16:9 ROIs against it would read misaligned
    # pixels and report confident nonsense, so refuse instead.
    assert normalize(_image(1024, 768), CANONICAL) is None


def test_pillarboxed_ultrawide_is_rejected():
    assert normalize(_image(2560, 1080), CANONICAL) is None


def test_empty_image_is_rejected():
    assert normalize(np.zeros((0, 0, 3), dtype=np.uint8), CANONICAL) is None
