import numpy as np
import pytest

from vndecorrelate.utils.dsp import (
    LR_to_MS,
    MS_to_LR,
    NormalizeMode,
    apply_stereo_width,
    cross_correlogram,
    encode_signal_to_side_channel,
    mono_to_stereo,
    peak_normalize,
    rms_normalize,
    stereo_to_mono,
    to_float32,
)


def test__apply_stereo_width():
    x1 = np.random.uniform(size=(100, 2))
    y1 = apply_stereo_width(x1, 1.0)

    assert 0.0 == pytest.approx(np.sum(LR_to_MS(y1)[:, 0]))
    assert 0.0 != pytest.approx(np.sum(LR_to_MS(y1)[:, 1]))

    x2 = np.random.uniform(size=(100, 2))
    y2 = apply_stereo_width(x2, 0.0)

    assert 0.0 == pytest.approx(np.sum(LR_to_MS(y2)[:, 1]))
    assert 0.0 != pytest.approx(np.sum(LR_to_MS(y2)[:, 0]))

    x3 = np.full((100, 2), 100)
    y3 = apply_stereo_width(x3, 0.5)

    assert 100 * 100 == pytest.approx(np.sum(y3))

    x4 = np.full((100, 2), 100)
    y4 = apply_stereo_width(x4, 0.25)

    assert not np.array_equal(x4, y4)


def test__encode_signal_to_side_channel():
    x1 = np.array([])
    x2 = np.array([])

    with pytest.raises(ValueError):
        encode_signal_to_side_channel(x1, x2)

    x1 = np.zeros((100, 2))

    with pytest.raises(ValueError):
        encode_signal_to_side_channel(x1, x2)

    x2 = np.zeros((120, 2))

    with pytest.raises(ValueError):
        encode_signal_to_side_channel(x1, x2)

    x2 = np.column_stack((np.zeros(100), np.full((100,), 100)))
    y1 = encode_signal_to_side_channel(x1, x2)
    y1 = LR_to_MS(y1)

    assert np.array_equal(y1[:, 1], (x2[:, 0] - x2[:, 1]) / 2)

    x2 = np.full((100, 2), 100)
    y1 = encode_signal_to_side_channel(x1, x2)
    y1 = LR_to_MS(y1)

    assert np.array_equal(y1[:, 1], np.zeros((100,)))


def test_to_float32():
    x1 = np.array([1, 2, 3, 4], dtype=np.int32)
    x2 = np.array([1, 2, 3, 4], dtype=np.float32)

    assert to_float32(x1).dtype == np.float32
    assert to_float32(x2).dtype == np.float32
    assert to_float32(x2) is x2


def test_peak_normalize():
    x1 = np.random.normal(loc=0.0, scale=100.0, size=(100, 100))
    x2 = np.zeros((10_000))
    x3 = np.random.uniform(low=2.0, high=10.0, size=(50_000, 2))
    x4 = np.random.uniform(size=(100, 100))

    peak_normalize(x1)
    peak_normalize(x2)
    peak_normalize(x3)
    peak_normalize(x4)

    assert np.max(np.abs(x1)) <= 1.0
    assert np.max(np.abs(x2)) <= 1.0
    assert np.max(np.abs(x3)) <= 1.0
    assert np.max(np.abs(x1)) <= 1.0

    x5 = np.array([0.707, 0.707, 0.707])
    x6 = np.array([0.707, 0.707, 0.707])
    peak_normalize(x5)

    assert not np.allclose(x5, x6)

    x7 = np.array([[0.707, 0.3535], [0.707, 0.3535]])
    x8 = np.array([[0.707, 0.3535], [0.707, 0.3535]])
    peak_normalize(x7)

    assert not np.allclose(x7, x8)


def test_rms_normalize():
    x = np.array([0.707, 0.707, 0.707])
    y = np.array([1.0, 1.0, 1.0])
    rms_normalize(x, y)

    assert np.allclose(x, y)

    x = np.array([[0.707, 0.3535], [0.707, 0.3535]])
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    rms_normalize(x, y, mode=NormalizeMode.DUAL_MONO)

    assert np.allclose(x, y)

    x = np.array([[0.707, 0.3535], [0.707, 0.3535]])
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    rms_normalize(x, y, mode=NormalizeMode.STEREO)

    assert np.allclose(np.array([0.55893258, 0.55893258]), y)

    x = np.array([0.707, 0.707, 0.707])
    y = np.array([[1.0, 1.0], [1.0, 1.0]])
    rms_normalize(x, y, mode=NormalizeMode.STEREO)

    assert x[0] == pytest.approx(y[0, 0])


def test_mono_to_stereo():
    x = np.array([1, 2, 3, 4])
    y = mono_to_stereo(x)
    y_truth = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

    assert np.array_equal(y, y_truth)

    x = np.array([])
    y = mono_to_stereo(x)
    y_truth = np.zeros((0, 2))

    assert np.array_equal(y, y_truth)

    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

    with pytest.raises(ValueError):
        mono_to_stereo(x)


def test_stereo_to_mono():
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = stereo_to_mono(x)
    y_truth = np.array([1, 2, 3, 4])

    assert np.array_equal(y, y_truth)

    x = np.zeros((0, 2))
    y = stereo_to_mono(x)
    y_truth = np.array([])

    assert np.array_equal(y, y_truth)

    x = np.array([])

    with pytest.raises(ValueError):
        stereo_to_mono(x)

    x = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        stereo_to_mono(x)


def test_LR_to_MS():
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = LR_to_MS(x)
    y_truth = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])

    assert np.array_equal(y, y_truth)

    x = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = LR_to_MS(x)
    y_truth = np.array([[1.5, -0.5], [3, -1], [4.5, -1.5], [6, -2]])

    assert np.array_equal(y, y_truth)

    x = np.array([])

    with pytest.raises(ValueError):
        stereo_to_mono(x)

    x = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        stereo_to_mono(x)


def test_MS_to_LR():
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = MS_to_LR(x)
    y_truth = np.array([[2, 0], [4, 0], [6, 0], [8, 0]])

    assert np.array_equal(y, y_truth)

    x = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    y = MS_to_LR(x)
    y_truth = np.array([[3, -1], [6, -2], [9, -3], [12, -4]])

    assert np.array_equal(y, y_truth)

    x = np.array([])

    with pytest.raises(ValueError):
        stereo_to_mono(x)

    x = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        stereo_to_mono(x)


def test_cross_correlogram():
    fs = 16000  # 16 kHz sample rate
    t = np.linspace(0, 2.0, 2 * fs)
    x = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    y = np.sin(2 * np.pi * 440 * t + np.pi / 6) + 0.05 * np.random.normal(
        size=len(t)
    )  # Phase shifted sin wave + small noise
    ccg = cross_correlogram(
        x,
        y,
        sample_rate_hz=fs,
        max_lag_seconds=0.05,
        window_size_seconds=0.02,
        stride_seconds=0.01,
    )

    assert ccg.shape[0] == 199  # 199 windows for a 2s signal at 10ms stride
    assert ccg.shape[1] == 1601  # 1601 lags for +/- 50ms at 16kHz
