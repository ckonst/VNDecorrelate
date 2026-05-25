import pytest
import scipy.io.wavfile as wavfile


@pytest.fixture(scope='session')
def guitar_signal():
    return wavfile.read('audio/guitar.wav')


@pytest.fixture(scope='session')
def drums_signal():
    return wavfile.read('audio/drums.wav')


@pytest.fixture(scope='session')
def viola_signal():
    return wavfile.read('audio/viola.wav')


@pytest.fixture(scope='session')
def vocal_signal():
    return wavfile.read('audio/vocal.wav')
