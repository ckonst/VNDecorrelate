import numpy as np

from decorrelator import Filter
from utils import dsp

class HaasDelay(Filter):

    def __init__(self, delay_time, fs=44100, channel=0, mode='LR'):
        self.delay_time = delay_time
        self.fs = fs
        self.channel = channel
        self.mode = mode

    def apply(self, input_sig: np.ndarray) -> np.ndarray:
        return haas_delay(input_sig, self.delay_time, self.fs, self.channel, mode=self.mode)

# TODO: Split into functions, each with a single responsibility.
def haas_delay(input_sig, delay_time, fs, channel, mode='LR'):
    """Return a stereo signal where the specified channel is delayed by delay_time.

    Parameters
    ----------
    input_sig : numpy.ndarray
        The input signal to apply the Haas Effect to.
    delay_time : float
        The time in ms to delay by.
    fs : int
        The sample rate.
    channel : int
        Which channel gets delayed. For stereo audio:
            0 is the left channel, 1 is the right channel.
            OR
            0 is the mid channel, 1 is the side channel.
    mode : string, optional
        Either 'LR' (Left-Right) or 'MS' (Mid-Side).
        If set to MS, then the delay will be applied to the mid or side channel
        depending on the channel argument.
        The default is 'LR'.

    Returns
    -------
    output_sig : numpy.ndarray
        The wet signal with one channel delayed.

    """

    # convert to 32 bit floating point, if it isn't already.
    audio_sig = dsp.to_float32(input_sig)
    # normalize so that the data ranges from -1 to 1 if it doesn't already.
    dsp.peak_normalize(audio_sig)

    delay_len_samples = round(delay_time * fs)
    mono = False

    # if the input was mono, convert it to stereo.
    if input_sig.ndim != 2:
        mono = True
        audio_sig = dsp.mono_to_stereo(audio_sig)

    # if applicable, convert the left-right signal to a mid-side signal.
    if mode == 'MS':
        if mono: # sides will be silent
            mids = dsp.stereo_to_mono(audio_sig)
            sides = mids # this is techincally incorrect, but we fix it later.
            # now stack them and store back into audio_sig
            audio_sig = np.column_stack((mids, sides))
        else:
            audio_sig = dsp.LR_to_MS(audio_sig)

    zero_padding_sig = np.zeros(delay_len_samples)
    wet_channel = np.concatenate((zero_padding_sig, audio_sig[:, channel]))
    dry_channel = np.concatenate((audio_sig[:, 0 if channel else 1], zero_padding_sig))

    # get the location of the wet and dry channels
    # then put them in a tuple so that they can be stacked
    location = (dry_channel, wet_channel) if channel else (wet_channel, dry_channel)
    audio_sig = np.column_stack(location)

    # convert back to left-right, if we delayed the mid and sides.
    if mode == 'MS':
        audio_sig = dsp.MS_to_LR(audio_sig)
        if mono:
            # we duplicated the mono channel, here we compensate for it.
            audio_sig *= 0.5

    return audio_sig
