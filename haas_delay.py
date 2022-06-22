import numpy as np

from decorrelator import Filter

# TODO: Finish implementing this class
class HaasDelay(Filter):

    def __init__(self):
        pass

    def apply(self, input_sig: np.ndarray) -> np.ndarray:
        pass

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
    if input_sig.dtype != np.float32:
        audio_sig = input_sig.astype(np.float32)
    else:
        audio_sig = input_sig
    # normalize so that the data ranges from -1 to 1 if it doesn't already.
    if np.max(np.abs(audio_sig)) > 1:
        audio_sig /= np.max(audio_sig)
    delay_len_samples = round(delay_time * fs)
    mono = False
    # if the input was mono, convert it to stereo.
    if input_sig.ndim != 2:
        mono = True
        audio_sig = np.column_stack((audio_sig, audio_sig))

    # if applicable, convert the left-right signal to a mid-side signal.
    # Mid-Side and L-R channel conversions:
    #    L = M + S
    #    R = M − S
    #    S = (L − R) / 2
    #    M = (L + R) / 2
    if mode == 'MS':
        mids = (audio_sig[:, 0] + audio_sig[:, 1]) * 0.5
        if mono: # sides will be silent
            sides = mids # this is techincally incorrect, but we fix it later.
        else:
            sides = (audio_sig[:, 0] - audio_sig[:, 1]) * 0.5
        # now stack them and store back into audio_sig
        audio_sig = np.column_stack((mids, sides))

    zero_padding_sig = np.zeros(delay_len_samples)
    wet_channel = np.concatenate((zero_padding_sig, audio_sig[:, channel]))
    dry_channel = np.concatenate((audio_sig[:, 0 if channel else 1], zero_padding_sig))

    # get the location of the wet and dry channels
    # then put them in a tuple so that they can be stacked
    location = (dry_channel, wet_channel) if channel else (wet_channel, dry_channel)
    audio_sig = np.column_stack(location)

    # convert back to left-right, if we delayed the mid and sides.
    if mode == 'MS':
        L = audio_sig[:, 0] + audio_sig[:, 1]
        R = audio_sig[:, 0] - audio_sig[:, 1]
        if mono:
            # we duplicated the mono channel, here we compensate for it.
            L *= 0.5
            R *= 0.5
        # now stack them and store back into audio_sig
        audio_sig = np.column_stack((L, R))

    return audio_sig
