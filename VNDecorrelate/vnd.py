import random
import scipy.io.wavfile as wavfile
import numpy as np

def tofloat32(input_sig):
    '''
    Converts an input signal to a 32 bit floating point format

    Parameters
    ----------
    input_sig : n-dimension numpy array
        The input to convert to a 32 bit floating point array.

    Returns
    -------
    input_sig : n-dimension numpy array
        The signal converted to 32 bit floating point array.
    '''
    input_sig = input_sig.astype(np.float32, copy=False)
    input_sig = normalize(input_sig)
    return input_sig

def restore(channel, original):
    '''
    Restores a channel of audio to the original level of the input
    after a convolution.

    Parameters
    ----------
    channel : numpy array
        The channel to restore.
    original : numpy array
        The original input to match the level to.

    Returns
    -------
    numpy array
        the channel normalized to the level of the original.

    '''
    return normalize(channel, min(original), max(original))
    

def normalize(channel, min=-1, max=1):
    '''
    Takes in an input channel and normalizes it to the min and max values.
    
    Parameters
    ----------
    channel : numpy array
        A single channel of the signal to normalize.
    min : int, optional
        The minimum value to set the channel to. The default is -1.
    max : int, optional
        The maximum value to set the channel to. The default is 1.

    Returns
    -------
    channel : numpy array
        The normalized channel.

    '''
    #normalize to -1 to 1 first
    if np.max(np.abs(channel)) != 0:
        channel /= np.max(np.abs(channel))
    if max != 1 and min != -1:
        pos = channel.clip(min=0) * max
        neg = channel.clip(max=0) * -min
        channel = pos + neg
    return channel

def vn_convolve(input_sig, impulse_response):
    '''
    Performs a convolution of the velvet noise sequence with a signal.
    We take advantage of the sparse nature of the sequence
    to do convolution very quickly.
    
    Parameters
    ----------
    input_sig : n-dimension numpy array
        The input signal to convolve with the generated VNS.
    impulse_response : 2d numpy array
        A list containing the velvet noise sequences 
        to convolve with the input signal for each channel.
    
    Returns
    ----------
    output_sig : 2d numpy array
         the output signal in stereo
    '''
    # if the input is mono, then duplicate it to stereo before convolving
    if input_sig.ndim == 1:
        input_sig = np.column_stack((input_sig, input_sig))
    # convert to 32 bit floating point, if it isn't already
    if input_sig.dtype != 'float32':
        input_sig = tofloat32(input_sig)           
    output_sig = np.empty((len(input_sig), 2)) 
    # perform the convolution
    # for each channel
    for i, channel in enumerate(input_sig.T):
        # for further optimization, store the non-zero values
        # of the VNS in two seperate arrays
        k_pos = []
        k_neg = []  
        if i < len(impulse_response):
            for j, v in enumerate(impulse_response[i]):
                if v == 0:
                    continue
                elif v > 0:
                    k_pos.append(i)
                else:
                    k_neg.append(i)
        # for each sample in the input signal
        # add the positive and negative sums
        for n, sample in enumerate(input_sig):
            pos_sum = sum(channel[n - imp] for m, imp in enumerate(k_pos))
            neg_sum = sum(channel[n - imp] for m, imp in enumerate(k_neg))
            output_sig[n][i] = (pos_sum - neg_sum) 
        # restore the original levels of the input signal
        output_sig[:,i] = restore(output_sig[:,i], input_sig[:,i])
    return output_sig

def vn_generate(p, fs, dur):
    '''
    Generates a velvet noise sequence for a finite impulse response
    to convolve with an input signal.
    To avoid audible smearing of transients the following are applied:
        - A segmented decay envelope.
        - Logarithmic impulse distribution.
    Segmented decay is preferred to exponential decay because it uses
    less multiplication, and produces satisfactory results.

    Parameters
    ----------
    p : int
        The density measured in spikes/impulses per second.
    fs : int
        The sample rate.
    dur : float
        The duration of the sequence in seconds.
    
    Returns
    ----------
    impulse_response: numpy array
        The generated velvet noise sequence.
    
    '''
    # size of the sequence in samples
    Ls = int(fs * dur) 
    # average spacing between two impulses (grid size)
    Td = fs / p
    # total number of impulses
    M = Ls / Td
    # calculate the grid size between each impulse logarithmically
    # use a lambda function to be able to apply this for each impulse index
    Tdl = lambda m : (Ls / 100) * pow(10, (m - 2) / M)
    # number of segments
    I = 4
    # coefficient values to for segmented decay can be set manually
    si = [0.95, 0.5, 0.25, 0.1]
    impulse_response = np.zeros((Ls,))
    for i, v in enumerate(impulse_response):
        # first, randomize sign of the impulse
        v = (2 * round(random.random()) - 1) 
        # now randomize the distance between impulses
        index = round(i * Td + random.random() * (Td - 1))
        if index < len(impulse_response):
            # multiply a segmented decay constant based on impulse index
            v *= si[int(index / (Ls / I))]
            # logarithmically distribute the impulses
            index = round(random.random() * (Tdl(i) - 1)) + round(sum(Tdl(j) for j in range(i)))
            impulse_response[index] = v
    return impulse_response

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DENSITY = 1000 # measured in spikes/impulses per second 
    VNS_DURATION = 0.03 # duration of VNS in seconds
    LR_DELAY = 0.02
    sample_rate, sig_float32 = wavfile.read("drums.wav")
    vns = [vn_generate(DENSITY, sample_rate, VNS_DURATION), vn_generate(DENSITY, sample_rate, VNS_DURATION)]
    result = vn_convolve(sig_float32, vns)
    wavfile.write("drums_decorrelated.wav", sample_rate, result)
    
    sr, sf32 = wavfile.read("correct.wav")
    
    plt.figure()
    plt.plot(vns[0])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Velvet Noise Sequence L')
    
    plt.figure()
    plt.plot(vns[1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Velvet Noise Sequence R')
    
    plt.figure()
    plt.plot(sig_float32)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Input')
    
    plt.figure()
    plt.plot(result)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Output')
    
    plt.figure()
    plt.plot(sf32)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Correct Solution')