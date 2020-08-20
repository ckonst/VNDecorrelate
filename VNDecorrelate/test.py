import random as r
import matplotlib.pyplot as plt
import numpy as np

def vn_generate(p, fs, dur):
    '''
    Generates a velvet noise sequence as a finite impulse response filter
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
    vns: numpy array
        The generated velvet noise sequence.
    
    '''
    # size of the sequence in samples
    Ls = int(np.floor(fs * dur))
    # average spacing between two impulses (grid size)
    Td = fs / p
    # total number of impulses
    M = int(np.floor(Ls / Td))
    # calculate the grid size between each impulse logarithmically
    # use a lambda function to be able to apply this for each impulse index
    Tdl = lambda m : (Ls / 100) * pow(10, (m-2) / float(M))
    # number of segments
    I = 4
    # coefficient values for segmented decay can be set manually
    si = [0.95, 0.5, 0.25, 0.1]
    vns = np.zeros((Ls,), np.float32)
    for m in range(M):
        # first, randomize sign of the impulse
        s = (2 * round(r.random()) - 1)
        #s = 1
        # now randomize the distance between impulses
        #k = round(m * Td + r.random() * (Td - 1))
        #k = int(m * Td)
        # multiply a segmented decay constant based on impulse index
        s *= si[int(m / (M / I)) % 4]
        # logarithmically distribute the impulses
        
        k = round(r.random() * (Tdl(m) - 1)) \
            + int(np.floor(sum((Tdl(i) for i in range(m)))))
            
        if k < len(vns):
            vns[k] = s
    return vns
r.seed(a=0)
vns = [vn_generate(1000, 44100, 0.03), \
    vn_generate(1000, 44100, 0.03)]
plt.figure()
plt.plot(vns[0])
plt.ylim(-1.1, 1.1)
plt.xlabel('Sample index')
plt.title('Figure 3.')
