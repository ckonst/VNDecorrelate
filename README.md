# VNDecorrelate
A Velvet-Noise Decorrelator for audio.

Decorrelation refers to the process of transforming an audio source signal into multiple output signals with different waveforms from each other, but with the same sound as the source signal. 

In music production, decorrelation is typically applied to the left and right audio channels, creating the perception of stereo width and space. This, however, may come at the cost of potential coloration or transient smearing artifacts. 

Velvet-Noise Decorrelation (VND) attempts to minimize these artifacts as well as computation cost while reducing the correlation of the outputs as much as possible.

## Velvet Noise

Velvet Noise is simply sparse noise where each value is either -1, 1, or 0:

![Basic Velvet Noise](tests/plots/Basic%20Velvet%20Noise%20Sequence.png)

To reduce transient smearing and frequency coloration you can apply a segmented decay envelope:

![Segmented Decaying Velvet Noise](tests/plots/Segmented%20Decaying%20Velvet%20Noise%20Sequence.png)

As well as logarithmically distributing the impulses towards the start of the sequence:

![Segmented Decaying Log Distributed Velvet Noise](tests/plots/Segmented%20Decaying%20Log%20Distributed%20Velvet%20Noise%20Sequence.png)

## Example Usage

Start by loading an audio file.

```python
import scipy.io.wavfile as wavfile
from vndecorrelate.decorrelation import *

fs, input_signal = wavfile.read("audio/viola.wav")
```

Then you can simply use the `VelvetNoise` class:

```python
velvet_noise = VelvetNoise(
    duration_seconds=0.03,
    num_impulses=30,
)
output_signal = velvet_noise.decorrelate(input_signal)
```
Or:

```python
# manually generate the velvet noise as numpy NDArrays
velvet_noise = generate_velvet_noise(
    duration_seconds=0.03,
    num_impulses=30,
)
# numerically equivalent to VelvetNoise.convolve
output_signal = convolve_velvet_noise(input_signal, velvet_noise)
```

Or you can create a chain of signal processors:

```python
chain = (
    SignalChain(sample_rate_hz=fs)
    .velvet_noise(
        duration_seconds=0.03,
        num_impulses=30,
        log_distribution_strength=1.0,
        seed=1,
    )
    .haas_effect(
        delay_time_seconds=0.02,
        delayed_channel=1, # Right Channel
        mode='LR',
    )
)
# SignalChain is lazy, so instatiation of its signal processors happens here
output_signal = chain(input_signal)
```
To listen back to the processed audio, simply save to a wav file locally.

```python
wavfile.write('audio/viola_dec.wav', fs, output_signal)
```