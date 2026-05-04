# VNDecorrelate
A Velvet-Noise Decorrelator for audio.

Decorrelation refers to the process of transforming an audio source signal into multiple output signals with different waveforms from each other, but with the same sound as the source signal. 

In music production, decorrelation is typically applied to the left and right audio channels, creating the perception of stereo width and space. This, however, may come at the cost of potential coloration or transient smearing artifacts. 

Velvet-Noise Decorrelation (VND) attempts to minimize these artifacts as well as computation cost while reducing the correlation of the outputs as much as possible.

## Velvet Noise

Velvet Noise is a sparse noise sequence generated from randomly time-shifted impulses with a random value of either -1 or 1:

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

## Optimization
`optimization.py` contains functions for optimizing `VelvetNoise` or `HaasEffect` for maximizing stereo seperation while maintaining polar sample symmetry and mono compatiblilty.

`optimize_velvet_noise` optimizes the concentration of impulses towards the start of the filter: $\kappa \in [0.0, 1.0]$, referred to as `log_distribution_strength`.

`optimize_haas_delay` optimizes the `delay_time_seconds` parameter: $\tau \in [0.0, max\_delay\_seconds]$

`symmetry_aware_objective` takes the input signal and converts it to polar samples to compute the scalar objective function defined by:

$f(\alpha) = E_w[\theta^2] - \lambda_1(E_w[\theta])^2 - \lambda_2(E_w[\theta^3])^2  - \lambda_3r^2 - \lambda_4(max|{\theta}| - \phi)^2$

where $\alpha$ is the input scalar to optimize, each $E_w[\theta^n]$ is a moment of the polar sample distribution: $E_w[\theta^2]$ is the weighted angular variance, $(E_w[\theta])^2$ is the weighted mean (centroid), and $(E_w[\theta^3])^2$ is the skewness. $r$ is the correlation between the input left and right channels, $\phi$ is the `angle_limit` parameter, and each $\lambda_n$ is a penalty weight.

Sample runs of `VelvetNoise.decorrelate` with unoptimized and optimized filters can be compared by their polar sample plots generated from `plot_polar_sample`:

![Unoptimized VN Vectorscope](tests/plots/Unoptimized%20VN%20Vectorscope.png)
![VN Optimized Vectorscope](tests/plots/VN%20Optimized%20Vectorscope.png)

## Visualization
To provide further visualization of the effects decorrelation `plot_correlogram` is provided. Short windows of typically ~20ms are taken from two signals to calculate normalized cross-correlation values at various lag distances. `sine_sweep` can be used to generate a test signal that can be compared before and after applying a velvet noise decorrelation.
![Sine Sweep Signal](tests/plots/Sine%20Sweep%20Signal.png)
We can use the auto correlogram as a baseline:
![Sine Sweep Auto Correlogram](tests/plots/Sine%20Sweep%20Auto%20Correlogram.png)
Plot the cross correlogram after filtering each channel with velvet noise:
![Velvet Noise Filtered Sine Sweep Cross Correlogram](tests/plots/Velvet%20Noise%20Filtered%20Sine%20Sweep%20Cross%20Correlogram.png)
And compare to the behavior of filtering with white noise:
![White Noise Filtered Sine Sweep Cross Correlogram](tests/plots/White%20Noise%20Filtered%20Sine%20Sweep%20Cross%20Correlogram.png)
