# VNDecorrelate
[![Version](https://img.shields.io/badge/version-1.0.1-blue?style=for-the-badge)](https://github.com/ckonst/VNDecorrelate/releases)[![PyPI](https://img.shields.io/pypi/v/VNDecorrelate?style=for-the-badge)](https://pypi.org/project/vndecorrelate)[![Tests](https://img.shields.io/github/actions/workflow/status/ckonst/VNDecorrelate/test-vnd.yaml?style=for-the-badge)](https://github.com/ckonst/VNDecorrelate/actions/workflows/test-vnd.yaml)

A Velvet-Noise Decorrelator for audio.

Decorrelation refers to the process of transforming an audio source signal into multiple output signals with different waveforms from each other, but with the same sound as the source signal [[1]](#1).

In music production, decorrelation is typically applied to the left and right audio channels, creating the perception of stereo width and space. This, however, may come at the cost of potential coloration or transient smearing artifacts. 

Velvet-Noise Decorrelation (VND) attempts to minimize these artifacts as well as computation cost while reducing the correlation of the outputs as much as possible [[2]](#2).

## Velvet Noise

Velvet Noise is a sparse noise sequence generated from randomly time-shifted impulses with a random value of either -1 or 1 [[2]](#2):

![Basic Velvet Noise](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Basic%20Velvet%20Noise%20Sequence.png)

To reduce transient smearing and frequency coloration you can apply a segmented decay envelope [[2]](#2):

![Segmented Decaying Velvet Noise](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Segmented%20Decaying%20Velvet%20Noise%20Sequence.png)

As well as logarithmically distributing the impulses towards the start of the sequence [[2]](#2):

![Segmented Decaying Log Distributed Velvet Noise](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Segmented%20Decaying%20Log%20Distributed%20Velvet%20Noise%20Sequence.png)

## Quick Start

First install the package into your environment:
```pip install vndecorrelate```

Then load an audio file.

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
wavfile.write('audio/viola_out.wav', fs, output_signal)
```

## Optimization
`optimization.py` contains functions for optimizing `VelvetNoise` or `HaasEffect` for maximizing stereo seperation while maintaining polar sample symmetry and mono compatiblilty.

`optimize_velvet_noise` optimizes the concentration of impulses towards the start of the filter referred to as `log_distribution_strength`: 
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Kappa.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Kappa.svg">
  <img alt="Kappa" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Kappa.svg">
</picture>

`optimize_haas_delay` optimizes the `delay_time_seconds` parameter:
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Tau.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Tau.svg">
  <img alt="Tau" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Tau.svg">
</picture>


`symmetry_aware_objective` takes the input signal and converts it to polar samples to compute the scalar objective function defined by:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Symmetry%20Aware%20Objective.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Symmetry%20Aware%20Objective.svg">
  <img alt="Symmetry Aware Objective" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Symmetry%20Aware%20Objective.svg">
</picture>

where α is the input scalar to optimize, each 
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Moment.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Moment.svg">
  <img alt="Moment" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Moment.svg">
</picture> is a moment of the polar sample distribution: <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Weighted%20Angular%20Variance.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Weighted%20Angular%20Variance.svg">
  <img alt="Weighted Angular Variance" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Weighted%20Angular%20Variance.svg">
</picture> is the weighted angular variance, <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Centroid.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Centroid.svg">
  <img alt="Centroid" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Centroid.svg">
</picture> is the weighted mean (centroid), and <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Skewness.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Skewness.svg">
  <img alt="Skewness" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Skewness.svg">
</picture> is the skewness. *r* is the correlation between the input left and right channels, φ is the `angle_limit` parameter, and each <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Lambda.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Lambda.svg">
  <img alt="Symmetry Aware Objective" src="https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/img/Lambda.svg">
</picture> is a penalty weight.

Sample runs of `VelvetNoise.decorrelate` with unoptimized and optimized filters can be compared by their polar sample plots generated from `plot_polar_sample`:

![Unoptimized VN Vectorscope](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Unoptimized%20VN%20Vectorscope.png)
![VN Optimized Vectorscope](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/VN%20Optimized%20Vectorscope.png)

## Visualization
To provide further visualization of the effects decorrelation `plot_correlogram` is provided. Short windows of typically ~20ms are taken from two signals to calculate normalized cross-correlation values at various lag distances. `sine_sweep` can be used to generate a test signal that can be compared before and after applying a velvet noise decorrelation.
![Sine Sweep Signal](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Sine%20Sweep%20Signal.png)
We can use the auto correlogram as a baseline:
![Sine Sweep Auto Correlogram](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Sine%20Sweep%20Auto%20Correlogram.png)
Plot the cross correlogram after filtering each channel with velvet noise:
![Velvet Noise Filtered Sine Sweep Cross Correlogram](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/Velvet%20Noise%20Filtered%20Sine%20Sweep%20Cross%20Correlogram.png)
And compare to the behavior of filtering with white noise:
![White Noise Filtered Sine Sweep Cross Correlogram](https://raw.githubusercontent.com/ckonst/VNDecorrelate/master/tests/plots/White%20Noise%20Filtered%20Sine%20Sweep%20Cross%20Correlogram.png)

## References
<a id="1"> </a>
[1] “What is ‘Decorrelation’? | Sweetwater”. <a
    href="https://www.sweetwater.com/insync/decorrelation/">
    https://www.sweetwater.com/insync/decorrelation/</a> (accessed Aug. 10, 2020).

<a id="2"> </a>
    [2] “Velvet-Noise Decorrelator”. <a
        href="http://www.dafx17.eca.ed.ac.uk/papers/DAFx17_paper_96.pdf">
        http://www.dafx17.eca.ed.ac.uk/papers/DAFx17_paper_96.pdf</a> (accessed Aug. 04, 2020).