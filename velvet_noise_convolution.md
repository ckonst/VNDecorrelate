# Velvet-Noise Convolution

The sparse velvet-noise convolution in Eq. (5) of Alary, Politis, and
Välimäki's [*Velvet-Noise Decorrelator*](https://www.dafx17.eca.ed.ac.uk/papers/DAFx17_paper_96.pdf)
is:

$$
(x \ast k)[n]
= \sum_{m=0}^{M_{+}} x\!\left[n - k_{+}[m]\right]
- \sum_{m=0}^{M_{-}} x\!\left[n - k_{-}[m]\right].
$$

## Terms

- $x[n]$ is the input signal at sample index $n$.
- $k$ is the sparse velvet-noise sequence (the convolution filter).
- $x \ast k$ is the output signal after convolution with $k$.
- $n$ is the output-sample index.
- $m$ is an index into a list of nonzero velvet-noise impulses.
- $k_{+}[m]$ is the sample location of the $m$th positive ($+1$) impulse.
- $k_{-}[m]$ is the sample location of the $m$th negative ($-1$) impulse.
- $M_{+}$ and $M_{-}$ are the upper summation limits for the positive- and
  negative-impulse location lists, respectively.

Separating positive and negative impulse locations lets an implementation use
only additions and subtractions for the nonzero filter coefficients; zero
coefficients are skipped.

## Segmented-Decay Convolution (Eq. 10)

With a segmented decay envelope, the paper groups the positive and negative
impulses by segment and scales each segment's partial sum:

$$
(x \ast k)[n]
= \sum_{i=0}^{I} s_i
\left(
    \sum_{m=0}^{M_{i,+}} x\!\left[n - k_{i,+}[m]\right]
    - \sum_{m=0}^{M_{i,-}} x\!\left[n - k_{i,-}[m]\right]
\right).
$$

### Additional Terms

- $i$ is the segment index.
- $I$ is the last segment index.
- $s_i$ is the decay coefficient applied to segment $i$.
- $k_{i,+}[m]$ and $k_{i,-}[m]$ are the locations of the positive and
  negative impulses in segment $i$.
- $M_{i,+}$ and $M_{i,-}$ are the upper summation limits for those two
  segment-specific impulse-location lists.

## Logarithmic Distribution Weights (Eq. 11, Repository Form)

Equation (11) is implemented by
[`generate_log_distribution`](src/vndecorrelate/utils/dsp.py) as a sequence
of exponentially increasing weights, rather than directly as a grid-size
expression:

$$
d_{\kappa}[m]
= \frac{10^{\frac{2\kappa m}{M}}}{1 + 99\kappa},
\qquad m \in \{0, \ldots, M\}.
$$

### Terms

- $d_{\kappa}[m]$ is the unnormalized distribution weight at impulse index
  $m$.
- $\kappa$ is `log_distribution_strength`, in $[0, 1]$.
- $M$ is `size`, which is the number of impulses; the function returns
  $M + 1$ weights.
- At $\kappa = 0$, every weight is $1$. Increasing $\kappa$ increasingly
  favors later weight values before the cumulative positions are normalized.

## Randomized Logarithmic Locations (Eq. 12, Repository Form)

The repository first converts the weights into normalized cumulative sample
locations, then adds the randomized offset:

$$
c_{\kappa}[m]
= \sum_{i=0}^{m} d_{\kappa}[i]
- \mathbb{1}_{\{\kappa = 0\}},
$$

$$
\ell_{\kappa}[m]
= L_s\,\frac{c_{\kappa}[m]}{c_{\kappa}[M]},
$$

$$
k[m]
= \operatorname{round}
\left(
    r_2[m] \, \max\!\left(0, d_{\kappa}[m] T_d - 1\right)
    + \ell_{\kappa}[m]
\right).
$$

### Additional Terms

- $c_{\kappa}[m]$ is the cumulative distribution weight. The indicator
  $\mathbb{1}_{\{\kappa = 0\}}$ reproduces the implementation's uniform-case
  off-by-one adjustment.
- $\ell_{\kappa}[m]$ is the cumulative location scaled to the FIR length.
- $L_s$ is `fir_length_samples`, the total FIR length in samples.
- $T_d$ is the average impulse interval (`sample_rate_hz / density`).
- $r_2[m]$ is a uniform random value in $[0, 1]$.
- $k[m]$ is the rounded sample location of impulse $m$.

`apply_log_distribution` implements the last expression. Although the helper
builds $M + 1$ values for normalization, the filter-generation loop uses the
first $M$ locations, one for each impulse.
