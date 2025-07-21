# VNDecorrelate
A Velvet-Noise Decorrelator for audio

## Example Usage

```python
from VNDecorrelate.decorrelation import SignalChain
import scipy.io.wavfile as wavfile
fs, sig_float32 = wavfile.read("audio/guitar.wav")
    chain = (
        SignalChain(sample_rate_hz=fs)
        .velvet_noise(
            duration=0.03,
            num_impulses=30,
            seed=1,
            use_log_distribution=True,
        )
        .haas_effect(
            delay_time_seconds=0.02,
            delayed_channel=1,
            mode='LR',
        )
    )
output_sig = chain(sig_float32)
wavfile.write('audio/guitar_dec.wav', fs, output_sig)
```

## Further Reading

For a detailed explanation of this project visit: https://ckonst.github.io/Python/VNDecorrelate/vnd.html
