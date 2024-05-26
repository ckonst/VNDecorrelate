# VNDecorrelate
A Velvet-Noise Decorrelator for audio

## Example Usage

```python
from VNDecorrelate.decorrelation import SignalChain
import scipy.io.wavfile as wavfile
fs, sig_float32 = wavfile.read("audio/guitar.wav")
chain = (
    SignalChain(fs=fs, num_ins=1, num_outs=2)
    .velvet_noise(fs=fs, duration=0.03, num_impulses=30, seed=1, use_log_distribution=True)
    .haas_effect(0.02, fs=fs, channel=1, mode='LR')
)
output_sig = chain.decorrelate(sig_float32)
wavfile.write('audio/guitar_dec.wav', fs, output_sig)
```

## Further Reading

For a detailed explanation of this project visit: https://ckonst.github.io/Python/VNDecorrelate/index.html
