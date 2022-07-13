# VNDecorrelate
A Velvet-Noise Decorrelator for audio

## Example Usage

```python
from VNDecorrelate.decorrelation import VelvetNoise
import scipy.io.wavfile as wavfile

fs, sig_float32 = wavfile.read("audio/guitar.wav")
vnd = VelvetNoise(fs=fs, density=1000, duration=0.03, width=1.0)

# you can call the decorrelate method or the decorrelator object itself to process the data
# output_sig = vnd.decorrelate(sig_float32)
output_sig = vnd(sig_float32)

wavfile.write('audio/guitar_dec.wav', fs, output_sig)
```

## Further Reading

For a detailed explanation of this project visit: https://ckonst.github.io/Python/VNDecorrelate/vnd.html
