# SA Testing Status

Current snapshot for the 9 selected benchmarks.

## Summary

- `fir` passed both data-channel and handshake-channel switching tests.
- `matvec` passed the data-channel switching test.
- `matrix` passed the data-channel switching test.
- The remaining benchmarks have not been tested yet.

## Benchmark Status

| Benchmark | Data-Channel Switching | Handshake-Channel Switching | Notes |
| --- | --- | --- | --- |
| `kernel_3mm` | Not tested | Not tested | Pending |
| `gsum` | Not tested | Not tested | Pending |
| `bicg` | Not tested | Not tested | Pending |
| `cnn` | Not tested | Not tested | Failed as SELECT Node switching model is not implemented |
| `matvec` | PASS | Not tested | Data-channel test passed |
| `stencil_2d` | Not tested | Not tested | Pending |
| `matrix` | PASS | Not tested | Data-channel test passed |
| `gcd` | Not tested | Not tested | Pending |
| `fir` | PASS | PASS | Both switching tests passed |

## Totals

- Data-channel switching: `3 / 9` benchmarks passed
- Handshake-channel switching: `1 / 9` benchmarks passed
