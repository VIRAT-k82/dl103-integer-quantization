# Integer-only quantization of ResNet-18 on CIFAR-10 (DL-103)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VIRAT-k82/dl103-integer-quantization/blob/main/quantisation.ipynb)

Small-scale reproduction of **Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018** ([arXiv:1712.05877](https://arxiv.org/abs/1712.05877)).
Course project, Deep Learning 2026, RGIPT.

I rebuilt the core of the paper in PyTorch (affine quantizer, fake quantization with a straight-through estimator, BN folding, activation range tracking, integer-only matmul check) and tested it on ResNet-18 / CIFAR-10 instead of ImageNet.

## Results (single run, seed 0)

FP32 baseline: **91.85 %** (15 epochs).

| Bits | PTQ (%) | QAT (%) |
|---|---|---|
| 8 | 91.91 | 92.04 |
| 6 | 91.75 | 92.17 |
| 4 | 87.69 | 91.21 |
| 3 | 29.49 | 85.94 |
| 2 | 10.00 | 10.00 (collapsed, chance level) |

Size and CPU latency (batch 1, 1 thread, 32x32 input, Colab CPU):

| Model | Size | Latency | Acc. |
|---|---|---|---|
| FP32 | 44.8 MB | 19.34 ms | 91.85 % |
| Fake-quant (simulation, still float) | 44.7 MB | 90.13 ms | 92.04 % |
| Real INT8 (PyTorch fbgemm) | 11.3 MB | 10.85 ms | 91.9 % |

Integer-only check on the final fc layer (256 test images): max |int - fake-quant| = 2.1e-5, and after fixed-point rescaling all outputs match exactly.

Things to keep in mind when reading the numbers:
- QAT got 4 extra epochs that the float model did not, and there is only one seed. The +0.1 to +0.2 point gains at 8 and 6 bits are not evidence that quantization improves accuracy.
- The paper reports 1.4 to 4 point drops at 8 bits on ImageNet. I see no drop on CIFAR-10. I did not investigate why.
- Latency is from PyTorch's int8 kernels on a Colab CPU, not a phone, so it only shows the trend.
- 2-bit training collapsed to chance and I did not find the cause.

Full discussion and comparison with the paper's tables is in `report.pdf`.

## Files

| File | What it is |
|---|---|
| `quantisation.ipynb` | Full notebook (training, PTQ, QAT, integer check, size/latency, ablation, plot), outputs saved |
| `fp32.pt` | Trained FP32 weights (so float training can be skipped) |
| `report/report.pdf`, `report/report.tex` | 4-page IEEE report (LaTeX source needs `report/bits_vs_acc.pdf`) |
| `results/bits_vs_acc.png` | Bit-width ablation plot |
| `requirements.txt` | Python packages |

## Setup and how to run

Open the notebook in Google Colab, set the runtime to **T4 GPU**, and run the cells from top to bottom.

```
pip install torch torchvision matplotlib      # Colab already has these
```

TODO: write the exact versions here (`import torch, torchvision; print(torch.__version__, torchvision.__version__)`).

- **Seed:** `torch.manual_seed(0)`, one run. Data loading uses worker processes and `shuffle=True`, so numbers can differ slightly between runs (PTQ calibration uses random batches; I saw about 0.05 points).
- **Time:** the full notebook takes roughly one hour on a T4 (float training about 15 min, QAT runs about 5-8 min each, plus the data download).
- **Skip float training:** get `fp32.pt` into the Colab session (upload it, or run `!wget -q https://github.com/VIRAT-k82/dl103-integer-quantization/raw/main/fp32.pt`) and load it instead of running the `fit(fp, epochs=15, lr=0.1)` cell:
  ```python
  fp = make_model().to(dev)
  fp.load_state_dict(torch.load('fp32.pt', map_location=dev))
  print(evaluate(fp))   # 0.9185
  ```
- The int8 size/latency cell uses `torch.ao.quantization`, which prints deprecation warnings in recent PyTorch versions. They are harmless for now, but the API is planned for removal.

## What is my own code and what is from libraries

My code: `qparams`, `fake_quant` (STE), `Observer`, `QLayer`, `wrap`, `set_mode`, the PTQ and QAT routines, and the integer-only matmul check.

From libraries: torchvision ResNet-18 and CIFAR-10, `fuse_conv_bn_eval` for BN folding, SGD and OneCycleLR, and PyTorch's `torch.ao.quantization` (fbgemm) **only** for the size and latency benchmark of a real int8 model.

## Differences from the paper

CIFAR-10 / ResNet-18 instead of ImageNet; BN folded once before QAT instead of simulated during training; activation EMA momentum 0.9 and no delayed start of activation quantization; quantization applied to the input of every conv/linear layer (so the identity shortcut and residual add stay float); ReLU instead of ReLU6; QAT only 4 epochs; rescaling shift rounds ties upward. Details are in Section VI of the report.

## References

- B. Jacob et al., CVPR 2018, arXiv:1712.05877 (main reference)
- Y. Bengio, N. Leonard, A. Courville, arXiv:1308.3432 (straight-through estimator)
- K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- A. Krizhevsky, "Learning multiple layers of features from tiny images", 2009
- PyTorch / torchvision documentation (BN fusion, quantization)

I did not use any official or third-party implementation of the paper's method (for example TensorFlow Lite or TF's quantize code) as a base.


