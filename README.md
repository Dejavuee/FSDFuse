# FSDFuse

**FSDFuse: Physics-inspired frequency decoupling and data-adaptive spatial decoupling for SAR–visible image fusion**

[Information Fusion, 139 (2027), 104829](https://doi.org/10.1016/j.inffus.2026.104829).


## Environment

Tested with Python 3.10, PyTorch 2.9.1 and CUDA 12.8.

```bash
git clone https://github.com/Dejavuee/FSDFuse.git
cd FSDFuse
conda create -n fsdfuse python=3.10 -y
conda activate fsdfuse
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU or another CUDA version, use the corresponding [PyTorch installation command](https://pytorch.org/get-started/locally/). A PyTorch scan fallback is included. Optional [VMamba selective-scan extensions](https://github.com/MzeroMiko/VMamba) and Triton accelerate GPU execution.

## Model

```text
fsdfuse/
  __init__.py
  models/
    model.py                  # FSDFuse and reconstruction decoders
    feature_modules.py        # Basic feature extractor and decoders
    frequency_decoupling.py   # ADFD
    spatial_decoupling.py     # DBSD
    fusion_modules.py         # LFSA, HFDA and band fusion
    mamba.py                  # State-space blocks and MSG
    cross_scan.py             # Directional scan and merge
    selective_scan.py         # Selective-scan backends
```

Run from the repository root:

```python
import torch
from fsdfuse import FSDFuse

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FSDFuse().to(device).eval()
visible_y = torch.rand(1, 1, 32, 32, device=device)
sar = torch.rand_like(visible_y)
with torch.no_grad():
    fused_y = model(visible_y, sar)
print(fused_y.shape)
```

Inputs are registered visible luminance and SAR tensors in [0, 1], shaped `B × 1 × H × W`; height and width must be divisible by 16. The output is fused luminance of the same shape. The example uses random initialization; trained weights are not included.

## Citation

If you use FSDFuse in your research, please cite:

```bibtex
@article{YAN2027104829,
  title = {FSDFuse: Physics-inspired frequency decoupling and data-adaptive spatial decoupling for SAR–visible image fusion},
  author = {Yunzhong Yan and Jun Li and Shuowei Liu and La Jiang and Tianpeng Liu and Zhen Liu},
  journal = {Information Fusion},
  volume = {139},
  pages = {104829},
  year = {2027},
  doi = {10.1016/j.inffus.2026.104829},
  url = {https://www.sciencedirect.com/science/article/pii/S1566253526007050}
}
```
