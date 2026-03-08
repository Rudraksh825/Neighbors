# nn-chain-explorer

Studies the **topology of learned image embedding spaces** by tracing nearest-neighbor chains. For every image, we follow the chain of nearest neighbors until a cycle is detected, then analyze the structure across DINOv2 and CLIP.

Key quantities: transient length τ(i), cycle length ℓ(i), hub score h(i), basin of attraction.

---

## Setup

```bash
pip install -r requirements.txt
```

Model weights (~330MB DINOv2, ~350MB CLIP) download automatically on first run.

---

## Quickstart

```bash
# Download 100K images (Tiny ImageNet, no login required)
python main.py setup-data --dataset imagenet

# Extract embeddings, trace chains, analyze, export
python main.py embed --model all
python main.py trace --model all
python main.py analyze --model all
python main.py export

# Open the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

Or run everything in one command (uses CIFAR-10, 1K images):
```bash
python main.py run-all
```

---

## Datasets

**Tiny ImageNet** (default for scale experiments — ungated, no login needed):
```bash
python main.py setup-data --dataset imagenet                        # 100K images, 200 classes
python main.py setup-data --dataset imagenet --n-per-class 50       # 10K images
```

**CIFAR-10** (quick smoke test, 1K images):
```bash
python main.py setup-data                                           # 1K images, 10 classes
```

**Full ImageNet-1k** (1.2M images — gated, requires HF auth):
```bash
huggingface-cli login   # accept terms at huggingface.co/datasets/ILSVRC/imagenet-1k
python main.py setup-data --dataset imagenet --hf-dataset ILSVRC/imagenet-1k
```

**Custom images:** place `.jpg/.png/.webp` files in `data/` (flat or in class-named subfolders), then run `embed` → `trace` → `analyze` → `export`.

---

## CLI Reference

```bash
python main.py setup-data [--dataset cifar10|imagenet] [--n-per-class N] [--hf-dataset HF_ID]
python main.py embed      --model dinov2|clip|all  [--batch-size 32] [--force]
python main.py trace      --model dinov2|clip|all  [--ef 50] [--force]
python main.py analyze    --model dinov2|clip|all  [--compare]
python main.py export     [--thumb-size 64] [--force]
python main.py status
python main.py run-all
```

Global flags: `--debug` (full tracebacks), `--force` (overwrite existing outputs).

---

## Notes

- **NN search:** uses hnswlib HNSW index (scales to 1M+ images). Falls back to brute-force NumPy if hnswlib is unavailable.
- **Large datasets:** embeddings are written via memmap during extraction to avoid RAM spikes (DINOv2 at 100K ≈ 293MB, at 1.2M ≈ 3.7GB).
- **Visualization:** the D3 force graph subsamples to 1000 nodes (top-500 hubs + 500 random) for datasets larger than 1000 images.
- **Apple Silicon:** MPS is used automatically. CUDA is used if available.
