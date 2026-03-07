# nn-chain-explorer

A research tool for studying the **topology of learned image embedding spaces** by tracing nearest-neighbor chains. For every image in a dataset, we follow the chain of nearest neighbors until a cycle is detected — then analyze the structure of those cycles across two embedding models: DINOv2 and CLIP.

Key quantities measured:
- **Transient length τ(i):** how many steps before entering a cycle
- **Cycle length ℓ(i):** length of the eventual cycle
- **Hub score h(i):** how many images point to image i as their nearest neighbor
- **Basin of attraction:** how many images eventually flow into a given cycle

---

## Setup

```bash
pip install -r requirements.txt
```

First-time model downloads:
- DINOv2 weights: ~330 MB (downloaded automatically via torch.hub)
- CLIP weights: ~350 MB (downloaded automatically via open_clip)

---

## Quickstart

Run the entire pipeline with one command:

```bash
python main.py run-all
```

This downloads data, extracts embeddings, traces chains, analyzes results, and generates the visualization. Then open the visualization:

```bash
python -m http.server 8080 --directory viz
# Open http://localhost:8080/index.html in your browser
```

---

## Step-by-Step Walkthrough

### 1. Download and sample data

```bash
python main.py setup-data
```

Downloads CIFAR-10 test split and samples 1000 images (100 per class, stratified, seed=42) into `data/{class_name}/`.

Expected output:
```
┌──────────┬───────┬──────────────────────────┐
│ Class    │ Count │ Sample                   │
├──────────┼───────┼──────────────────────────┤
│ airplane │   100 │ data/airplane/0000.jpg   │
│ ...
✓ 1000 images saved to data/
```

### 2. Extract embeddings

```bash
python main.py embed --model all
```

Extracts DINOv2 (768-dim) and CLIP (512-dim) embeddings, L2-normalizes them, and saves:
- `embeddings/dinov2.npy` — shape (1000, 768)
- `embeddings/clip.npy` — shape (1000, 512)
- `embeddings/index.json` — image metadata

### 3. Trace NN chains

```bash
python main.py trace --model all
```

Builds a FAISS index, computes the nearest-neighbor map, and traces chains for every image until a cycle is detected. Saves:
- `results/chains_dinov2.json`
- `results/chains_clip.json`
- `results/nn_map_dinov2.npy`
- `results/nn_map_clip.npy`

### 4. Analyze

```bash
python main.py analyze --model all
```

Computes transient/cycle length distributions, hub scores, basins of attraction, and a cross-encoder comparison. Saves:
- `results/stats_dinov2.json`
- `results/stats_clip.json`
- `results/comparison.json`

### 5. Export for visualization

```bash
python main.py export
```

Generates `viz/data.json` (~15 MB) containing all results plus 64×64 base64 thumbnails for every image.

### 6. Open the visualization

```bash
python -m http.server 8080 --directory viz
```

Open `http://localhost:8080/index.html`. Features:
- Force-directed graph of images connected by NN edges
- Toggle between DINOv2 and CLIP
- Color nodes by class, transient length, or hub score
- Hover nodes for thumbnail preview
- Click nodes for step-by-step chain animation
- Stats panel with histograms and top-10 hub gallery

---

## Checking pipeline status

```bash
python main.py status
```

Shows which output files exist and their sizes.

---

## Using Your Own Images

Place images in `data/` as flat files or in class-named subfolders:

```
data/
├── cats/
│   ├── img001.jpg
│   └── img002.png
└── dogs/
    └── img003.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`. Then run:

```bash
python main.py embed --model all
python main.py trace --model all
python main.py analyze --model all
python main.py export
```

If images are in flat `data/` (no subfolders), class label will be `"unknown"`.

---

## Running the Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb
```

The notebook includes UMAP projections, hub galleries, and cross-encoder comparison plots. All cells have try/except blocks that print friendly messages if results files are missing.

---

## GPU Acceleration

**Apple Silicon (M1/M2/M3):** MPS is used automatically. No changes needed.

**CUDA:** The device detection order is CUDA → MPS → CPU. If you have an NVIDIA GPU, CUDA will be used automatically.

For faster FAISS on CUDA, replace `faiss-cpu` with `faiss-gpu` in `requirements.txt`:
```
faiss-gpu>=1.7.4
```

---

## CLI Reference

```bash
python main.py setup-data                    # download + sample 1000 images
python main.py embed --model dinov2          # extract DINOv2 embeddings
python main.py embed --model clip            # extract CLIP embeddings
python main.py embed --model all             # both
python main.py trace --model dinov2          # trace chains
python main.py trace --model all             # both models
python main.py analyze --model all           # stats + cross-encoder comparison
python main.py analyze --compare             # cross-encoder comparison only
python main.py export                        # generate viz/data.json
python main.py status                        # show what's been computed
python main.py run-all                       # full pipeline

# Flags
--force           re-run even if output exists
--debug           show full Python tracebacks
--batch-size N    override batch size for embed (default: 32)
--max-steps N     override max chain steps for trace (default: 100)
```

---

## Known Limitations

- Designed for ~1000 image scale. For 10K+ images, the D3 force simulation may be slow; consider subsampling for visualization.
- CIFAR-10 only by default; custom datasets work but class labels won't have semantic meaning in the CIFAR-10 color scheme.
- `viz/index.html` requires internet access to load D3 from CDN. For offline use, download D3 and reference it locally.
- UMAP in the notebook can take 1–2 minutes on CPU for 1000 images.