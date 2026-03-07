# Implementation Journal — nn-chain-explorer

## [Phase 0] Project Scaffold — directories, requirements, initial structure
**What was done:**
- Created directories: src/, data/, embeddings/, results/, viz/, notebooks/
- Created src/__init__.py
- Created requirements.txt with pinned minimum versions
- Created journal.md stub

**Decisions made:**
- Decision: Use faiss-cpu (not faiss-gpu)
  - Alternatives considered: faiss-gpu for CUDA acceleration
  - Reason: User is on macOS; faiss-cpu works on all platforms including Apple Silicon
- Decision: Include umap-learn in requirements
  - Reason: Required for notebook Cell 8 UMAP projections

## [Phase 1] embed.py — DINOv2 integration
**What was done:**
- Implemented `_load_dinov2` using `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`
- Used `forward_features(x)['x_norm_clstoken']` for CLS token extraction (shape [batch, 768])
- Applied ImageNet mean/std normalization with resize(256) + center_crop(224)
- L2-normalized all outputs; added norm check (mean/min/max should all be 1.0000)

**Decisions made:**
- Decision: Use resize(256) then center_crop(224) instead of resize(224) directly
  - Reason: Standard ImageNet preprocessing preserves more image context
- Decision: Norm check threshold set to 1e-4 for mean, 1e-3 for min
  - Reason: Float32 arithmetic introduces small rounding; perfect 1.0 is not guaranteed

**Verification:**
- Norm check prints mean=1.0000, min=1.0000, max=1.0000 [OK] after L2 normalization

## [Phase 1] embed.py — CLIP integration
**What was done:**
- Implemented `_load_clip` using `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`
- Used the returned `preprocess` transform directly (no double-normalization)
- Used `model.encode_image(x)` for feature extraction (shape [batch, 512])
- Cast to float32 before L2 normalization (CLIP may return float16 on some devices)

**Decisions made:**
- Decision: Use open_clip's built-in preprocess rather than custom transforms
  - Reason: open_clip's preprocess already includes the correct normalization for CLIP; custom transforms risk double-normalization

## [Phase 1] embed.py — Device selection and OOM handling
**What was done:**
- Device priority: CUDA → MPS → CPU
- OOM retry: halve batch_size up to 3 times, then fall back to CPU with batch_size=8
- On fallback, model is moved to CPU and extraction continues

**Decisions made:**
- Decision: MPS before CPU in device chain
  - Reason: Apple Silicon MPS gives ~3-5x speedup over CPU for ViT inference
- Decision: Up to 3 OOM halvings before CPU fallback
  - Alternatives: immediate CPU fallback, or fixed small batch
  - Reason: 3 halvings covers 32→16→8→4 which handles most GPU memory sizes

## [Phase 2] chain.py — FAISS index choice
**What was done:**
- Used `faiss.IndexFlatIP` (inner product) instead of L2 distance index

**Decisions made:**
- Decision: IndexFlatIP over IndexFlatL2 or approximate indexes
  - Alternatives: IndexFlatL2 (Euclidean), IndexIVFFlat (approximate)
  - Reason: After L2 normalization, inner product == cosine similarity. IndexFlatIP is exact (no approximation error) and fast enough for N=1000. IndexIVFFlat would require training and introduces recall < 1.0, which would corrupt cycle detection.

## [Phase 2] chain.py — Cycle detection algorithm
**What was done:**
- Used visited-set approach: dict mapping node → position in chain
- On revisit, immediately extract transient_length and cycle_length from positions

**Decisions made:**
- Decision: Visited-set (hash map) over Floyd's algorithm
  - Alternatives: Floyd's tortoise-and-hare (O(1) space), Brent's algorithm
  - Reason: For research purposes we need the full chain sequence, cycle_nodes list, and cycle_entry_node — Floyd's only detects the cycle existence, not the path. The visited-set gives all information in one pass. Space is O(chain_length) which is small (max_steps=100).

## [Phase 2] chain.py — Self-exclusion implementation
**What was done:**
- Queried FAISS with k=2 and used `np.where(indices[:,0] == arange, indices[:,1], indices[:,0])`
- Added assertion check after nn_map construction; logged result

**Verification:**
- Self-exclusion check logs "0 self-loops found [OK]" for normalized embeddings

## [Phase 3] analyze.py — Statistics chosen
**What was done:**
- Implemented all 12 statistics from spec (7 per-model + 5 cross-encoder)

**Decisions made:**
- Decision: Save raw `in_degree` list to stats JSON (needed by export for hub gallery ranking)
  - Alternatives: only save histogram
  - Reason: Export and notebook need per-image hub scores; histogram alone is insufficient
- Decision: Identify distinct cycles by frozenset of cycle_nodes (sorted tuple as dict key)
  - Reason: Two chains ending in the same cycle will have the same sorted cycle_nodes set regardless of entry point

## [Phase 4] export.py — Thumbnail encoding
**What was done:**
- Resize to 64×64 with LANCZOS resampling, encode as JPEG at quality=70
- Failed thumbnails replaced with 128-grey placeholder (same base64 encoding)

**Decisions made:**
- Decision: 64×64 at quality=70
  - Alternatives: 32×32 (too small to identify images), 128×128 (would push data.json to ~60MB)
  - Reason: 64×64 at q=70 yields ~15KB per image × 1000 = ~15MB total, within browser comfort zone

## [Phase 5] viz/index.html — D3 force simulation parameters
**What was done:**
- `forceManyBody().strength(-60)`: moderate repulsion to spread nodes without fragmenting
- `forceLink().distance(30).strength(0.3)`: short link distance keeps NN clusters tight
- `forceCollide(r+3)`: prevents node overlap based on hub-scaled radius
- `forceCenter`: keeps graph centered initially

**Decisions made:**
- Decision: strength=-60 for charge (not -200 which is D3 default)
  - Reason: With 1000 nodes and many edges, strong repulsion causes layout to explode; -60 gives a compact but readable graph
- Decision: Node radius ∝ sqrt(hub_score) via d3.scaleSqrt (min 4px, max 18px)
  - Reason: sqrt scale prevents hubs from being excessively large while still showing relative importance

## [Phase 6] notebooks/analysis.ipynb — UMAP parameters
**What was done:**
- `n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42`

**Decisions made:**
- Decision: metric='cosine' (not 'euclidean')
  - Reason: Embeddings are L2-normalized; cosine distance is the natural metric for comparing them (identical to Euclidean distance on unit sphere, but semantically cleaner)
- Decision: n_neighbors=15
  - Alternatives: 5 (very local), 50 (very global)
  - Reason: 15 balances local cluster structure with global topology; standard default for image embeddings at N=1000
- Decision: random_state=42 for reproducibility
  - Reason: UMAP is stochastic; fixing seed ensures consistent figures across runs
