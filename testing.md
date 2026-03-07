# Testing Guide — nn-chain-explorer

Run these tests in order after the pipeline has completed. Each test verifies a specific correctness property.

---

## Test 1: setup-data — 1000 images, 100 per class

**What this step does:** Downloads CIFAR-10 test split and samples 1000 images (100 per class) as JPEGs.

**What this test verifies:** Correct stratified sampling and file organization.

**Command:**
```
python main.py setup-data
```

**Expected output (correct):**
```
┌──────────┬───────┬─────────────────────────────┐
│ Class    │ Count │ Sample                      │
├──────────┼───────┼─────────────────────────────┤
│ airplane │   100 │ data/airplane/0000.jpg      │
│ ...      │   100 │ ...                         │
└──────────┴───────┴─────────────────────────────┘
✓ 1000 images saved to data/
```

**Failure indicators:**
- Count < 100 for any class → sampling logic error
- Download error message → network issue; manually provide images
- `OSError` → disk full or permissions issue

---

## Test 2: embed --model dinov2 — shape (1000, 768), norms = 1.0

**What this step does:** Extracts DINOv2 CLS token embeddings for all images, L2-normalizes, saves to embeddings/dinov2.npy.

**What this test verifies:** Correct shape, L2 normalization, and output file creation.

**Command:**
```
python main.py embed --model dinov2
```

**Expected output (correct):**
```
✓ 1000 embeddings extracted
✓ Output shape: (1000, 768)
✓ All embedding norms = 1.0000
  Embedding norm check: mean=1.0000, min=1.0000, max=1.0000  [OK]
✓ Saved to embeddings/dinov2.npy
```

**Failure indicators:**
- Shape (1000, X) where X ≠ 768 → wrong model or wrong feature extraction
- Norm check `[FAIL]` → L2 normalization skipped or broken
- NaN error → corrupt image passed PIL check

---

## Test 3: embed --model clip — shape (1000, 512), norms = 1.0

**What this step does:** Extracts CLIP ViT-B-32 image embeddings, L2-normalizes, saves to embeddings/clip.npy.

**What this test verifies:** Correct shape and normalization for CLIP.

**Command:**
```
python main.py embed --model clip
```

**Expected output (correct):**
```
✓ 1000 embeddings extracted
✓ Output shape: (1000, 512)
✓ All embedding norms = 1.0000
✓ Saved to embeddings/clip.npy
```

**Failure indicators:**
- Shape dim ≠ 512 → wrong model or wrong encode_image call
- `open_clip` import error → `pip install open_clip_torch`

---

## Test 4: Embedding cache hit — [SKIP] message

**What this step does:** Re-runs embed when output file already exists, without --force.

**What this test verifies:** Caching works; no redundant re-extraction.

**Command:**
```
python main.py embed --model dinov2
```

**Expected output (correct):**
```
[SKIP] embeddings/dinov2.npy already exists (1000 x 768). Use --force to re-extract.
```

**Failure indicators:**
- Re-extraction starts (progress bar appears) → cache check not working
- Error on shape mismatch → index.json and .npy are out of sync (expected if data changed)

---

## Test 5: embed --force — re-extracts even if file exists

**What this step does:** Forces re-extraction ignoring cached .npy file.

**What this test verifies:** --force flag bypasses cache correctly.

**Command:**
```
python main.py embed --model dinov2 --force
```

**Expected output (correct):**
- Progress bar appears and extraction runs (no `[SKIP]` message)
- Summary shows `✓ 1000 embeddings extracted`

**Failure indicators:**
- `[SKIP]` message despite `--force` → flag not being passed through

---

## Test 6: trace --model dinov2 — all chains terminate by cycle

**What this step does:** Builds FAISS index, computes NN map, traces all 1000 chains.

**What this test verifies:** All chains in a finite dataset eventually cycle; 0 terminate by max_steps.

**Command:**
```
python main.py trace --model dinov2
```

**Expected output (correct):**
```
✓ Terminated by max_steps: 0  [OK]
✓ All 1000 chains traced
```

**Failure indicators:**
- `Terminated by max_steps: N > 0` → max_steps too small, or cycle detection bug
- `Self-loop detected` assertion error → nn_map exclusion logic failed

---

## Test 7: trace --model clip — all chains terminate by cycle

**What this step does:** Same as Test 6 but for CLIP embeddings.

**What this test verifies:** CLIP chains also cycle; 0 by max_steps.

**Command:**
```
python main.py trace --model clip
```

**Expected output (correct):**
```
✓ Terminated by max_steps: 0  [OK]
✓ All 1000 chains traced
```

**Failure indicators:** Same as Test 6.

---

## Test 8: Self-exclusion check — 0 self-loops

**What this step does:** After building nn_map with k=2, verifies no image points to itself.

**What this test verifies:** The k=2 self-exclusion logic is working correctly.

**Command:**
```
python main.py trace --model dinov2
```

**Expected output (correct):**
```
✓ Self-exclusion check: 0 self-loops found  [OK]
```

**Failure indicators:**
- `WARNING: N self-loop(s) found` → exclusion logic failed; check indices[:, 0] == arange condition

---

## Test 9: Chain internal consistency

**What this step does:** Checks that `chain[transient_length] == cycle_entry_node` for every chain.

**What this test verifies:** The cycle detection math is correct.

**Command (Python snippet):**
```python
import json
with open('results/chains_dinov2.json') as f:
    data = json.load(f)
chains = data['chains']
errors = []
for i, c in chains.items():
    if c['terminated_by'] == 'cycle':
        tau = c['transient_length']
        entry = c['cycle_entry_node']
        actual = c['chain'][tau]
        if actual != entry:
            errors.append(f"chain {i}: chain[{tau}]={actual} but cycle_entry_node={entry}")
print(f"Errors: {len(errors)}")
if errors:
    for e in errors[:5]:
        print(e)
```

**Expected output (correct):**
```
Errors: 0
```

**Failure indicators:**
- `Errors: N > 0` → trace_chain cycle detection logic has an off-by-one error

---

## Test 10: analyze --model dinov2 — sane statistics

**What this step does:** Computes per-model statistics from chain results.

**What this test verifies:** Fixed point rate is plausible (>50%); max hub < N.

**Command:**
```
python main.py analyze --model dinov2
```

**Expected output (correct):**
- `Fixed point rate (ℓ=1)` > 50%
- `Max hub in-degree` > 0 and < 1000
- `Distinct cycles` > 0 and < 1000
- `results/stats_dinov2.json` created

**Failure indicators:**
- Fixed point rate = 0% → cycle_length values not being computed correctly
- Max hub = 0 → nn_map not loaded correctly

---

## Test 11: analyze --compare — comparison.json written

**What this step does:** Computes cross-encoder comparison statistics (requires both models).

**What this test verifies:** Agreement rate is computed and comparison.json is written.

**Command:**
```
python main.py analyze --compare
```

**Expected output (correct):**
```
NN Agreement Rate: XX.X%
Pearson r (τ_DINOv2 vs τ_CLIP): X.XXXX
✓ Saved to results/comparison.json
```

**Failure indicators:**
- `Missing: results/chains_clip.json` → run trace --model clip first
- Agreement rate = 100% → suspiciously high; likely a bug in comparison logic
- Agreement rate = 0% → also suspicious for real embeddings

---

## Test 12: export — valid viz/data.json with 1000 images

**What this step does:** Assembles all results + thumbnails into viz/data.json.

**What this test verifies:** Output is valid JSON with the correct number of image entries.

**Command:**
```
python main.py export
```

**Then verify:**
```python
import json
with open('viz/data.json') as f:
    d = json.load(f)
print(len(d['images']))          # expect 1000
print(d['metadata']['models'])   # expect ['dinov2', 'clip']
print(d['images'][0].keys())     # expect id, path, class, thumbnail
```

**Expected output (correct):**
```
1000
['dinov2', 'clip']
dict_keys(['id', 'path', 'class', 'thumbnail'])
```

**Failure indicators:**
- JSON parse error → corrupted write (disk full?)
- `len(d['images'])` < 1000 → some images failed thumbnail encode
- `thumbnail` values don't start with `data:image/jpeg;base64,` → encoding bug

---

## Test 13: Visualization loads — graph renders

**What this step does:** Opens viz/index.html in a browser after serving via HTTP.

**What this test verifies:** D3 graph renders with nodes, model toggle works, node click shows chain.

**Command:**
```
python -m http.server 8080 --directory viz
# Then open: http://localhost:8080/index.html
```

**Expected behavior (correct):**
- Force-directed graph appears within a few seconds
- Nodes are colored by class (10 distinct colors)
- Hovering a node shows thumbnail tooltip
- Clicking a node shows chain in bottom inspector bar
- Clicking DINOv2/CLIP buttons switches model data

**Failure indicators:**
- Error overlay: "data.json not found" → file not exported, or opened as file:// instead of http://
- Blank graph with no nodes → JSON parse error in browser console
- "D3 not loaded" → no internet access for CDN

---

## Test 14: Corrupt image handling

**What this step does:** Places a 0-byte .jpg in data/ then runs embed.

**What this test verifies:** Corrupt images are skipped with a warning, not a crash.

**Command:**
```
# Create a corrupt image
python -c "open('data/corrupt_test.jpg', 'w').close()"
python main.py embed --model dinov2 --force
```

**Expected output (correct):**
```
WARNING: Skipping corrupt/unreadable image: data/corrupt_test.jpg
✓ 1000 embeddings extracted   (or 999 if using existing data)
✗ 1 images skipped
```

**Failure indicators:**
- Crash with PIL error or unhandled exception → error handling in _open_image_rgb not working
- No warning printed → corrupt image silently ignored without logging

**Cleanup:**
```
rm data/corrupt_test.jpg
```

---

## Test 15: Notebook execution

**What this step does:** Executes the analysis notebook non-interactively.

**What this test verifies:** All cells run to completion without errors when pipeline results exist.

**Command:**
```
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --output notebooks/analysis_executed.ipynb
```

**Expected output (correct):**
- Exit code 0
- `notebooks/analysis_executed.ipynb` created
- No `[SKIP]` messages for core cells (only if UMAP is missing)

**Failure indicators:**
- Non-zero exit code → a cell raised an unhandled exception
- `CellExecutionError` mentioning a specific cell → check that cell's try/except block
- UMAP section fails → `pip install umap-learn` and retry
