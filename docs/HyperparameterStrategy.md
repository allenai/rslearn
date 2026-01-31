# Hyperparameter Experimentation Strategy

This document outlines a framework for efficient hyperparameter search with rslearn,
focusing on maximizing experimentation velocity while managing preprocessing costs.

## Goals

**Primary Goal:** Maximize experiments per unit wall-clock time.

This translates to maximizing GPU utilization, which only occurs during `fit`. All
preprocessing (prepare, ingest, materialize) is overhead that reduces experimentation
throughput.

## Axioms

### Axiom 1: GPU time is the only time that produces learning

- Prepare, ingest, materialize = CPU/IO = no model improvement
- Fit = GPU = model improvement

**Corollary:** Any preprocessing is "waste" from a learning perspective. Minimize it.

### Axiom 2: Tier 1 parameters have zero marginal preprocessing cost

- Changing learning rate, model architecture, transforms = instant new experiment
- These should be exhausted first before exploring parameters that require preprocessing

**Corollary:** Always run Tier 1 experiments to convergence before paying Tier 2/3 costs.

### Axiom 3: Tier 2/3 exploration is a tax on experimentation velocity

- Each preprocessing-requiring change = hours of CPU/IO before GPU spins up
- Must be justified by expected improvement exceeding time cost

**Corollary:** Batch Tier 2/3 changes together when possible to amortize the cost.

### Axiom 4: Agent-driven experimentation reveals engineering priorities

- If an agent repeatedly wants to test a Tier 2/3 parameter, that's a signal
- Engineering investment should promote high-demand parameters to lower tiers

**Corollary:** Track which parameters block experimentation most frequently.

## Tiered Hyperparameters

### Concept

Hyperparameters are categorized by the preprocessing cost required to change them:

| Tier | Cost | Pipeline Steps Required |
|------|------|------------------------|
| **Tier 1** | Zero | None - change at train time |
| **Tier 2** | Medium | Prepare + Materialize (reuses tile store) |
| **Tier 3** | High | Ingest + Prepare + Materialize (new data download) |

The key question separating Tier 2 from Tier 3:

> "Do the items I need already exist in my local tile store?"

- **Yes** → Tier 2 (re-slice/re-composite existing data)
- **No** → Tier 3 (fetch new data from remote APIs)

### Tier 1: Train-time Only

These parameters can be changed between training runs with zero preprocessing cost.

| Category | Parameters |
|----------|------------|
| **Model** | Architecture, backbone, decoder, pretrained weights |
| **Optimizer** | Learning rate, weight_decay, momentum, optimizer type |
| **Scheduler** | LR schedule, warmup steps, decay rate |
| **Training** | Batch size, epochs, gradient accumulation |
| **Transforms** | Crop size (≤ window), flip, rotate, normalize, color jitter |
| **Task** | Loss function, class weights, task head configuration |
| **Data Loading** | num_workers, prefetch factor, subset of materialized bands |

**Strategy:** Exhaust Tier 1 experiments first. Find the best model/optimizer/transform
configuration before investing in Tier 2/3 exploration.

### Tier 2: Prepare + Materialize

These parameters require re-running `prepare` and `materialize`, but reuse the existing
tile store (no re-download).

| Parameter | Why Tier 2 |
|-----------|-----------|
| `grid_size` | Different window tiling of same geographic area |
| `window_size` | Same as above |
| `query_config.max_matches` | Different number of mosaics per window |
| `query_config.space_mode` | MOSAIC vs CONTAINS vs INTERSECTS matching |
| `query_config.time_mode` | WITHIN vs BEFORE vs AFTER temporal matching |
| `zoom_offset` | Different stored resolution vs window resolution |
| `remap` | Pixel value remapping configuration |
| `resampling_method` | bilinear vs nearest vs cubic |
| `compositing_method` | FIRST_VALID vs MEAN vs MEDIAN |
| `band_sets.format` | GeoTIFF vs PNG output format |
| Shrinking time/space extent | Subset of already-ingested items |

**Note on grid_size:** This can effectively be moved to Tier 1 by materializing large
windows and using `patch_size` in the training config to randomly crop smaller patches:

```yaml
# Materialize once at 512×512
rslearn dataset add_windows --grid_size 512

# Train with different "effective" grid sizes - no re-materialize needed
split_config:
  patch_size: 128  # or 64, or 256
```

### Tier 3: Full Re-ingest

These parameters require downloading new data from remote sources.

| Parameter | Why Tier 3 |
|-----------|-----------|
| Expanding bounding box | New geographic items needed |
| Expanding time range | New temporal items needed |
| New bands not in tile store | Must download additional data |
| Different data source | Completely different items |
| `data_source.init_args` changes | May fetch different items |

**Tier 3 has external dependencies:**
- API rate limits (Copernicus, Planet, etc.)
- Network bandwidth
- Cloud egress costs
- Data availability (some historical data may be offline)

Tier 2 is fully local and controllable. Tier 3 depends on the outside world.

### Fixed Parameters (Not Hyperparameters)

Some parameters are fixed by design choice to reduce experiment complexity:

| Parameter | Fixed Value | Rationale |
|-----------|-------------|-----------|
| `resolution` | 10m | Sentinel-2 native, no resampling artifacts |

### Tier Promotion Through Engineering

With engineering investment, parameters can be promoted to cheaper tiers:

| Parameter | Current Tier | Achievable Tier | Engineering Required |
|-----------|--------------|-----------------|---------------------|
| `grid_size` | 2 | 1 | Use `patch_size` (already available) |
| `max_matches` | 2 | 1 | Over-materialize, subsample at train time |
| `band` subset | 3 | 1 | Ingest all bands, select at train time |
| `resolution` | 2 | 1 | Materialize at finest, downsample in transform |
| Shrink time/space | 3 | 2 | Items exist, just different matching |

**Priority for engineering investment:**

```
Priority = (frequency requested) × (current preprocessing cost) × (feasibility)
```

## Evaluation

### Principle: Canonical Test Pipeline

Training pipelines vary across experiments, but evaluation must be consistent for
apples-to-apples comparison.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING (varies)                           │
│  grid_size, bands, transforms, model architecture, etc.         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (model weights)
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION (fixed)                          │
│  Canonical test set @ 10m, fixed metrics, consistent protocol   │
└─────────────────────────────────────────────────────────────────┘
```

### Canonical Test Dataset

The test dataset is materialized once and never changed:

```bash
# Materialize once at canonical settings
rslearn dataset add_windows \
  --root ./test_canonical \
  --resolution 10 \
  --grid_size 512 \
  --group test \
  ...

rslearn dataset prepare --root ./test_canonical
rslearn dataset ingest --root ./test_canonical
rslearn dataset materialize --root ./test_canonical

# Version control it
git tag test-v1.0
```

This becomes the **immutable reference** for all evaluations.

### Evaluation Contract

All models must produce outputs conforming to a fixed specification:

```python
@dataclass
class EvaluationContract:
    # Spatial (fixed at 10m)
    output_resolution: float = 10.0  # meters/pixel
    
    # For segmentation tasks
    output_classes: list[str]  # e.g., ["background", "solar_farm", ...]
    output_format: str = "class_probabilities"  # (C, H, W) softmax
    
    # For detection tasks
    box_format: str = "xyxy_pixels"
    confidence_threshold: float = 0.5
```

Since resolution is fixed at 10m for both training and evaluation, no cross-resolution
adaptation is needed.

### Metrics Registry

Use a fixed set of metrics computed identically for every evaluation:

```python
CANONICAL_METRICS = {
    "segmentation": [
        ("iou", per_class_iou),
        ("f1", per_class_f1),
        ("accuracy", pixel_accuracy),
    ],
    "detection": [
        ("mAP@50", mean_ap_50),
        ("mAP@50:95", mean_ap_coco),
        ("precision", precision_at_threshold),
        ("recall", recall_at_threshold),
    ],
}
```

### Results Tracking

Each experiment records:

```python
@dataclass
class ExperimentResult:
    # Identity
    experiment_id: str
    timestamp: datetime
    git_commit: str
    
    # Training configuration (varies)
    train_grid_size: int
    model_arch: str
    learning_rate: float
    batch_size: int
    transforms: list[str]
    # ... all Tier 1/2/3 params that were used
    
    # Evaluation configuration (fixed)
    test_dataset_version: str  # "test-v1.0"
    eval_resolution: float = 10.0
    
    # Metrics (comparable across experiments)
    metrics: dict[str, float]  # {"iou": 0.85, "f1": 0.82, ...}
```

### Comparison Table

All experiments with the same `test_dataset_version` are directly comparable:

```
experiment_id | grid | model      | lr     | test_v | iou   | f1
--------------+------+------------+--------+--------+-------+------
exp_001       | 128  | resnet50   | 1e-4   | v1.0   | 0.72  | 0.68
exp_002       | 128  | swin_t     | 1e-4   | v1.0   | 0.78  | 0.74
exp_003       | 256  | swin_t     | 1e-4   | v1.0   | 0.81  | 0.77  ← best
exp_004       | 128  | swin_t     | 3e-4   | v1.0   | 0.76  | 0.73
```

### Test Set Versioning

When the test set must change (new labels, bug fixes, expanded coverage):

```
test-v1.0  (original)
test-v1.1  (fixed mislabeled examples)
test-v2.0  (added new geographic region)
```

**Rule:** Never compare metrics across test versions. When the test set changes,
re-evaluate the top candidate models on the new version to maintain continuity:

```python
def promote_test_version(old_version, new_version, top_k_models):
    """Re-evaluate top models on new test set."""
    for model in top_k_models:
        old_score = evaluate(model, old_version)
        new_score = evaluate(model, new_version)
        log_version_transition(model, old_version, new_version, old_score, new_score)
```

## Strategy Summary

### Phase 1: Exhaust Tier 1

```
Fix: data pipeline (one grid_size, one query_config)
Vary: model, optimizer, scheduler, transforms, loss, batch size

→ Find best Tier 1 configuration with current data
```

### Phase 2: Identify Tier 2/3 Bottlenecks

```
Observe: Which Tier 2/3 params does experimentation suggest varying?
Measure: How much preprocessing time would each change cost?
Decide: Is expected improvement worth the time investment?
```

### Phase 3: Promote High-Value Params

```
For frequently-requested Tier 2/3 params:
  - Engineer lazy/caching solutions
  - Move to Tier 1
  - Unlock new experiment space at zero marginal cost
```

### Decision Framework

When considering a parameter change:

1. **If Tier 1:** Just do it (zero cost)

2. **If Tier 2/3:**
   - Estimate preprocessing time `T`
   - Estimate remaining experiments `N` in this sweep
   - If `T > N × avg_fit_time`: defer or batch with other changes
   - Otherwise: pay the tax, run preprocessing
