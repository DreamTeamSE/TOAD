# Running the TOAD SAM-3 pipeline on HiPerGator

This has never been run end-to-end anywhere (locally or on HPG) — everything
here is validated by syntax checks and isolated unit tests of the tricky
parts (augmentation alignment, box math), not by an actual training run.
Treat the first submission as a smoke test.

## What's here

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time conda env creation (`seko-sam3`) + SAM-3 package install |
| `prepare_data.slurm` | Regenerates `train.csv`/`train_seko.csv`/`test_seko.csv` with HPG-absolute paths, caches SEKO's predicted masks. CPU only, run once. |
| `train_sam3.slurm` | Fine-tunes SAM-3. GPU job. |
| `eval_sam3.slurm` | Evaluates a checkpoint on the test set, writes `results.json`. GPU job. |

Results loading and visualization live in `src/utils/`:

- `load_results.py` — loads `metrics.json` (training curves) and/or `results.json` (eval table), saves PNG plots, prints a summary including the lowest/highest-DICE test images.
- `visualize_sam3_predictions.py` — runs a checkpoint on sample images and saves `[original | SEKO pred+box | ground truth | SAM-3 prediction]` comparison panels. Can auto-pick the worst/best images from a `results.json` via `--from-results`.

## Before you submit anything

**1. Things only you know — fill in every `REPLACE_WITH_...` placeholder:**
- `--account` / `--qos` in all three `.slurm` files (your PI's HPG allocation group).
- `REPO_ROOT` / `DATA_ROOT` in all three `.slurm` files — where you've put the repo and `seko_storage/` under `/blue/<group>/$USER/...` (`/home` has a small quota, don't put `seko_storage` there).
- `SAM3_SRC_DIR` in `setup_env.sh` — see next point.
- `CKPT_PATH` in `eval_sam3.slurm` (or pass it via `sbatch --export=CKPT_PATH=...`), once you have a trained checkpoint from `train_sam3.slurm`.

**2. Transfer data to HPG.** Nothing here does this for you:
- `seko_storage/` (images, GT masks, `V2_best_model.keras`) — e.g. `rsync -av seko_storage/ hpg:/blue/<group>/$USER/seko_storage/`
- The base SAM-3 checkpoint, `checkpoints/sam3.pt` (~3.4GB) — either transfer it the same way, or leave `--ckpt` pointing at a path that doesn't exist and `train_sam3.py`/`eval_sam3.py` will fall back to downloading from HuggingFace. That requires `huggingface-cli login` (or an `HF_TOKEN` env var) on HPG *and* the compute node having outbound internet access — not guaranteed on every HPC network policy, worth checking before relying on it.
- Your patched SAM-3 source clone — `environment_sam3.yml` notes it "is not available on PyPI" and must be installed from a local clone. This repo doesn't know where that clone lives (your laptop, presumably); transfer it to HPG and point `SAM3_SRC_DIR` at it in `setup_env.sh`.

**3. Run order:**
```
bash setup_env.sh                    # once
sbatch prepare_data.slurm            # once (or whenever source data changes)
sbatch train_sam3.slurm              # produces sam3_toad_best.pt + metrics.json
sbatch --export=CKPT_PATH=<path from previous step> eval_sam3.slurm   # produces results.json
```

**4. `--time` on `train_sam3.slurm` is a guess (24h), not a measurement** — there's no real per-epoch timing to base it on since this has never run. Recommended: first submit with `--epochs 1` and a short `--time` as a smoke test, check the per-epoch wall-clock `train_sam3.py` prints in `logs/toad-train_<jobid>.out`, then size `--time` for the real run from that.

## After a run — pulling results back

```bash
# on your laptop
scp hpg:/blue/<group>/$USER/seko_storage/outputs/run_<jobid>/metrics.json .
scp hpg:/blue/<group>/$USER/seko_storage/outputs/eval_<jobid>/results.json .

conda activate seko   # or any env with matplotlib/pandas — load_results.py doesn't need torch/sam3
python src/utils/load_results.py --metrics metrics.json --results results.json --out-dir plots/
```

`visualize_sam3_predictions.py` needs the actual model (torch + sam3), so run
that one on HPG itself (or another machine with the `seko-sam3` env) rather
than pulling data back first:

```bash
# on HPG, in the seko-sam3 env
python src/utils/visualize_sam3_predictions.py \
    --ckpt /blue/<group>/$USER/seko_storage/outputs/run_<jobid>/sam3_checkpoints/sam3_toad_best.pt \
    --from-results /blue/<group>/$USER/seko_storage/outputs/eval_<jobid>/results.json \
    --n 5 \
    --out-dir viz/
# then scp viz/ back to your laptop to look at the PNGs
```

## Known gaps this doesn't cover

- No multi-GPU / `torch.compile` / mixed-precision tuning — `train_sam3.py` runs batch_size=1 on a single GPU as-is.
- `freeze_backbone()`'s parameter-name match (`"backbone.vision_backbone.trunk"`) is unverified against the real installed `sam3` package — it'll raise loudly if it matches 0 parameters rather than silently training the whole backbone, so you'll know immediately if it's wrong, but I haven't confirmed the string is right.
- Osteophyte is completely out of scope right now (merged into cartilage) — see `train_sam3.py`'s module docstring for why.
