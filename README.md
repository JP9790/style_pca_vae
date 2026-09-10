# Style PCA / VAE for ProMP

Code for learning **Probabilistic Movement Primitives (ProMPs)**, extracting **style** with **PCA** or a **β-VAE**, and evaluating personalization on:

1. **Handwriting trajectories** (`ProMP/`) — letters/digits as `(time, x, y)` CSVs  
2. **Driving segments** (`Traffic/`) — NGSIM US-101 / I-80 lane-keeping and lane-change windows  

This repository contains **source code only**. Data, libraries, checkpoints, plots, and ANOVA result tables are gitignored.

## Setup

```bash
git clone https://github.com/JP9790/style_pca_vae.git
cd style_pca_vae
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Trajectory CSVs are expected as `time,x,y` with a header row, under a layout like:

```text
output/
  User_1/
    task_a/
      *.csv
    task_b/
      ...
  User_2/
    ...
```

For digits, scripts often use folders such as `user_1_digit_library_output/` with the same `task_*` convention.

---

## Handwriting pipeline (`ProMP/`)

### 1. Fit ProMPs and run PCA in weight space

```bash
python ProMP/promp_pca.py \
  --output_dir output \
  --n_basis 20 \
  --width 0.05 \
  --out ProMP/promp_pca_results.npz
```

### 2. Partition PCA components into task vs style (ANOVA)

```bash
python ProMP/style_anova.py \
  --pca ProMP/promp_pca_results.npz \
  --alpha 0.05 \
  --out_json ProMP/style_anova_results.json \
  --out_csv ProMP/style_anova_results.csv
```

### 3. Train a ProMP β-VAE (single user)

```bash
python ProMP/promp_vae.py \
  --output_dir output \
  --user_id 1 \
  --n_basis 20 \
  --latent_dim 16 \
  --hidden_size 128 \
  --beta 1.0 \
  --epochs 100 \
  --out_dir ProMP/promp_vae
```

Digit / multi-user variants:

```bash
python ProMP/promp_vae_digit.py
python ProMP/promp_vae_digit_all_users.py
python ProMP/promp_vae_alphabet_all_users.py
```

### 4. VAE-side style ANOVA and personalization

```bash
# Style ANOVA on VAE latents
python ProMP/vae_style_anova.py
python ProMP/vae_style_anova_digit.py

# Build / personalize ProMP libraries with identified style
python ProMP_basic.py --input_dir library_out --output_dir generic_library
python ProMP/personalize_promp_library.py
python ProMP/personalize_promp_library_vae_style.py
python ProMP/personalize_digit_library_and_evaluate.py
python ProMP/personalize_alphabet_library_and_evaluate.py
```

### 5. Evaluation and plots

```bash
python ProMP/evaluate_digit_library_vae_style.py
python ProMP/evaluate_all_users_digit_libraries.py
python ProMP/evaluate_all_users_alphabet_libraries.py
python ProMP/digit_three_way_anova_pca_vae.py
python ProMP/plot_detailed_demo_generic_pca_vae.py
```

Alternative personalization methods (offset scaling, covariance style, etc.) are documented in [`ProMP/README_personalization_methods.md`](ProMP/README_personalization_methods.md).

### Optional: image → stroke trajectories

If you start from handwritten PNGs, thin and extract strokes with `nist_test_adapter/`:

```bash
cd nist_test_adapter
python process_images.py ../Test_write thinned_images
python extract_strokes.py thinned_images ../output
python generate_trajectories.py ../output ../output
```

---

## Traffic / NGSIM pipeline (`Traffic/`)

Place the NGSIM vehicle trajectory CSV at the repo root (or pass `--csv`). Then:

```bash
# 1) Build fixed-length lateral (d, v) segments
python Traffic/prepare_dataset.py \
  --csv "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data.csv" \
  --output Traffic/ngsim_us101_i80_segments.npz

# 2) Train/validation split by driver
python Traffic/train_val_split.py \
  --segments Traffic/ngsim_us101_i80_segments.npz \
  --out Traffic/train_val_split.npz

# 3) Learn style (PCA + β-VAE) and score validation segments
python Traffic/style_learn_evaluate.py \
  --segments Traffic/ngsim_us101_i80_segments.npz \
  --split Traffic/train_val_split.npz \
  --out-dir Traffic/style_eval_results \
  --n-style 5 \
  --device cpu

# 4) Plots and ANOVA on metrics (DTW/RMSE, lane-change, |a_lat|)
python Traffic/plot_style_eval_results.py
python Traffic/anova_dtw_rmse.py
python Traffic/anova_lane_change_metrics.py
python Traffic/anova_lat_acc_magnitude.py
```

`Traffic/config.py` holds sampling rates, segment lengths, and maneuver labels (`LK`, `LCL`, `LCR`).

---

## Project layout

```text
ProMP/                 # Handwriting ProMP + PCA/VAE + personalization + plots
Traffic/               # NGSIM preprocessing, style learning, ANOVA
nist_test_adapter/     # Optional PNG thinning / stroke extraction
ProMP_basic.py         # Fit generic ProMP library from trajectory folders
requirements.txt
```

## Notes

- Default ProMP settings are typically `--n_basis 20` and `--width 0.05`.
- VAE training uses PyTorch; set `--device cuda` when a GPU is available.

