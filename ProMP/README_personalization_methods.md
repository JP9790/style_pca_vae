# Alternative Personalization Methods for ProMP Libraries

This document describes various methods to utilize identified style components for personalizing ProMP libraries, beyond the simple additive style offset approach.

## Current Approach (Baseline)

**Method 1: Simple Style Offset**
- Compute average style coordinates: `z̄^(s) = (1/N) Σ z_i^(s)`
- Reconstruct style offset: `s = V_s * z̄^(s)`
- Personalize: `μ_k^styled = μ_k + s`

**Limitations:**
- Assumes style is constant across all tasks
- Only modifies mean, covariance unchanged
- No control over personalization strength

---

## Alternative Methods

### Method 2: Style Scaling/Modulation

**Concept:** Control the strength of personalization by scaling the style offset.

**Formula:**
```
μ_k^styled = μ_k + α * V_s * z̄^(s)
```

**Parameters:**
- `α = 0.5`: Subtle personalization (half strength)
- `α = 1.0`: Full personalization (baseline)
- `α = 1.5`: Enhanced personalization (150% strength)

**Use Cases:**
- When you want to gradually adapt a library
- When style might be too strong/weak
- For fine-tuning personalization

---

### Method 3: Task-Specific Style Modulation

**Concept:** Allow style to vary per task while maintaining global consistency.

**Formula:**
```
μ_k^styled = μ_k + α * V_s * z̄^(s) + (1-α) * V_s * z_k^(s)
```

Where `z_k^(s)` is the style coordinate computed from task k's weights only.

**Parameters:**
- `α = 0.0`: Pure task-specific style
- `α = 0.5`: Balanced (default)
- `α = 1.0`: Pure global style (baseline)

**Use Cases:**
- When style might vary across tasks
- When some tasks have more style information than others
- For adaptive personalization

---

### Method 4: Style in Covariance

**Concept:** Modify both mean and covariance to reflect style-dependent variability.

**Formula:**
```
μ_k^styled = μ_k + V_s * z̄^(s)
Σ_k^styled = Σ_k + β * V_s * V_s^T
```

**Parameters:**
- `β`: Scale factor for style covariance (e.g., 0.1)

**Use Cases:**
- When style affects not just mean but also variability
- For more expressive personalization
- When you want style-dependent uncertainty

---

### Method 5: Style Interpolation

**Concept:** Blend between generic and fully personalized ProMPs.

**Formula:**
```
μ_k^styled = (1-α) * μ_k^generic + α * (μ_k^generic + V_s * z̄^(s))
```

**Parameters:**
- `α = 0.0`: Pure generic (no personalization)
- `α = 0.5`: Balanced
- `α = 1.0`: Fully personalized

**Use Cases:**
- When you want to preserve some generic characteristics
- For conservative personalization
- When generic library is already good

---

### Method 6: Style-Weighted Combination

**Concept:** Weight personalization based on confidence in style estimation per task.

**Formula:**
```
confidence_k = f(variance of style coordinates for task k)
μ_k^styled = μ_k + confidence_k * V_s * z̄^(s)
```

**Use Cases:**
- When some tasks have more reliable style information
- For adaptive confidence-based personalization
- When style estimation quality varies

---

### Method 7: Style Subspace Projection

**Concept:** Decompose ProMP into task-related and style-related parts, replace only style part.

**Formula:**
```
μ_k = μ_task + μ_style
μ_k^styled = μ_task + V_s * z̄^(s)
```

**Use Cases:**
- When you want to preserve task-specific structure
- For clean separation of task and style
- When task components are well-identified

---

### Method 8: Multi-Component Style

**Concept:** Weight different style components differently to emphasize certain aspects.

**Formula:**
```
μ_k^styled = μ_k + V_s * (W * z̄^(s))
```

Where `W` is a diagonal weight matrix for style components.

**Use Cases:**
- When different style components have different importance
- For fine-grained style control
- When you want to emphasize certain style aspects

---

## Implementation

Run the comparison script to generate personalized libraries using all methods:

```bash
python ProMP/alternative_personalization_methods.py \
    --library User_1_ProMP_library/promp_library.npz \
    --pca ProMP/promp_pca_results.npz \
    --anova ProMP/style_anova_results.json \
    --target_user 1 \
    --trajectories_dir output \
    --out_dir ProMP/personalization_methods_comparison
```

## Choosing a Method

1. **Simple Offset (Method 1)**: Default, good baseline
2. **Scaling (Method 2)**: When you need to control strength
3. **Task-Specific (Method 3)**: When style varies by task
4. **Covariance (Method 4)**: When style affects variability
5. **Interpolation (Method 5)**: When you want conservative personalization
6. **Weighted (Method 6)**: When confidence varies by task
7. **Subspace (Method 7)**: When task/style separation is important
8. **Multi-Component (Method 8)**: When fine-grained control is needed

## Combining Methods

You can also combine methods:
- Use Method 2 (scaling) with Method 3 (task-specific)
- Use Method 4 (covariance) with Method 6 (weighted)
- Experiment with different parameter combinations
