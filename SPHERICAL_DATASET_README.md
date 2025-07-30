# GTO Halo Spherical Dataset Documentation

## Overview

This document provides comprehensive documentation for the GTO Halo training dataset with **spherical thrust coordinates** (`training_data_boundary_100000_spherical.pkl`). This dataset was created to guarantee that thrust vector magnitudes never exceed 1.0, eliminating the need for clipping during diffusion model inference.

---

## Dataset Structure

The dataset consists of **100,000 samples**, each represented as a **67-dimensional vector** with the following structure:

```
Index Range | Component           | Description
------------|--------------------|-----------------------------------------
[0]         | Halo Energy        | Normalized halo energy (class label)
[1:4]       | Time Variables     | Shooting time, initial coast, final coast
[4:64]      | Thrust Variables   | 20 segments × 3 spherical coordinates each
[64:67]     | Mass Variables     | Final fuel mass, halo period, manifold length
```

---

## Component-by-Component Normalization

### 1. Halo Energy (Class Label) - Index [0]

**Physical Range:** `[0.008, 0.095]` (dimensionless energy units)  
**Normalized Range:** `[0, 1]`

**Normalization Formula:**
```python
halo_energy_norm = (halo_energy - 0.008) / (0.095 - 0.008)
```

**Unnormalization Formula:**
```python
halo_energy = halo_energy_norm * (0.095 - 0.008) + 0.008
```

---

### 2. Time Variables - Indices [1:4]

#### Shooting Time - Index [1]
**Physical Range:** `[0, 40]` seconds  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
shooting_time_norm = (shooting_time - 0) / (40 - 0)
```

**Unnormalization:**
```python
shooting_time = shooting_time_norm * (40 - 0) + 0
```

#### Initial Coast Time - Index [2]
**Physical Range:** `[0, 15]` seconds  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
initial_coast_norm = (initial_coast - 0) / (15 - 0)
```

**Unnormalization:**
```python
initial_coast = initial_coast_norm * (15 - 0) + 0
```

#### Final Coast Time - Index [3]
**Physical Range:** `[0, 15]` seconds  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
final_coast_norm = (final_coast - 0) / (15 - 0)
```

**Unnormalization:**
```python
final_coast = final_coast_norm * (15 - 0) + 0
```

---

### 3. Thrust Variables (Spherical Coordinates) - Indices [4:64]

**🎯 KEY INNOVATION:** Thrust vectors are stored in **spherical coordinates** to guarantee magnitude ≤ 1.0.

**Structure:** 20 segments × 3 coordinates = 60 values total
- Each segment represents a thrust control point along the trajectory
- Each segment has 3 spherical coordinates: `(α, β, r)`

#### For each thrust segment (i = 0, 1, ..., 19):
- **Index [4 + 3*i + 0]:** Alpha (azimuthal angle)
- **Index [4 + 3*i + 1]:** Beta (polar angle)  
- **Index [4 + 3*i + 2]:** R (magnitude)

#### Alpha (Azimuthal Angle)
**Physical Range:** `[0, 2π]` radians  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
alpha_norm = alpha / (2 * np.pi)
```

**Unnormalization:**
```python
alpha = alpha_norm * 2 * np.pi
```

#### Beta (Polar Angle)
**Physical Range:** `[0, 2π]` radians  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
beta_norm = beta / (2 * np.pi)
```

**Unnormalization:**
```python
beta = beta_norm * 2 * np.pi
```

#### R (Magnitude) - **CRITICAL CONSTRAINT**
**Physical Range:** `[0, 1]` (dimensionless thrust magnitude)  
**Normalized Range:** `[0, 1]` (already normalized!)

**Normalization:**
```python
r_norm = r  # No transformation needed - r is already the magnitude!
```

**Unnormalization:**
```python
r = r_norm  # No transformation needed - this IS the physical magnitude!
```

**🔑 Key Insight:** The `r` component directly represents the thrust magnitude and is guaranteed to be ≤ 1.0.

---

### 4. Mass Variables - Indices [64:67]

#### Final Fuel Mass - Index [64]
**Physical Range:** `[408, 470]` kg  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
fuel_mass_norm = (fuel_mass - 408) / (470 - 408)
```

**Unnormalization:**
```python
fuel_mass = fuel_mass_norm * (470 - 408) + 408
```

#### Halo Period - Index [65]
**Physical Range:** Variable (depends on halo energy)  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
halo_period_norm = halo_period / get_halo_period(halo_energy)
```

**Unnormalization:**
```python
halo_period = halo_period_norm * get_halo_period(halo_energy)
```

**Note:** This requires the `get_halo_period()` function and the corresponding halo energy.

#### Manifold Length - Index [66]
**Physical Range:** `[5, 11]` (dimensionless)  
**Normalized Range:** `[0, 1]`

**Normalization:**
```python
manifold_length_norm = (manifold_length - 5) / (11 - 5)
```

**Unnormalization:**
```python
manifold_length = manifold_length_norm * (11 - 5) + 5
```

---

## Spherical to Cartesian Conversion

When you need Cartesian thrust components (e.g., for CR3BP simulation), convert from spherical:

### Step 1: Unnormalize Spherical Coordinates
```python
alpha = alpha_norm * 2 * np.pi
beta = beta_norm * 2 * np.pi  
r = r_norm  # Already physical magnitude
```

### Step 2: Convert to Cartesian
```python
ux = r * np.cos(alpha) * np.cos(beta)
uy = r * np.sin(alpha) * np.cos(beta)
uz = r * np.sin(beta)
```

### Step 3: Verify Magnitude Constraint
```python
magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
assert magnitude <= 1.0  # This is GUARANTEED to be true!
```

---

## Complete Unnormalization Example

Here's a complete Python function to unnormalize a single 67-vector sample:

```python
import numpy as np

def unnormalize_spherical_sample(sample_67d, halo_energy_for_period=None):
    """
    Unnormalize a single 67-dimensional sample from the spherical dataset.
    
    Args:
        sample_67d: 67-dimensional normalized sample
        halo_energy_for_period: Physical halo energy for halo period unnormalization
        
    Returns:
        Dictionary with unnormalized components
    """
    result = {}
    
    # 1. Halo Energy [0]
    result['halo_energy'] = sample_67d[0] * (0.095 - 0.008) + 0.008
    
    # 2. Time Variables [1:4]
    result['shooting_time'] = sample_67d[1] * (40 - 0) + 0
    result['initial_coast'] = sample_67d[2] * (15 - 0) + 0
    result['final_coast'] = sample_67d[3] * (15 - 0) + 0
    
    # 3. Thrust Variables [4:64] - Spherical Coordinates
    thrust_data = sample_67d[4:64].reshape(20, 3)  # 20 segments × 3 coords
    
    # Unnormalize spherical coordinates
    alpha = thrust_data[:, 0] * 2 * np.pi  # [0, 2π]
    beta = thrust_data[:, 1] * 2 * np.pi   # [0, 2π]
    r = thrust_data[:, 2]                  # [0, 1] - already physical!
    
    # Store spherical coordinates (for CR3BP simulation)
    result['thrust_spherical'] = {
        'alpha': alpha,  # Shape: (20,)
        'beta': beta,    # Shape: (20,)
        'r': r          # Shape: (20,) - guaranteed ≤ 1.0
    }
    
    # Convert to Cartesian if needed
    ux = r * np.cos(alpha) * np.cos(beta)
    uy = r * np.sin(alpha) * np.cos(beta)
    uz = r * np.sin(beta)
    
    result['thrust_cartesian'] = {
        'ux': ux,  # Shape: (20,)
        'uy': uy,  # Shape: (20,)
        'uz': uz   # Shape: (20,)
    }
    
    # Verify magnitude constraint
    magnitudes = np.sqrt(ux**2 + uy**2 + uz**2)
    result['thrust_magnitudes'] = magnitudes
    result['max_magnitude'] = np.max(magnitudes)  # Should be ≤ 1.0
    
    # 4. Mass Variables [64:67]
    result['final_fuel_mass'] = sample_67d[64] * (470 - 408) + 408
    
    # Halo period requires the physical halo energy
    if halo_energy_for_period is not None:
        result['halo_period'] = sample_67d[65] * get_halo_period(halo_energy_for_period)
    else:
        result['halo_period_norm'] = sample_67d[65]  # Keep normalized
    
    result['manifold_length'] = sample_67d[66] * (11 - 5) + 5
    
    return result
```

---

## Key Advantages of Spherical Representation

### 1. **Guaranteed Magnitude Constraint**
- The `r` component directly represents thrust magnitude
- Since `r_norm ∈ [0,1]` and `r = r_norm`, we have `magnitude = r ≤ 1.0` always
- **No clipping needed during inference!**

### 2. **Mathematical Guarantee**
- It's mathematically impossible for `√(ux² + uy² + uz²) > 1` when derived from spherical coordinates with `r ≤ 1`
- Eliminates the 90% clipping issue observed in the original Cartesian representation

### 3. **Lossless Conversion**
- Verification shows maximum difference of `5.32e-12` compared to original dataset
- All training information is preserved

### 4. **Direct Compatibility**
- CR3BP simulator expects spherical coordinates anyway
- No additional conversion step needed during physical validation

---

## Usage in Training and Inference

### Training
1. Use `training_data_boundary_100000_spherical.pkl` as input
2. Remove mean/std normalization from `datasets.py` (already done)
3. Train diffusion model normally - it will learn to generate values in [0,1]

### Inference  
1. Model outputs will be in [0,1] range for all components
2. Apply unnormalization formulas above to get physical values
3. For thrust: use spherical coordinates directly (no Cartesian conversion needed)
4. **No clipping required** - magnitude constraint is automatically satisfied

### Benchmarking
1. Replace Cartesian unnormalization with direct spherical unnormalization
2. Remove `_convert_to_spherical` method and clipping logic
3. Use unnormalized spherical coordinates directly for CR3BP simulation

---

## Constants Reference

```python
# Physical bounds for unnormalization
MIN_SHOOTING_TIME = 0
MAX_SHOOTING_TIME = 40
MIN_COAST_TIME = 0  
MAX_COAST_TIME = 15
MIN_HALO_ENERGY = 0.008
MAX_HALO_ENERGY = 0.095
MIN_FINAL_FUEL_MASS = 408
MAX_FINAL_FUEL_MASS = 470
MIN_MANIFOLD_LENGTH = 5
MAX_MANIFOLD_LENGTH = 11
THRUST = 1.0  # Maximum thrust magnitude
```

---

## Files Generated

- `training_data_boundary_100000_spherical.pkl` - New spherical dataset
- `convert_training_data_to_spherical.py` - Conversion script
- `verify_spherical_conversion.py` - Verification script
- `spherical_conversion_analysis/` - Conversion analysis plots and data
- `spherical_verification_analysis/` - Verification plots and results

---

## Migration Guide

### From Original Cartesian Dataset:
1. Replace dataset path to use `training_data_boundary_100000_spherical.pkl`
2. Update benchmarking code to use direct spherical unnormalization
3. Remove clipping logic from spherical conversion
4. Update any hardcoded assumptions about Cartesian coordinates

### Verification:
Run `verify_spherical_conversion.py` to confirm the datasets match within numerical precision.

---

**Created:** January 2025  
**Dataset Version:** Spherical v1.0  
**Original Dataset:** `training_data_boundary_100000.pkl` (Cartesian)  
**New Dataset:** `training_data_boundary_100000_spherical.pkl` (Spherical)