# Installation Guide - Fixing Dependencies

## Common Errors & Solutions

### Error 1: absl-py / TensorFlow Compatibility
**Problem:** Old TensorFlow versions try to install `absl-py==0.8.1` which doesn't support Python 3.10+

**Solution:** Added `absl-py>=1.0.0` and `tensorflow>=2.10.0,<2.16.0` to requirements.txt

### Error 2: numpy/scipy Incompatibility ⚠️
**Problem:** 
```
ValueError: All ufuncs must have type `numpy.ufunc`
```

**Root Cause:** numpy 1.26+ breaks compatibility with scipy < 1.12 and TensorFlow 2.10-2.15

**Solution:** Use compatible versions:
- `numpy>=1.21,<1.24` (compatible with TensorFlow 2.10-2.15)
- `scipy>=1.9.0,<1.12.0` (compatible with numpy 1.23.x)

## What I Changed in requirements.txt

✅ **Fixed versions:**
```
numpy>=1.21,<1.24          # Was: numpy==1.26.4 (too new)
scipy>=1.9.0,<1.12.0       # Added: explicit scipy version
absl-py>=1.0.0             # Added: before tensorflow
tensorflow>=2.10.0,<2.16.0 # Was: tensorflow (no version)
```

## Installation Steps (Python 3.10)

### Step 1: Clean install (recommended)
```bash
# Activate your virtual environment first
conda activate pisa  # or: source venv/bin/activate

# Uninstall problematic packages
pip uninstall -y numpy scipy tensorflow keras

# Install from updated requirements
pip install -r sim_data/requirements.txt
```

### Step 2: Verify installation
```bash
python -c "import numpy; import scipy; import tensorflow as tf; print(f'numpy: {numpy.__version__}'); print(f'scipy: {scipy.__version__}'); print(f'tensorflow: {tf.__version__}')"
```

**Expected output:**
```
numpy: 1.23.x
scipy: 1.11.x  
tensorflow: 2.15.x
```

### Step 3: Install bpy (Blender Python)
```bash
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
```
