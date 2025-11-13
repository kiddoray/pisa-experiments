#!/bin/bash
# Installation script for pisa-experiments
# Fixes numpy/scipy/tensorflow compatibility issues

set -e

echo "========================================"
echo "PISA Experiments - Installation Script"
echo "========================================"
echo ""

# Check if in virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "WARNING: No virtual environment detected!"
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $PYTHON_VERSION"

# Uninstall problematic packages
echo ""
echo "Step 1: Removing potentially incompatible packages..."
pip uninstall -y numpy scipy tensorflow keras 2>/dev/null || true

# Install requirements
echo ""
echo "Step 2: Installing dependencies from requirements.txt..."
pip install -r sim_data/requirements.txt

# Verify installation
echo ""
echo "Step 3: Verifying installation..."
python -c "
import numpy
import scipy
import tensorflow as tf

print('✅ numpy:', numpy.__version__)
print('✅ scipy:', scipy.__version__)
print('✅ tensorflow:', tf.__version__)

# Check compatibility
assert numpy.__version__.startswith('1.23') or numpy.__version__.startswith('1.22'), 'numpy version should be 1.22 or 1.23'
assert scipy.__version__.startswith('1.'), 'scipy should be version 1.x'
assert tf.__version__.startswith('2.'), 'tensorflow should be version 2.x'

print('')
print('All core packages installed successfully!')
"

# Install bpy
echo ""
echo "Step 4: Installing bpy (Blender Python)..."
echo "This may take a while..."
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/

echo ""
echo "========================================"
echo "✅ Installation complete!"
echo "========================================"
echo ""
echo "You can now run the simulation:"
echo "  bash scripts/sim_data/generate_single_sample.sh"
echo ""
