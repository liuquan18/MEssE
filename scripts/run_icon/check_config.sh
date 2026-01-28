#!/bin/bash
# Check script for ICON LAM run configuration

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ICONDIR=$BASE_DIR/build/gcc
export EXPDIR=$BASE_DIR/experiment

echo "========================================="
echo "Checking ICON LAM Configuration"
echo "========================================="
echo ""

# Check 1: ICON executable
echo "1. Checking ICON executable..."
if [ -f "$ICONDIR/bin/icon" ]; then
    echo "   ✓ ICON executable found: $ICONDIR/bin/icon"
    ls -lh "$ICONDIR/bin/icon"
else
    echo "   ✗ ICON executable NOT found at: $ICONDIR/bin/icon"
    exit 1
fi

# Check 2: Environment script
echo ""
echo "2. Checking environment script..."
if [ -f "$BASE_DIR/environment/activate_levante_env" ]; then
    echo "   ✓ Environment script found: $BASE_DIR/environment/activate_levante_env"
else
    echo "   ✗ Environment script NOT found"
    exit 1
fi

# Check 3: Plugin library
echo ""
echo "3. Checking plugin library..."
PLUGIN_LIB="$ICONDIR/externals/comin/build/plugins/python_adapter/libpython_adapter.so"
if [ -f "$PLUGIN_LIB" ]; then
    echo "   ✓ Plugin library found: $PLUGIN_LIB"
else
    echo "   ✗ Plugin library NOT found at: $PLUGIN_LIB"
    exit 1
fi

# Check 4: Plugin script
echo ""
echo "4. Checking plugin script..."
PLUGIN_SCRIPT="$BASE_DIR/scripts/plugin/scripts/comin_plugin.py"
if [ -f "$PLUGIN_SCRIPT" ]; then
    echo "   ✓ Plugin script found: $PLUGIN_SCRIPT"
else
    echo "   ✗ Plugin script NOT found at: $PLUGIN_SCRIPT"
    exit 1
fi

# Check 5: NAMELIST file
echo ""
echo "5. Checking NAMELIST file..."
if [ -f "$EXPDIR/NAMELIST_ICON" ]; then
    echo "   ✓ NAMELIST_ICON found: $EXPDIR/NAMELIST_ICON"
    echo "   File size: $(du -h $EXPDIR/NAMELIST_ICON | cut -f1)"
else
    echo "   ✗ NAMELIST_ICON NOT found"
    exit 1
fi

# Check 6: SBATCH script
echo ""
echo "6. Checking SBATCH script..."
if [ -f "$EXPDIR/icon-lam.sbatch" ]; then
    echo "   ✓ SBATCH script found: $EXPDIR/icon-lam.sbatch"
else
    echo "   ✗ SBATCH script NOT found"
    exit 1
fi

# Check 7: Input data files
echo ""
echo "7. Checking input data files..."
if [ -f "$EXPDIR/init_ML_20210714T000000Z.nc" ]; then
    echo "   ✓ Initial condition file found"
else
    echo "   ⚠ Warning: Initial condition file may be missing"
fi

# Check 8: YAC installation
echo ""
echo "8. Checking YAC installation..."
if [ -d "$BASE_DIR/build/YAC/yac/install" ]; then
    echo "   ✓ YAC installed at: $BASE_DIR/build/YAC/yac/install"
else
    echo "   ⚠ Warning: YAC installation not found"
fi

if [ -d "$BASE_DIR/build/YAC/yaxt/install" ]; then
    echo "   ✓ YAXT installed at: $BASE_DIR/build/YAC/yaxt/install"
else
    echo "   ⚠ Warning: YAXT installation not found"
fi

echo ""
echo "========================================="
echo "Configuration Check Complete!"
echo "========================================="
echo ""
echo "To submit the job, run:"
echo "  cd $BASE_DIR/scripts/run_icon"
echo "  ./run_icon_LAM.sh"
echo ""
