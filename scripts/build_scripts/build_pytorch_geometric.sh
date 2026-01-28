#!/bin/bash
# Install PyTorch Geometric for GNN support

echo "=========================================="
echo "Installing PyTorch Geometric (PyG)"
echo "=========================================="
echo ""

# Activate Python virtual environment
source /work/mh1498/m301257/work/MEssE/environment/python/py_venv/bin/activate

# Check Python and PyTorch versions
echo "Checking environment..."
python --version
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo ""

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
echo ""

# Method 1: Using pip (recommended for CPU version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch-geometric

# Verify installation
echo ""
echo "Verifying installation..."
python << 'EOF'
try:
    import torch
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import knn_graph
    
    print("✓ PyTorch Geometric successfully installed!")
    print(f"  Version: {torch_geometric.__version__}")
    print("  Available modules:")
    print("    - GCNConv (Graph Convolutional Network)")
    print("    - GATConv (Graph Attention Network)")
    print("    - knn_graph (k-NN graph construction)")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    pos = torch.randn(100, 2)
    edge_index = knn_graph(pos, k=6)
    print(f"  ✓ Created test graph: 100 nodes, {edge_index.shape[1]} edges")
    
except Exception as e:
    print(f"✗ Installation failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "You can now use the GNN-enabled plugin:"
    echo "  comin_plugin_gnn.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Installation failed!"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if pip is working: pip --version"
    echo "2. Try manual installation:"
    echo "   pip install torch-geometric"
    echo "3. Check PyTorch version compatibility"
    echo ""
    exit 1
fi
