import sys
from pathlib import Path

# Two different import styles are in play, so tests need two different
# directories on sys.path:
#   - MEssE/comin_plugin_torch/*.py (graph_utils, gnn_online, ...) are
#     imported by bare module name, matching how COMIN execs the plugin
#     files themselves (flat, no package machinery) — needs
#     comin_plugin_torch/ on sys.path.
#   - MEssE/utils/*.py (icon_online_helper, healpix_grids) are imported as
#     a real package, `MEssE.utils.*` — needs the repo root (the parent of
#     MEssE/) on sys.path.
_TESTS_DIR = Path(__file__).resolve().parent
_MESSE_DIR = _TESTS_DIR.parent
_PROJECT_ROOT = _MESSE_DIR.parent
_COMIN_PLUGIN_TORCH_DIR = _MESSE_DIR / "comin_plugin_torch"

for _p in (_PROJECT_ROOT, _COMIN_PLUGIN_TORCH_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
