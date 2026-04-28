import os
import sys
import types
import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def _resolve_fieldspacenn_repo() -> Path:
    env_path = os.environ.get("FIELDSPACENN_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    this_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            this_dir.parents[1] / "FieldSpaceNN",
            this_dir.parents[0] / "FieldSpaceNN",
            this_dir.parents[2] / "FieldSpaceNN",
        ]
    )

    for candidate in candidates:
        if (candidate / "fieldspacenn").is_dir():
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise ImportError(
        "Cannot locate FieldSpaceNN repository. Set FIELDSPACENN_PATH or place "
        f"FieldSpaceNN next to MEssE. Searched: {searched}"
    )


def _import_fieldspacenn_symbols() -> Tuple[Any, Any, Any]:
    try:
        from fieldspacenn.src.models.mg_transformer.mg_transformer import MG_Transformer
        from fieldspacenn.src.modules.field_space.field_space_attention import (
            FieldSpaceAttentionConfig,
        )
        from fieldspacenn.src.modules.grids.grid_utils import healpix_grid_to_mgrid

        return MG_Transformer, FieldSpaceAttentionConfig, healpix_grid_to_mgrid
    except Exception:
        repo_root = _resolve_fieldspacenn_repo()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # Bypass fieldspacenn.__init__ (which imports the training stack) and
        # expose the package as a namespace rooted at the cloned repository.
        pkg_name = "fieldspacenn"
        pkg_dir = repo_root / pkg_name
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            pkg.__file__ = str(pkg_dir / "__init__.py")
            sys.modules[pkg_name] = pkg

        try:
            MG_Transformer = importlib.import_module(
                "fieldspacenn.src.models.mg_transformer.mg_transformer"
            ).MG_Transformer
            FieldSpaceAttentionConfig = importlib.import_module(
                "fieldspacenn.src.modules.field_space.field_space_attention"
            ).FieldSpaceAttentionConfig
            healpix_grid_to_mgrid = importlib.import_module(
                "fieldspacenn.src.modules.grids.grid_utils"
            ).healpix_grid_to_mgrid
            return MG_Transformer, FieldSpaceAttentionConfig, healpix_grid_to_mgrid
        except Exception as exc:
            raise ImportError(
                "FieldSpaceNN import failed after repository discovery. "
                "Install required runtime packages in the MEssE environment "
                "(at least: omegaconf, einops, healpy) or set FIELDSPACENN_PATH. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc


class FieldSpaceAttentionWrapper(nn.Module):
    """Adapter exposing a Conv-like API over a single-zoom field-space attention model.

    Input/Output tensors follow the plugin contract:
    - ua:       (faces_per_rank, nlev, nside, nside)
    - calendar: (faces_per_rank, cal_channels, nside, nside)
    - output:   (faces_per_rank, nlev, nside, nside)
    """

    def __init__(
        self,
        nlev: int,
        calendar_channels: int,
        hpx_level: int,
        first_face: int,
        faces_per_rank: int,
        att_dim: int = 32,
        n_head_channels: int = 16,
    ):
        super().__init__()

        if nlev <= 0:
            raise ValueError(f"nlev must be positive, got {nlev}")
        if calendar_channels <= 0:
            raise ValueError(
                f"calendar_channels must be positive, got {calendar_channels}"
            )
        if faces_per_rank <= 0:
            raise ValueError(f"faces_per_rank must be positive, got {faces_per_rank}")

        self.nlev = int(nlev)
        self.calendar_channels = int(calendar_channels)
        self.hpx_level = int(hpx_level)
        self.first_face = int(first_face)
        self.faces_per_rank = int(faces_per_rank)
        self.nside = 2**self.hpx_level

        # Calendar enters as a separate branch and is projected to ua channels before fusion.
        self.calendar_projection = nn.Conv2d(
            self.calendar_channels, self.nlev, kernel_size=1, bias=True
        )

        MG_Transformer, FieldSpaceAttentionConfig, healpix_grid_to_mgrid = (
            _import_fieldspacenn_symbols()
        )

        mgrids = healpix_grid_to_mgrid(zoom_max=self.hpx_level)
        block_conf = FieldSpaceAttentionConfig(
            token_zoom=self.hpx_level,
            q_zooms=[self.hpx_level],
            kv_zooms=[self.hpx_level],
            target_zooms=[self.hpx_level],
            att_dim=att_dim,
            token_len_depth=1,
            token_len_time=1,
            token_overlap_space=False,
            token_overlap_time=False,
            token_overlap_depth=False,
            token_overlap_mlp_time=False,
            token_overlap_mlp_depth=False,
            seq_len_zoom=-1,
            seq_len_time=-1,
            seq_len_depth=-1,
            with_var_att=False,
            shift=False,
            multi_shift=False,
            update="shift",
            separate_mlp_norm=True,
        )

        self.model = MG_Transformer(
            mgrids=mgrids,
            block_configs={"0": block_conf},
            in_zooms=[self.hpx_level],
            in_features=1,
            n_groups_variables=[1],
            n_head_channels=n_head_channels,
            att_dim=att_dim,
            use_mask=False,
        )

    @staticmethod
    def _ua_to_field_space(ua_face: torch.Tensor) -> torch.Tensor:
        """Convert (b, nlev, nside, nside) to (b, v, t, n, d, f)."""
        b, nlev, nside, _ = ua_face.shape
        cells_depth = ua_face.permute(0, 2, 3, 1).reshape(b, nside * nside, nlev)
        return cells_depth.unsqueeze(1).unsqueeze(2).unsqueeze(-1).contiguous()

    @staticmethod
    def _field_space_to_ua(
        x_field: torch.Tensor, nlev: int, nside: int
    ) -> torch.Tensor:
        """Convert (b, v, t, n, d, f) to (b, nlev, nside, nside)."""
        cells_depth = x_field.squeeze(1).squeeze(1).squeeze(-1).contiguous()
        return (
            cells_depth.view(cells_depth.shape[0], nside, nside, nlev)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _face_sample_config(self, global_face_index: int) -> Dict[int, Dict[str, int]]:
        return {
            self.hpx_level: {
                "patch_index": int(global_face_index),
                "zoom_patch_sample": 0,
            }
        }

    def forward(self, ua: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        if ua.ndim != 4:
            raise ValueError(
                f"ua must be 4D (faces, nlev, nside, nside), got {ua.shape}"
            )
        if calendar.ndim != 4:
            raise ValueError(
                "calendar must be 4D (faces, cal_channels, nside, nside), "
                f"got {calendar.shape}"
            )

        if ua.shape[0] != calendar.shape[0]:
            raise ValueError(
                "ua and calendar must have the same face count, "
                f"got {ua.shape[0]} vs {calendar.shape[0]}"
            )
        if ua.shape[1] != self.nlev:
            raise ValueError(f"expected nlev={self.nlev}, got {ua.shape[1]}")
        if calendar.shape[1] != self.calendar_channels:
            raise ValueError(
                f"expected calendar channels={self.calendar_channels}, got {calendar.shape[1]}"
            )
        if ua.shape[-1] != self.nside or ua.shape[-2] != self.nside:
            raise ValueError(
                f"expected ua spatial size ({self.nside}, {self.nside}), got {ua.shape[-2:]}"
            )
        if calendar.shape[-2:] != ua.shape[-2:]:
            raise ValueError(
                f"calendar spatial size {calendar.shape[-2:]} must match ua {ua.shape[-2:]}"
            )

        calendar_bias = self.calendar_projection(calendar)
        fused_ua = ua + calendar_bias

        outputs = []
        for local_face_idx in range(fused_ua.shape[0]):
            global_face_index = self.first_face + local_face_idx
            x_face = fused_ua[local_face_idx : local_face_idx + 1]
            x_field = self._ua_to_field_space(x_face)

            sample_configs = self._face_sample_config(global_face_index)
            masks = {self.hpx_level: torch.zeros_like(x_field, dtype=torch.bool)}

            out_groups = self.model(
                x_zooms_groups=[{self.hpx_level: x_field}],
                mask_zooms_groups=[masks],
                emb_groups=[{}],
                sample_configs=sample_configs,
            )
            out_field = out_groups[0][self.hpx_level]
            out_face = self._field_space_to_ua(
                out_field, nlev=self.nlev, nside=self.nside
            )
            outputs.append(out_face)

        out = torch.cat(outputs, dim=0)
        if out.shape != ua.shape:
            raise RuntimeError(
                f"wrapper output shape {out.shape} does not match ua shape {ua.shape}"
            )

        return out
