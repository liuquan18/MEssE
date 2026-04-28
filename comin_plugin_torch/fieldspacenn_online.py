import copy
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _ensure_fieldspacenn_on_path() -> None:
    """Make the sibling FieldSpaceNN checkout importable when it is not installed."""
    src_root = Path(__file__).resolve().parents[2]
    fieldspacenn_root = src_root / "FieldSpaceNN"
    if fieldspacenn_root.exists():
        path = str(fieldspacenn_root)
        if path not in sys.path:
            sys.path.insert(0, path)


def load_online_config():
    """Load the MEssE FieldSpaceNN online-training config."""
    from omegaconf import OmegaConf

    default_dir = Path(__file__).resolve().parents[1] / "config"
    config_dir = Path(os.environ.get("MESSE_CONFIG_DIR", str(default_dir)))
    config_name = os.environ.get("MESSE_CONFIG_NAME", "fieldspacenn_online")
    config_path = config_dir / config_name
    if config_path.suffix == "":
        config_path = config_path.with_suffix(".yaml")
    return OmegaConf.load(config_path)


class FieldSpaceNNSnapshot:
    def __init__(
        self,
        x_zooms: Dict[int, torch.Tensor],
        mask_zooms: Dict[int, torch.Tensor],
        emb_group: Dict[str, Any],
        sample_configs: Dict[int, Dict[str, Any]],
        unix_seconds: float,
    ) -> None:
        self.x_zooms = x_zooms
        self.mask_zooms = mask_zooms
        self.emb_group = emb_group
        self.sample_configs = sample_configs
        self.unix_seconds = unix_seconds


class OnlineFieldSpaceNNTrainer:
    """GPU-local FieldSpaceNN trainer used from the COMIN callback."""

    def __init__(
        self,
        cfg: Any,
        owned_face_ids: torch.Tensor,
        nlev: int,
        device: Optional[torch.device] = None,
        use_ddp: Optional[bool] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        rank: Optional[int] = None,
    ) -> None:
        _ensure_fieldspacenn_on_path()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fieldspacenn.src.utils.losses import MGMultiLoss
        from fieldspacenn.src.modules.grids.grid_utils import encode_zooms

        self.cfg = cfg
        self.device = torch.device("cuda", 0) if device is None else device
        self.rank = rank
        self.log_fn = log_fn or (lambda msg: None)
        self.nlev = int(nlev)
        self.hpx_level = int(cfg.online.hpx_level)
        self.in_zooms = [int(z) for z in cfg.online.in_zooms]
        self.max_zoom = max(self.in_zooms)
        self.patch_zoom = int(cfg.online.patch_zoom)
        self.forecast_horizon_steps = int(cfg.online.forecast_horizon_steps)
        self.owned_face_ids = owned_face_ids.to(self.device, dtype=torch.long)
        self.encode_zooms = encode_zooms

        if self.max_zoom != self.hpx_level:
            raise ValueError(
                "The current COMIN regridder produces the max model zoom directly; "
                "set online.hpx_level equal to max(online.in_zooms)."
            )

        sampling_zooms = OmegaConf.to_container(cfg.sampling_zooms, resolve=True)
        self.base_sample_configs = self._load_sampling_configs(sampling_zooms)

        model = instantiate(cfg.model)
        self.model = model.to(self.device)

        if use_ddp is None:
            use_ddp = dist.is_available() and dist.is_initialized()
        ddp_enabled = bool(cfg.ddp.enabled) and bool(use_ddp)
        wrap_single = bool(cfg.ddp.get("wrap_single_process", False))
        world_size = dist.get_world_size() if ddp_enabled and dist.is_initialized() else 1
        if ddp_enabled and (world_size > 1 or wrap_single):
            self.forward_model = DDP(
                self.model,
                device_ids=[self.device.index or 0],
                broadcast_buffers=False,
                find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
            )
        else:
            self.forward_model = self.model

        optimizer_cfg = cfg.optimizer
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(),
            lr=float(optimizer_cfg.lr),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
        )

        loss_cfg = OmegaConf.to_container(cfg.loss.zooms, resolve=True)
        self.loss_zooms = MGMultiLoss(loss_cfg, grid_layers=self.model.grid_layers).to(
            self.device
        )
        self._shape_logged = False

    def _load_sampling_configs(self, sampling_zooms: Mapping[Any, Any]) -> Dict[int, Dict[str, Any]]:
        out = {}
        for zoom_key, conf in sampling_zooms.items():
            zoom = int(zoom_key)
            out[zoom] = {
                "n_past_ts": int(conf.get("n_past_ts", 0)),
                "n_future_ts": int(conf.get("n_future_ts", 0)),
                "zoom_patch_sample": int(conf.get("zoom_patch_sample", self.patch_zoom)),
                "mask_n_last_ts": int(conf.get("mask_n_last_ts", 0)),
            }
        return out

    def _sample_configs_with_patch_index(self) -> Dict[int, Dict[str, Any]]:
        sample_configs = {}
        for zoom, conf in self.base_sample_configs.items():
            sample_configs[zoom] = dict(conf)
            sample_configs[zoom]["patch_index"] = self.owned_face_ids
        return sample_configs

    def _patch_index_zooms(self) -> Dict[int, torch.Tensor]:
        return {zoom: self.owned_face_ids for zoom in self.in_zooms}

    def _faces_to_max_zoom_tensor(self, ua_faces: torch.Tensor) -> torch.Tensor:
        # Input: (faces, nlev, nside, nside)
        faces, nlev, nside, _ = ua_faces.shape
        expected_nside = 2 ** self.hpx_level
        if int(nside) != expected_nside:
            raise ValueError(
                "Unexpected HEALPix nside: got {}, expected {}".format(
                    int(nside), expected_nside
                )
            )
        x = ua_faces.reshape(faces, nlev, nside * nside).permute(0, 2, 1)
        return x.contiguous().view(faces, 1, 1, nside * nside, nlev, 1)

    def _rescale_zoom(
        self, x: torch.Tensor, in_zoom: int, out_zoom: int
    ) -> torch.Tensor:
        if in_zoom == out_zoom:
            return x

        b, v, t, n, d, f = x.shape
        if in_zoom > out_zoom:
            scale = 4 ** (in_zoom - out_zoom)
            return x.view(b, v, t, n // scale, scale, d, f).mean(dim=4)

        scale = 4 ** (out_zoom - in_zoom)
        return (
            x.view(b, v, t, n, 1, d, f)
            .expand(b, v, t, n, scale, d, f)
            .reshape(b, v, t, n * scale, d, f)
        )

    def _make_embeddings(self, unix_seconds: float) -> Dict[str, Any]:
        batch = int(self.owned_face_ids.numel())
        time_value = torch.full(
            (batch, 1), float(unix_seconds), device=self.device, dtype=torch.float32
        )
        return {
            "TimeEmbedder": {zoom: time_value for zoom in self.in_zooms},
            "VariableEmbedder": torch.zeros(
                (batch, 1), device=self.device, dtype=torch.long
            ),
        }

    def prepare_snapshot(
        self, ua_faces: torch.Tensor, unix_seconds: float
    ) -> FieldSpaceNNSnapshot:
        ua_faces = ua_faces.to(self.device, dtype=torch.float32, non_blocking=True)
        max_zoom_tensor = self._faces_to_max_zoom_tensor(ua_faces)

        x_zooms = {}
        for zoom in self.in_zooms:
            x_zooms[zoom] = self._rescale_zoom(
                max_zoom_tensor, self.max_zoom, zoom
            ).contiguous()

        sample_configs = self._sample_configs_with_patch_index()
        x_zooms = self.encode_zooms(
            x_zooms, sample_configs, self._patch_index_zooms()
        )
        x_zooms = {
            zoom: tensor.detach().contiguous() for zoom, tensor in x_zooms.items()
        }
        mask_zooms = {
            zoom: torch.zeros_like(tensor, dtype=torch.bool)
            for zoom, tensor in x_zooms.items()
        }
        emb_group = self._make_embeddings(unix_seconds)

        if not self._shape_logged:
            shapes = ", ".join(
                "z{}={}".format(zoom, tuple(x_zooms[zoom].shape))
                for zoom in sorted(x_zooms)
            )
            self.log_fn("[rank={}] FieldSpaceNN input shapes: {}".format(self.rank, shapes))
            self._shape_logged = True

        return FieldSpaceNNSnapshot(
            x_zooms=x_zooms,
            mask_zooms=mask_zooms,
            emb_group=emb_group,
            sample_configs=sample_configs,
            unix_seconds=float(unix_seconds),
        )

    def train_step(
        self, source: FieldSpaceNNSnapshot, target: FieldSpaceNNSnapshot
    ) -> Dict[str, Any]:
        self.forward_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.forward_model(
            x_zooms_groups=[source.x_zooms],
            mask_zooms_groups=[source.mask_zooms],
            emb_groups=[target.emb_group],
            sample_configs=target.sample_configs,
        )
        output = outputs[0]
        loss, loss_dict = self.loss_zooms(
            output,
            target.x_zooms,
            mask=source.mask_zooms,
            sample_configs=target.sample_configs,
            prefix="train/",
            emb=target.emb_group,
        )
        if not torch.is_tensor(loss):
            raise RuntimeError("FieldSpaceNN loss config did not produce a tensor loss.")

        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "loss_dict": copy.deepcopy(loss_dict),
        }
