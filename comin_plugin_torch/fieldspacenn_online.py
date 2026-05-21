import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _ensure_fieldspacenn_on_path() -> None:
    """Make the sibling FieldSpaceNN checkout importable when it is not installed.

    Override the auto-detected path by setting the ``FIELDSPACENN_ROOT`` environment
    variable to the absolute path of the FieldSpaceNN checkout directory.
    """
    env_root = os.environ.get("FIELDSPACENN_ROOT", "")
    if env_root:
        fieldspacenn_root = Path(env_root).resolve()
    else:
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
    if not config_path.suffix:
        config_path = config_path.with_suffix(".yaml")
    return OmegaConf.load(config_path)


@dataclass
class FieldSpaceNNSnapshot:
    x_zooms_groups: List[Dict[int, torch.Tensor]]
    mask_zooms_groups: List[Dict[int, torch.Tensor]]
    emb_groups: List[Dict[str, Any]]
    sample_configs: Dict[int, Dict[str, Any]]
    unix_seconds: float


class OnlineFieldSpaceNNTrainer:
    """GPU-local FieldSpaceNN trainer used from the COMIN callback."""

    def __init__(
        self,
        cfg: Any,
        owned_face_ids: torch.Tensor,
        nlev_groups: List[int],
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
        self.nlev_groups = [int(n) for n in nlev_groups]
        self.hpx_level = int(cfg.online.hpx_level)
        self.in_zooms = [int(z) for z in cfg.online.in_zooms]
        self.max_zoom = max(self.in_zooms)
        self.patch_zoom = int(cfg.online.patch_zoom)
        self.owned_face_ids = owned_face_ids.to(self.device, dtype=torch.long)
        self.encode_zooms = encode_zooms
        self._input_shape_logged = False

        if self.max_zoom != self.hpx_level:
            raise ValueError(
                f"online.hpx_level ({self.hpx_level}) must equal max(online.in_zooms) "
                f"({self.max_zoom}); the COMIN regridder produces max-zoom tensors directly."
            )

        sampling_zooms = OmegaConf.to_container(cfg.sampling_zooms, resolve=True)
        self.base_sample_configs = self._load_sampling_configs(sampling_zooms)

        self.model = instantiate(cfg.model).to(self.device)
        self.forward_model = self._wrap_ddp(cfg, use_ddp)
        self.optimizer = instantiate(cfg.optimizer, params=self.forward_model.parameters())

        loss_cfg = OmegaConf.to_container(cfg.loss.zooms, resolve=True)
        self.criterion = MGMultiLoss(loss_cfg, grid_layers=self.model.grid_layers).to(
            self.device
        )

    def _wrap_ddp(self, cfg: Any, use_ddp: Optional[bool]) -> torch.nn.Module:
        if use_ddp is None:
            use_ddp = dist.is_available() and dist.is_initialized()
        if not (bool(cfg.ddp.enabled) and bool(use_ddp)):
            return self.model

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        wrap_single = bool(cfg.ddp.get("wrap_single_process", False))
        if world_size <= 1 and not wrap_single:
            return self.model

        return DDP(
            self.model,
            device_ids=[self.device.index or 0],
            broadcast_buffers=False,
            find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
        )

    def _load_sampling_configs(
        self, sampling_zooms: Mapping[Any, Any]
    ) -> Dict[int, Dict[str, Any]]:
        return {
            int(zoom_key): {
                "n_past_ts": int(conf.get("n_past_ts", 0)),
                "n_future_ts": int(conf.get("n_future_ts", 0)),
                "zoom_patch_sample": int(conf.get("zoom_patch_sample", self.patch_zoom)),
                "mask_n_last_ts": int(conf.get("mask_n_last_ts", 0)),
            }
            for zoom_key, conf in sampling_zooms.items()
        }

    def _sample_configs_with_patch_index(self) -> Dict[int, Dict[str, Any]]:
        return {
            zoom: {**conf, "patch_index": self.owned_face_ids}
            for zoom, conf in self.base_sample_configs.items()
        }

    def _faces_to_max_zoom_tensor(self, ua_faces: torch.Tensor) -> torch.Tensor:
        # ua_faces: (faces, nlev, nside, nside)
        faces, nlev, nside, _ = ua_faces.shape
        expected_nside = 2**self.hpx_level
        if nside != expected_nside:
            raise ValueError(
                f"Expected HEALPix nside={expected_nside}, got {nside}."
            )
        # Reshape to FieldSpaceNN layout: (faces, variables=1, time=1, pixels, depth, features=1)
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
            assert n % scale == 0, (
                f"Pixel count {n} is not divisible by downscale factor {scale} "
                f"(in_zoom={in_zoom}, out_zoom={out_zoom})."
            )
            return x.view(b, v, t, n // scale, scale, d, f).mean(dim=4)
        scale = 4 ** (out_zoom - in_zoom)
        return (
            x.view(b, v, t, n, 1, d, f)
            .expand(b, v, t, n, scale, d, f)
            .reshape(b, v, t, n * scale, d, f)
        )

    def _make_embeddings(self, unix_seconds: float) -> Dict[str, Any]:
        batch = self.owned_face_ids.numel()
        time_value = torch.full(
            (batch, 1), unix_seconds, device=self.device, dtype=torch.float32
        )
        return {
            "TimeEmbedder": {zoom: time_value for zoom in self.in_zooms},
            "VariableEmbedder": torch.zeros(
                (batch, 1), device=self.device, dtype=torch.long
            ),
        }

    @torch.no_grad()
    def prepare_snapshot(
        self, faces_list: List[torch.Tensor], unix_seconds: float
    ) -> FieldSpaceNNSnapshot:
        sample_configs = self._sample_configs_with_patch_index()
        patch_index_zooms = {zoom: self.owned_face_ids for zoom in self.in_zooms}

        x_zooms_groups = []
        mask_zooms_groups = []
        emb_groups = []

        for ua_faces in faces_list:
            ua_faces = ua_faces.to(self.device, dtype=torch.float32, non_blocking=True)
            max_zoom_tensor = self._faces_to_max_zoom_tensor(ua_faces)

            x_zooms = {
                zoom: self._rescale_zoom(max_zoom_tensor, self.max_zoom, zoom).contiguous()
                for zoom in self.in_zooms
            }
            x_zooms = self.encode_zooms(x_zooms, sample_configs, patch_index_zooms)
            x_zooms = {zoom: t.detach().contiguous() for zoom, t in x_zooms.items()}
            mask_zooms = {
                zoom: torch.zeros_like(t, dtype=torch.bool) for zoom, t in x_zooms.items()
            }

            x_zooms_groups.append(x_zooms)
            mask_zooms_groups.append(mask_zooms)
            emb_groups.append(self._make_embeddings(unix_seconds))

        if not self._input_shape_logged:
            for g, xz in enumerate(x_zooms_groups):
                shapes = ", ".join(
                    f"z{z}={tuple(xz[z].shape)}" for z in sorted(xz)
                )
                self.log_fn(f"[rank={self.rank}] Group {g} input shapes: {shapes}")
            self._input_shape_logged = True

        return FieldSpaceNNSnapshot(
            x_zooms_groups=x_zooms_groups,
            mask_zooms_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
            unix_seconds=float(unix_seconds),
        )

    def train_step(
        self, source: FieldSpaceNNSnapshot, target: FieldSpaceNNSnapshot
    ) -> Dict[str, Any]:
        self.forward_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.forward_model(
            x_zooms_groups=source.x_zooms_groups,
            mask_zooms_groups=source.mask_zooms_groups,
            emb_groups=target.emb_groups,
            sample_configs=target.sample_configs,
        )

        losses = []
        total_loss_dict = {}
        for i, (output, x_zooms_tgt, mask_zooms_src, emb_tgt) in enumerate(zip(
            outputs, target.x_zooms_groups, source.mask_zooms_groups, target.emb_groups
        )):
            loss_i, loss_dict_i = self.criterion(
                output,
                x_zooms_tgt,
                mask=mask_zooms_src,
                sample_configs=target.sample_configs,
                prefix="train/",
                emb=emb_tgt,
            )
            losses.append(loss_i)
            total_loss_dict.update({f"g{i}/{k}": v for k, v in loss_dict_i.items()})

        loss = sum(losses)
        if not torch.is_tensor(loss):
            raise RuntimeError(
                "FieldSpaceNN loss config did not produce a tensor loss."
            )

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_dict": {k: float(v) for k, v in total_loss_dict.items()},
        }
