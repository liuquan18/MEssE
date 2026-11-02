"""Configuration of fieldspace_RF_plugin.py, read from one YAML file
(``config/fieldspace_rf_r2b8.yaml``; the runscript sets ``MESSE_RF_CONFIG``).

There are no defaults here: every key is required and unknown keys are errors,
so a file always states a whole run and a typo stops it at startup, not after
the dry run. The attention and compress blocks are not listed in the file:
`fieldspace_RF_online.autoencoder_configs` derives them from the levels and
channels, so their zoom lists always agree.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class Run:
    icon_var: str
    sample_seconds: float
    dry_run_seconds: float
    save_interval_seconds: float
    log_seconds: float


@dataclass(frozen=True)
class Grid:
    dir: str
    levels: List[int]
    latent_level: int


@dataclass(frozen=True)
class Forcings:
    icon_fields: List[str]
    insolation: bool
    statics: List[str]
    static_file: str
    reconstructor: List[str]

    @property
    def names(self) -> List[str]:
        """All forcings in column order; the forecaster gets all of them."""
        return self.icon_fields + ["insolation"] * self.insolation + self.statics


@dataclass(frozen=True)
class Blocks:
    att_dim: int
    n_head_channels: int
    hidden_dim: int
    embed_dim: int


@dataclass(frozen=True)
class Training:
    grad_clip: float


@dataclass(frozen=True)
class Reconstructor:
    n_blocks: int
    latent_channels: int
    inner_channels: Optional[int]
    time_input: bool
    ema_decay: float
    lr: float
    lr_warmup_seconds: float


@dataclass(frozen=True)
class Forecaster:
    n_blocks: int
    n_context: int
    lr: float
    replay_seconds: float
    replay_warmup_seconds: float
    replay_batch: int


@dataclass(frozen=True)
class Config:
    run: Run
    grid: Grid
    forcings: Forcings
    blocks: Blocks
    training: Training
    reconstructor: Reconstructor
    forecaster: Forecaster

    def __post_init__(self):
        unknown = set(self.forcings.reconstructor) - set(self.forcings.names)
        if unknown:
            raise ValueError(f"forcings.reconstructor: not among {self.forcings.names}: {sorted(unknown)}")

    def trainer_kwargs(self, sample_seconds: float) -> Dict:
        """Keywords of `OnlineReconstructorForecaster`, with model times turned
        into sample steps of the realized interval."""
        steps = lambda seconds: int(seconds // sample_seconds)  # noqa: E731
        r, f, names = self.reconstructor, self.forecaster, self.forcings.names
        return dict(
            n_context=f.n_context,
            n_forcings=len(names),
            reconstructor_forcings=[names.index(name) for name in self.forcings.reconstructor],
            latent_channels=r.latent_channels,
            inner_channels=r.inner_channels,
            n_blocks=r.n_blocks,
            n_forecaster_blocks=f.n_blocks,
            att_dim=self.blocks.att_dim,
            n_head_channels=self.blocks.n_head_channels,
            hidden_dim=self.blocks.hidden_dim,
            embed_dim=self.blocks.embed_dim,
            ema_decay=r.ema_decay,
            reconstructor_time=r.time_input,
            lr_recon=r.lr,
            lr_forecast=f.lr,
            lr_warmup=steps(r.lr_warmup_seconds),
            replay_size=steps(f.replay_seconds),
            replay_warmup=steps(f.replay_warmup_seconds),
            replay_batch=f.replay_batch,
            grad_clip=self.training.grad_clip,
        )


def _section(cls, values: Dict, name: str):
    """``cls`` from a YAML mapping with exactly its keys. YAML reads ``1e-3``
    without a dot as a string, so float fields are converted."""
    keys = {field.name: field.type for field in fields(cls)}
    unknown, missing = sorted(set(values) - set(keys)), sorted(set(keys) - set(values))
    if unknown or missing:
        raise ValueError(f"{name}: unknown keys {unknown}, missing keys {missing}")
    return cls(**{key: float(value) if keys[key] is float else value for key, value in values.items()})


def load_config(path: str) -> Config:
    with open(path) as file:
        raw = yaml.safe_load(file)
    sections = {field.name: field.type for field in fields(Config)}
    unknown, missing = sorted(set(raw) - set(sections)), sorted(set(sections) - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown sections {unknown}, missing sections {missing}")
    return Config(**{name: _section(cls, raw[name], name) for name, cls in sections.items()})
