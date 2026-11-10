"""Tests for fieldspace_RF_config.py: the file states a whole run, or it stops at startup."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from fieldspace_RF_config import POSITION_NAMES, load_config
from fieldspace_RF_online import OnlineReconstructorForecaster
from icon_nested_mgrid import patch_mgrid
from test_icon_nested_mgrid import synthetic_global_mgrid

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "fieldspace_rf_r2b8.yaml"
LON_LAT_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_lon_lat.yaml")


def _write(tmp_path, change):
    """The repository's config after ``change(raw)``, as a file."""
    raw = yaml.safe_load(REPO_CONFIG.read_text())
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("path", [REPO_CONFIG, LON_LAT_CONFIG], ids=["default", "lon_lat"])
def test_repository_config_builds_the_models(path):
    config = load_config(path)
    levels = config.grid.levels
    mgrids = patch_mgrid(synthetic_global_mgrid(n_levels=levels[-1] - levels[0] + 1), torch.arange(10))
    trainer = OnlineReconstructorForecaster(
        mgrids, levels, config.grid.latent_level, 1, device=torch.device("cpu"), **config.trainer_kwargs(config.run.sample_seconds)
    )
    latent = config.grid.latent_level - levels[0]
    assert trainer.model.reconstructor.latent_features[latent] == config.reconstructor.latent_channels


def test_model_times_become_sample_steps_and_forcing_names_become_columns(tmp_path):
    def change(raw):
        raw["forcings"].update(icon_fields=["a", "b"], insolation=True, statics=["c"], reconstructor=["a", "c"])
        raw["reconstructor"]["lr_warmup_seconds"] = 3000
        raw["forecaster"].update(replay_seconds=6000, replay_warmup_seconds=1200)

    kwargs = load_config(_write(tmp_path, change)).trainer_kwargs(600.0)
    assert (kwargs["lr_warmup"], kwargs["replay_size"], kwargs["replay_warmup"]) == (5, 10, 2)
    assert kwargs["n_forcings"] == 4 and kwargs["reconstructor_forcings"] == [0, 3]


def test_position_adds_three_columns_after_the_statics(tmp_path):
    def change(raw):
        raw["forcings"].update(
            icon_fields=["a"], insolation=True, statics=["c"], position=True, reconstructor=["a", "c", *POSITION_NAMES]
        )

    config = load_config(_write(tmp_path, change))
    assert config.forcings.names == ["a", "insolation", "c", "cos_lat_cos_lon", "cos_lat_sin_lon", "sin_lat"]
    kwargs = config.trainer_kwargs(600.0)
    assert kwargs["n_forcings"] == 6 and kwargs["reconstructor_forcings"] == [0, 2, 3, 4, 5]


def test_lon_lat_config_differs_from_the_default_only_by_the_position():
    default, lon_lat = load_config(REPO_CONFIG), load_config(LON_LAT_CONFIG)
    assert lon_lat.forcings.position and not default.forcings.position
    assert lon_lat.forcings.reconstructor == default.forcings.reconstructor + POSITION_NAMES
    assert replace(lon_lat, forcings=default.forcings) == default


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda raw: raw["reconstructor"].update(lr_recon=1e-3), "unknown keys \\['lr_recon'\\]"),
        (lambda raw: raw["forecaster"].pop("n_context"), "missing keys \\['n_context'\\]"),
        (lambda raw: raw["forcings"].pop("position"), "missing keys \\['position'\\]"),
        (lambda raw: raw["forcings"].update(reconstructor=["sin_lat"]), "forcings.reconstructor"),  # position: false
        (lambda raw: raw.update(extra={}), "unknown sections \\['extra'\\]"),
        (lambda raw: raw.pop("blocks"), "missing sections \\['blocks'\\]"),
        (lambda raw: raw["forcings"].update(reconstructor=["sst"]), "forcings.reconstructor"),
    ],
)
def test_wrong_files_stop_at_startup(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, change))


def test_exponents_without_a_dot_are_numbers(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("lr: 1.0e-3", "lr: 1e-3"))
    assert load_config(path).reconstructor.lr == 1e-3
