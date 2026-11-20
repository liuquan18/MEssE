"""Tests for fieldspace_RF_config.py: the file states a whole run, or it stops at startup."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from fieldspace_RF_config import ORIENTATION_NAMES, POSITION_NAMES, load_config
from fieldspace_RF_online import OnlineReconstructorForecaster
from icon_nested_mgrid import ghost_cells, patch_mgrid
from icosahedron import icosahedral_mgrid
from test_icon_nested_mgrid import synthetic_global_mgrid

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "fieldspace_rf_r2b8.yaml"
LON_LAT_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_lon_lat.yaml")
NEIGHBOURS_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_neighbours.yaml")
LOCAL_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_neighbours_local.yaml")
LOCAL_POS_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_neighbours_local_pos.yaml")
LOCAL_T3_CONFIG = REPO_CONFIG.with_name("fieldspace_rf_r2b8_local_t3.yaml")


def _write(tmp_path, change):
    """The repository's config after ``change(raw)``, as a file."""
    raw = yaml.safe_load(REPO_CONFIG.read_text())
    change(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    "path",
    [REPO_CONFIG, LON_LAT_CONFIG, NEIGHBOURS_CONFIG, LOCAL_CONFIG, LOCAL_POS_CONFIG],
    ids=["default", "lon_lat", "neighbours", "local", "local_pos"],
)
def test_repository_config_builds_the_models(path):
    config = load_config(path)
    levels = config.grid.levels
    n_zooms = levels[-1] - levels[0] + 1
    if config.grid.canonical_order:  # an owned backbone cell with all 12 neighbours, and these as ghost cells
        globe = icosahedral_mgrid(n_zooms, n_zooms)
        owned = torch.nonzero(~globe[0]["adjc_mask"].any(1))[:1, 0]
        backbone = torch.cat([owned, torch.from_numpy(ghost_cells(owned.numpy(), globe[0]["adjc"].numpy()))])
    else:
        globe, owned, backbone = synthetic_global_mgrid(n_levels=n_zooms), torch.arange(10), torch.arange(10)
    trainer = OnlineReconstructorForecaster(
        patch_mgrid(globe, backbone),
        levels,
        config.grid.latent_level,
        1,
        n_owned=len(owned),
        device=torch.device("cpu"),
        **config.trainer_kwargs(config.run.sample_seconds),
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


def test_orientation_adds_two_columns_last(tmp_path):
    def change(raw):
        raw["grid"]["canonical_order"] = True
        raw["forcings"].update(icon_fields=["a"], insolation=False, statics=[], position=True, orientation=True, reconstructor=ORIENTATION_NAMES)

    config = load_config(_write(tmp_path, change))
    assert config.forcings.names == ["a", *POSITION_NAMES, "frame_north", "frame_east"]
    assert config.trainer_kwargs(600.0)["reconstructor_forcings"] == [4, 5]


def test_neighbours_config_differs_from_the_default_only_by_the_new_switches():
    default, neighbours = load_config(REPO_CONFIG), load_config(NEIGHBOURS_CONFIG)
    assert neighbours.grid.canonical_order and neighbours.grid.ghost_rings == 1 and neighbours.forcings.orientation
    assert neighbours.reconstructor.token_overlap_space and neighbours.forecaster.token_overlap_space
    assert neighbours.forcings.reconstructor == default.forcings.reconstructor + ORIENTATION_NAMES
    kwargs = neighbours.trainer_kwargs(600.0)
    assert kwargs["reconstructor_overlap"] and kwargs["forecaster_overlap"]
    same = replace(
        neighbours,
        grid=default.grid,
        forcings=default.forcings,
        reconstructor=replace(neighbours.reconstructor, token_overlap_space=False),
        forecaster=replace(neighbours.forecaster, token_overlap_space=False),
    )
    assert same == default
    assert not any(load_config(REPO_CONFIG).trainer_kwargs(600.0)[key] for key in ("reconstructor_overlap", "forecaster_overlap"))


def test_local_configs_differ_from_neighbours_by_the_attention_and_the_position():
    neighbours, local, local_pos = load_config(NEIGHBOURS_CONFIG), load_config(LOCAL_CONFIG), load_config(LOCAL_POS_CONFIG)
    assert local.reconstructor.attention == local_pos.reconstructor.attention == "local"
    assert neighbours.reconstructor.attention == "global"
    assert local.trainer_kwargs(600.0)["reconstructor_attention"] == "local"
    assert replace(local, reconstructor=neighbours.reconstructor) == neighbours
    # The position run: the lon_lat branch's three columns, in both models.
    assert local_pos.forcings.position and not local.forcings.position
    assert local_pos.forcings.names == local.forcings.names[:4] + POSITION_NAMES + ORIENTATION_NAMES
    assert local_pos.forcings.reconstructor == ["ts_wtr", "sic", "oromea", *POSITION_NAMES, *ORIENTATION_NAMES]
    assert replace(local_pos, forcings=local.forcings) == local
    assert all(load_config(p).reconstructor.attention == "global" for p in (REPO_CONFIG, LON_LAT_CONFIG))


def test_local_t3_config_differs_from_local_pos_only_by_the_forecasters_time_tokens_and_scope():
    local_pos, local_t3 = load_config(LOCAL_POS_CONFIG), load_config(LOCAL_T3_CONFIG)
    assert (local_t3.forecaster.n_context, local_t3.forecaster.token_len_time, local_t3.forecaster.attention) == (11, 3, "local")
    kwargs = local_t3.trainer_kwargs(600.0)
    assert (kwargs["n_context"], kwargs["forecaster_token_len_time"], kwargs["forecaster_attention"]) == (11, 3, "local")
    assert replace(local_t3, forecaster=local_pos.forecaster) == local_pos
    # Every other config packs all slots into one token and attends globally, as before the key existed.
    for path in (REPO_CONFIG, LON_LAT_CONFIG, NEIGHBOURS_CONFIG, LOCAL_CONFIG, LOCAL_POS_CONFIG):
        forecaster = load_config(path).forecaster
        assert forecaster.token_len_time == forecaster.n_context + 1 and forecaster.attention == "global"


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
        (lambda raw: raw["grid"].update(ghost_rings=2), "grid.ghost_rings must be 0 or 1"),
        (lambda raw: raw["grid"].update(ghost_rings=1), "\\['grid.ghost_rings'\\] need grid.canonical_order"),
        (lambda raw: raw["forecaster"].update(token_overlap_space=True), "forecaster.token_overlap_space.* need grid.canonical_order"),
        (lambda raw: raw["forcings"].update(orientation=True), "forcings.orientation.* need grid.canonical_order"),
        (lambda raw: raw["reconstructor"].update(attention="local"), "reconstructor.attention: local.* need grid.canonical_order"),
        (lambda raw: raw["reconstructor"].update(attention="regional"), "reconstructor.attention must be global or local"),
        (lambda raw: raw["reconstructor"].pop("attention"), "missing keys \\['attention'\\]"),
        (lambda raw: raw["forecaster"].pop("attention"), "missing keys \\['attention'\\]"),
        (lambda raw: raw["forecaster"].pop("token_len_time"), "missing keys \\['token_len_time'\\]"),
        (lambda raw: raw["forecaster"].update(token_len_time=3), "forecaster.token_len_time must divide n_context \\+ 1 = 11"),
        (lambda raw: raw["forecaster"].update(token_len_time=0), "forecaster.token_len_time must divide"),
        (lambda raw: raw["forecaster"].update(token_len_time=22), "forecaster.token_len_time must divide"),
        (lambda raw: raw["forecaster"].update(attention="regional"), "forecaster.attention must be global or local"),
        (lambda raw: raw["forecaster"].update(attention="local"), "forecaster.attention: local.* need grid.canonical_order"),
    ],
)
def test_wrong_files_stop_at_startup(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, change))


def test_exponents_without_a_dot_are_numbers(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(REPO_CONFIG.read_text().replace("lr: 1.0e-3", "lr: 1e-3"))
    assert load_config(path).reconstructor.lr == 1e-3
