"""Unit tests for messe.common.training: normalization statistics and log averaging."""

import pytest
import torch

from messe.common_utils.training import RunningMeanStd, mean_stats


# ----------------------------------------------------------------------------
# RunningMeanStd
# ----------------------------------------------------------------------------


def test_running_mean_std_matches_torch_moments():
    torch.manual_seed(0)
    n_samples, shape = 50, (6, 3)
    samples = torch.randn(n_samples, *shape) * 2.0 + 5.0

    acc = RunningMeanStd(shape=shape, n_samples=n_samples, device=torch.device("cpu"))
    for i in range(n_samples):
        done = acc.update(samples[i])
        assert done == (i == n_samples - 1)

    assert acc.done
    expected_mean = samples.mean(dim=0)
    expected_std = samples.std(dim=0, unbiased=False)
    assert torch.allclose(acc.stats.mean, expected_mean, atol=1e-4)
    assert torch.allclose(acc.stats.std, expected_std, atol=1e-4)


def test_running_mean_std_freezes_after_n_samples():
    acc = RunningMeanStd(shape=(2,), n_samples=3, device=torch.device("cpu"))
    acc.update(torch.tensor([1.0, 1.0]))
    acc.update(torch.tensor([2.0, 2.0]))
    assert not acc.done
    acc.update(torch.tensor([3.0, 3.0]))
    assert acc.done
    frozen_mean = acc.stats.mean.clone()

    # Further updates must be no-ops once frozen.
    assert acc.update(torch.tensor([1000.0, -1000.0])) is True
    assert torch.equal(acc.stats.mean, frozen_mean)
    assert acc.count == 3


def test_running_mean_std_normalize_denormalize_roundtrip():
    acc = RunningMeanStd(shape=(4,), n_samples=2, device=torch.device("cpu"))
    acc.update(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    acc.update(torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert acc.done

    x = torch.tensor([2.5, 3.5, 4.5, 5.5])
    normalized = acc.normalize(x)
    x_back = acc.denormalize(normalized)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_running_mean_std_raises_before_finalized():
    acc = RunningMeanStd(shape=(2,), n_samples=5, device=torch.device("cpu"))
    acc.update(torch.zeros(2))
    with pytest.raises(AssertionError):
        acc.normalize(torch.zeros(2))


# ----------------------------------------------------------------------------
# mean_stats
# ----------------------------------------------------------------------------


def test_mean_stats_averages_numbers_and_per_level_dicts():
    rows = [{"recon": {4: 1.0, 8: 3.0}, "fc_rmse": 2.0}, {"recon": {4: 3.0, 8: 5.0}, "fc_rmse": 4.0}]
    assert mean_stats(rows) == {"recon": {4: 2.0, 8: 4.0}, "fc_rmse": 3.0}
