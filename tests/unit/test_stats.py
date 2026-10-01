import numpy as np
import pytest

from beyond_answer_confidence.metrics.ranking import auroc
from beyond_answer_confidence.stats import inference, resampling

SEEDS = [0, 1, 7, 12345]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", [1, 5, 50])
def test_resample_indices_shapes_and_determinism(seed: int, n: int) -> None:
    a = resampling.resample_indices(n, 20, np.random.default_rng(seed))
    b = resampling.resample_indices(n, 20, np.random.default_rng(seed))
    assert len(a) == 20
    assert all(len(i) == n and i.min() >= 0 and i.max() < n for i in a)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("seed", SEEDS)
def test_cluster_resamples_keep_clusters_whole(seed: int) -> None:
    clusters = np.array(["a", "a", "b", "c", "c", "c"])
    for idx in resampling.resample_indices(
        6, 30, np.random.default_rng(seed), clusters
    ):
        picked = clusters[idx]
        for c in set(picked):
            assert (picked == c).sum() % (clusters == c).sum() == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_bootstrap_matches_index_loop(seed: int) -> None:
    v = np.random.default_rng(99).normal(size=30)
    boot = resampling.bootstrap(
        len(v), lambda i: float(v[i].mean()), 50, np.random.default_rng(seed)
    )
    rng = np.random.default_rng(seed)
    manual = [v[rng.integers(0, 30, 30)].mean() for _ in range(50)]
    assert np.array_equal(boot, manual)


def test_bootstrap_means_vectorised_draw() -> None:
    v = np.arange(10.0)
    boot = resampling.bootstrap_means(v, 100, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    assert np.array_equal(boot, v[rng.integers(0, 10, (100, 10))].mean(axis=1))
    assert boot.shape == (100,)


def test_constant_data_gives_degenerate_intervals() -> None:
    v = np.full(20, 3.0)
    assert resampling.percentile_ci(
        resampling.bootstrap_means(v, 200, np.random.default_rng(0))
    ) == (3.0, 3.0)
    boot = resampling.cluster_bootstrap(
        v, np.arange(20) % 4, 50, np.random.default_rng(0)
    )
    assert (boot == 3.0).all()


def test_percentile_ci_ignores_nan() -> None:
    lo, hi = resampling.percentile_ci(np.array([np.nan, *range(101)]), 0.9)
    assert (lo, hi) == pytest.approx((5.0, 95.0))


def test_auroc_interval_contains_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(size=200) < 0.5
    s = y + rng.normal(size=200)
    res = resampling.auroc_interval(s, y, 200, np.random.default_rng(1))
    assert res["auroc"] == auroc(s, y)
    assert res["ci"][0] < res["auroc"] < res["ci"][1]
    assert res["n_positive"] == int(y.sum())


def test_paired_difference_of_identical_scores_is_zero() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(size=50) < 0.5
    s = rng.normal(size=50)
    res = resampling.paired_auroc_difference(s, s, y, 50, np.random.default_rng(0))
    assert res["diff"] == 0.0
    assert res["ci95"] == [0.0, 0.0]
    assert res["ci90"] == [0.0, 0.0]


def test_month_block_indices_keep_period_sizes() -> None:
    month = np.array(["2024-01", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05"])
    period = np.array([False, False, False, True, True, True])
    idx = resampling.month_block_indices(
        month, period, np.random.default_rng(0), block=2
    )
    months_pre = set(month[idx][~period[idx]])
    assert months_pre <= {"2024-01", "2024-02"}
    assert len(set(month[idx][period[idx]])) >= 1


def test_bca_bound_brackets_mean() -> None:
    v = np.random.default_rng(0).normal(1.0, 1.0, 200)
    upper = resampling.bca_bound(v, np.mean, 0.05, "upper", 500, seed=0)
    lower = resampling.bca_bound(v, np.mean, 0.05, "lower", 500, seed=0)
    assert lower < v.mean() < upper


def test_smece_interval_caps_resamples_and_reports_tail() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1, 60)
    y = rng.uniform(size=60) < p
    res = resampling.smece_interval(
        p, y, 30, np.random.default_rng(0), max_resamples_above=(50, 10), threshold=0.5
    )
    assert res["resamples"] == 10
    assert 0.0 <= res["tail"] <= 1.0
    assert res["ci"][0] <= res["ci"][1]
    sub = resampling.smece_resamples(p, y, 5, np.random.default_rng(0), subsample=True)
    assert sub.shape == (5,)


def test_holm_known_answer() -> None:
    assert inference.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adj = inference.holm_adjust({"a": {"p": 0.01}, "b": {"p": 0.04}})
    assert adj["a"]["p_holm"] == pytest.approx(0.02)
    assert adj["b"]["p_holm"] == pytest.approx(0.04)


def test_mean_interval_and_tail() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0])
    res = inference.mean_interval(v, 500, np.random.default_rng(0), threshold=10.0)
    assert res["mean"] == 2.5
    assert res["tail"] == 0.0
    assert 1.0 <= res["ci"][0] <= res["ci"][1] <= 4.0


def test_bootstrap_equivalence() -> None:
    res = inference.bootstrap_equivalence(
        np.zeros(10), 0.1, 100, np.random.default_rng(0)
    )
    assert res["p"] == 0.0
    assert res["ci90"] == [0.0, 0.0]


def test_auroc_vs_threshold() -> None:
    y = np.array([True] * 10 + [False] * 10)
    s = np.arange(20.0)[::-1]
    res = inference.auroc_vs_threshold(s, y, 0.5, 100, np.random.default_rng(0))
    assert res["auroc"] == 1.0
    assert res["p"] == 0.0


def test_t_tests_and_tost() -> None:
    v = np.array([0.1, -0.1, 0.05, -0.05, 0.0])
    assert inference.t_below(v, 1.0)["p"] < 0.001
    assert inference.t_above(v, -1.0)["p"] < 0.001
    assert inference.tost_t(v, 0.5)["p"] < 0.001
    assert inference.tost_t(v, 0.01)["p"] > 0.05


def test_spearman_and_fisher() -> None:
    x = np.arange(50.0)
    assert inference.spearman(x, x) == pytest.approx(1.0)
    noisy = x + np.random.default_rng(0).normal(0, 5, 50)
    res = inference.spearman_above_fisher(x, noisy, 0.3)
    assert res["p"] < 0.01


def test_sign_flip() -> None:
    d = np.full(30, -1.0)
    res = inference.sign_flip_below(d, 999, np.random.default_rng(0))
    assert res["mean"] == -1.0
    assert res["p"] == pytest.approx(1 / 1000, abs=0.01)
    assert inference.tail_share(np.array([0.0, 1.0, 2.0, 3.0]), 2.0) == 0.5
