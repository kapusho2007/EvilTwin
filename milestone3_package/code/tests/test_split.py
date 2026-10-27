"""Step-8 validation scheme: every fold holds both classes and negatives from every session, and no
training or validation window lies within D of a test window (overlapping windows share frames)."""
import numpy as np

from pipeline import evaluate as E


def _toy():
    rng = np.random.default_rng(0)
    tau = np.r_[np.arange(0, 600.0), np.arange(5000, 5600.0), np.arange(90000, 91000.0)]
    tau = np.sort(np.r_[tau, tau + rng.uniform(0, 1, len(tau))])
    y = ((tau > 5100) & (tau < 5500)).astype(int)
    return tau, y


def test_folds_hold_both_classes_and_all_sessions():
    tau, y = _toy()
    fid, sid = E.folds(tau, y), E.sessions(tau)
    assert len(np.unique(sid)) == 3
    for k in range(5):
        m = fid == k
        assert y[m].sum() > 0 and (y[m] == 0).sum() > 0
        assert set(np.unique(sid[m & (y == 0)])) == {0, 1, 2}


def test_purge_gap():
    tau, y = _toy()
    fid, D = E.folds(tau, y), 5.0
    for k in range(5):
        tr, va, te = E.split(tau, y, fid, k, D)
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
        for other in (tr, va):
            d = np.abs(tau[other][:, None] - tau[te][None, :]).min()
            assert d >= D
        assert np.abs(tau[tr][:, None] - tau[va][None, :]).min() >= D
