"""
Stage 4: Model Training. Dataset-agnostic: receives arrays only.

Each model exposes fit(Xtr, ytr, Xva, yva) and score(X) (higher = more attack-like), and lists its
hyperparameter grid in GRID. Supervised models get class weights from y_train (cost-sensitive
learning, Step-8 imbalance rule); one-class models are fitted on benign training windows only.

  XGB     gradient-boosted trees on View A, NaN routed by a learned default branch.
  RF      random forest, reference baseline (Milestone 2 model), fixed hyperparameters.
  OCSVM   one-class SVM (RBF) on the normalized View A, benign only.
  AE      dense autoencoder on the normalized View A, benign only; score = reconstruction MSE.
  LSTM    2-layer LSTM on View B (last 64 frames of the AP inside the window).
  Rule    best single feature, direction and threshold chosen on validation (the bar to beat).
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


def pos_weight(y) -> float:
    """w+ = n- / n+ from the training labels, so both classes carry equal total loss."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def _pr(y, s):
    return average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan


class XGB:
    GRID = [dict(max_depth=d, learning_rate=lr) for d in (3, 5) for lr in (0.05, 0.1)]

    def __init__(self, max_depth=3, learning_rate=0.1, weighting="cost", seed=0):
        self.p, self.weighting, self.seed = dict(max_depth=max_depth, learning_rate=learning_rate), weighting, seed

    def fit(self, Xtr, ytr, Xva, yva):
        self.m = XGBClassifier(n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                               scale_pos_weight=pos_weight(ytr) if self.weighting == "cost" else 1.0,
                               eval_metric="aucpr", early_stopping_rounds=30, tree_method="hist",
                               random_state=self.seed, n_jobs=4, **self.p)
        self.m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class RF:
    GRID = [{}]

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                        n_jobs=4, random_state=self.seed).fit(Xtr, ytr)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class OCSVM:
    GRID = [dict(nu=nu, gamma=g) for nu in (0.01, 0.05, 0.1) for g in ("scale", 0.1, 0.01)]
    MAX_TRAIN = 5000                                  # O(n^2) kernel: benign subsample

    def __init__(self, nu=0.05, gamma="scale", seed=0):
        self.p, self.seed = dict(nu=nu, gamma=gamma), seed

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        b = np.where(np.asarray(ytr) == 0)[0]
        if len(b) > self.MAX_TRAIN:
            b = np.random.default_rng(self.seed).choice(b, self.MAX_TRAIN, replace=False)
        self.m = OneClassSVM(kernel="rbf", **self.p).fit(Xtr[b])
        return self

    def score(self, X):
        return -self.m.decision_function(X)


def _seed_torch(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class _AENet(torch.nn.Module):
    def __init__(self, d, b):
        super().__init__()
        L, R = torch.nn.Linear, torch.nn.ReLU
        self.enc = torch.nn.Sequential(L(d, 32), R(), L(32, 16), R(), L(16, b))
        self.dec = torch.nn.Sequential(L(b, 16), R(), L(16, 32), R(), L(32, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class AE:
    """d -> 32 -> 16 -> b -> 16 -> 32 -> d, MSE, Adam 1e-3, batch 128, early stopping on the
    benign validation loss (patience 10, max 200 epochs)."""
    GRID = [dict(bottleneck=b) for b in (4, 8)]

    def __init__(self, bottleneck=4, seed=0, max_epochs=200, patience=10):
        self.b, self.seed, self.max_epochs, self.patience = bottleneck, seed, max_epochs, patience

    def fit(self, Xtr, ytr, Xva, yva):
        _seed_torch(self.seed)
        xt = torch.tensor(Xtr[np.asarray(ytr) == 0], dtype=torch.float32)
        xv = torch.tensor(Xva[np.asarray(yva) == 0], dtype=torch.float32)
        self.net = _AENet(xt.shape[1], self.b)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(self.seed)
        best, state, wait = np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            for i in torch.randperm(len(xt), generator=g).split(128):
                opt.zero_grad()
                loss = ((self.net(xt[i]) - xt[i]) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(xv) - xv) ** 2).mean().item()
            if v < best - 1e-6:
                best, state, wait = v, {k: t.clone() for k, t in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epochs = ep + 1
        return self

    def score(self, X):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return ((self.net(x) - x) ** 2).mean(1).numpy()


class SeqScaler:
    """Per-channel (x - median) / IQR fitted on the real frames of the training sequences only;
    NaN -> 0 (= the training median) after scaling. Binary channels and 'present' pass unchanged."""

    def fit(self, S):
        real = S[..., -1] == 1
        v = S[real]
        self.med = np.nanmedian(v, 0)
        iqr = np.nanpercentile(v, 75, 0) - np.nanpercentile(v, 25, 0)
        self.iqr = np.where(iqr > 0, iqr, 1.0)
        binary = np.all(np.isin(v[~np.isnan(v).any(1)], [0, 1]), 0)
        self.med[binary], self.iqr[binary] = 0.0, 1.0
        return self

    def transform(self, S):
        out = (S - self.med) / self.iqr
        out[S[..., -1] == 0] = 0.0
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


class _LSTMNet(torch.nn.Module):
    def __init__(self, d, h1, h2, dropout):
        super().__init__()
        self.l1 = torch.nn.LSTM(d, h1, batch_first=True)
        self.l2 = torch.nn.LSTM(h1, h2, batch_first=True) if h2 else None     # h2 = 0: one layer
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(h2 or h1, 1)

    def forward(self, x):
        h, _ = self.l1(x)
        if self.l2 is not None:
            h, _ = self.l2(self.drop(h))
        return self.out(self.drop(h[:, -1])).squeeze(1)


class LSTM:
    """LSTM(h1) -> LSTM(h2) -> linear on the last step; dropout 0.3, Adam 1e-3, batch 256,
    BCE with pos_weight = n-/n+ (cost-sensitive), early stopping on validation PR-AUC
    (patience 5, max 40 epochs)."""
    GRID = [dict(h1=64, h2=32), dict(h1=32, h2=16)]

    def __init__(self, h1=64, h2=32, dropout=0.3, lr=1e-3, weighting="cost", seed=0, max_epochs=40, patience=5):
        self.h1, self.h2, self.dropout, self.lr, self.weighting = h1, h2, dropout, lr, weighting
        self.seed, self.max_epochs, self.patience = seed, max_epochs, patience

    def fit(self, Str, ytr, Sva, yva):
        _seed_torch(self.seed)
        xt, yt = torch.tensor(Str), torch.tensor(np.asarray(ytr), dtype=torch.float32)
        self.net = _LSTMNet(xt.shape[2], self.h1, self.h2, self.dropout)
        pw = pos_weight(ytr) if self.weighting == "cost" else 1.0
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        best, state, wait = -np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            for i in torch.randperm(len(xt), generator=g).split(256):
                opt.zero_grad()
                lossf(self.net(xt[i]), yt[i]).backward()
                opt.step()
            v = _pr(yva, self.score(Sva))
            if v > best + 1e-4:
                best, state, wait = v, {k: t.clone() for k, t in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epochs = ep + 1
        return self

    def score(self, S):
        self.net.eval()
        with torch.no_grad():
            return torch.cat([torch.sigmoid(self.net(torch.tensor(b))) for b in np.array_split(S, max(1, len(S) // 4096))]).numpy()


class _HNet(torch.nn.Module):
    """LSTM over View B, its last hidden state concatenated with the View A vector before the head."""
    def __init__(self, d, da, h1, h2, dropout):
        super().__init__()
        self.seq = _LSTMNet(d, h1, h2, dropout)
        self.out = torch.nn.Linear((h2 or h1) + da, 1)

    def forward(self, x):
        s, a = x
        h, _ = self.seq.l1(s)
        if self.seq.l2 is not None:
            h, _ = self.seq.l2(self.seq.drop(h))
        return self.out(torch.cat([self.seq.drop(h[:, -1]), a], 1)).squeeze(1)


class HLSTM(LSTM):
    """LSTM + View A (2.2, fix 3): input is the pair (sequences, normalized View A)."""

    def fit(self, Xtr, ytr, Xva, yva):
        _seed_torch(self.seed)
        st, at = (torch.tensor(x) for x in Xtr)
        yt = torch.tensor(np.asarray(ytr), dtype=torch.float32)
        self.net = _HNet(st.shape[2], at.shape[1], self.h1, self.h2, self.dropout)
        pw = pos_weight(ytr) if self.weighting == "cost" else 1.0
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        best, state, wait = -np.inf, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            for i in torch.randperm(len(st), generator=g).split(256):
                opt.zero_grad()
                lossf(self.net((st[i], at[i])), yt[i]).backward()
                opt.step()
            v = _pr(yva, self.score(Xva))
            if v > best + 1e-4:
                best, state, wait = v, {k: t.clone() for k, t in self.net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epochs = ep + 1
        return self

    def score(self, X):
        s, a = X
        self.net.eval()
        idx = np.array_split(np.arange(len(s)), max(1, len(s) // 4096))
        with torch.no_grad():
            return torch.cat([torch.sigmoid(self.net((torch.tensor(s[i]), torch.tensor(a[i])))) for i in idx]).numpy()


class Rule:
    """Best single feature: x_j >= t (or <= t). j and direction by ROC-AUC on train + validation
    (more data, so the choice is stable across folds), threshold on validation. feature= fixes j."""
    GRID = [{}]

    def __init__(self, names=None, feature=None, seed=0):
        self.names, self.fixed = names, feature

    def fit(self, Xtr, ytr, Xva, yva):
        med = np.nanmedian(Xtr, 0)
        X = np.vstack([Xtr, Xva])
        X = np.where(np.isnan(X), med, X)
        y = np.r_[ytr, yva]
        if self.fixed is not None:
            self.j = self.names.index(self.fixed)
            auc_j = roc_auc_score(y, X[:, self.j])
        else:
            auc = np.array([roc_auc_score(y, X[:, j]) if np.std(X[:, j]) > 0 else 0.5 for j in range(X.shape[1])])
            self.j = int(np.argmax(np.abs(auc - 0.5)))
            auc_j = auc[self.j]
        self.sign, self.med = (1.0 if auc_j >= 0.5 else -1.0), med
        self.feature = self.names[self.j] if self.names else self.j
        return self

    def score(self, X):
        x = X[:, self.j]
        return self.sign * np.where(np.isnan(x), self.med[self.j], x)
