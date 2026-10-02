"""
C-MAPSS turbofan predictive-maintenance pipeline.

1. Anomaly detection  - learn what a *healthy* engine looks like from early-life cycles,
                        score every cycle by its distance from that baseline (Mahalanobis),
                        smooth it causally (EWMA), and flag a sustained departure.
2. Failure prediction - gradient-boosted regression of remaining useful life (RUL) from
                        causal (past-only) sensor features plus the anomaly score.
3. Alerts             - three tiers driven by thresholds:
                          WATCH   : smoothed anomaly score above healthy threshold for 5 cycles
                          PLAN    : predicted RUL <= T_plan for 3 consecutive cycles
                          URGENT  : predicted RUL <= T_urgent for 3 consecutive cycles
4. Evaluation         - each alert is checked against the engine's real failure cycle:
                        caught / late / missed / false (premature) + warning lead time.

Model selection and thresholds are tuned only on the training engines using 5-fold
cross-validation grouped by engine (out-of-fold predictions). The NASA test set is used
once, at the end, as an untouched hold-out.
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = "data"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125          # standard piecewise-linear target: engines treated as "healthy" above this
HEALTHY_CYCLES = 30    # first N cycles of each training engine define "healthy" for the fleet
BASELINE = 20          # first N cycles of EACH engine define that engine's own baseline
EWMA_ALPHA = 0.1
SEED = 7

# Alert-evaluation policy (fixed business definitions, not tuned)
POLICY = {
    # tier: (window W = alert is useful only if it fires with <= W cycles of life left,
    #        min lead L = alert must leave at least L cycles to act)
    "WATCH":  {"W": 150, "L": 30},
    "PLAN":   {"W": 60,  "L": 10},
    "URGENT": {"W": 30,  "L": 3},
}
PERSIST = {"WATCH": 5, "PLAN": 3, "URGENT": 3}
MISS_COST = 10  # a missed/late failure costs 10x an unnecessary (premature) maintenance action


def load(fd):
    tr = pd.read_csv(f"{DATA}/train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(f"{DATA}/test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul = np.loadtxt(f"{DATA}/RUL_{fd}.txt")
    tr["fail_cycle"] = tr.groupby("unit").cycle.transform("max")
    last = te.groupby("unit").cycle.max()
    te["fail_cycle"] = te.unit.map(last + pd.Series(rul, index=last.index))
    for d in (tr, te):
        d["true_rul"] = d.fail_cycle - d.cycle
    return tr, te


class Preprocessor:
    """Fit on training engines only: operating-condition clusters, healthy baseline per
    condition, sensor selection, and the healthy-state covariance for the anomaly score."""

    def fit(self, df):
        # FD002/FD004 fly six distinct regimes (altitude/Mach/throttle); FD001/FD003 fly one
        self.n_cond = 6 if (df.op1.max() - df.op1.min()) > 1 else 1
        if self.n_cond > 1:
            self.km = KMeans(self.n_cond, n_init=10, random_state=SEED).fit(df[["op1", "op2", "op3"]])
        cond = self._cond(df)
        healthy = df.cycle <= HEALTHY_CYCLES
        self.mu, self.sd = {}, {}
        for c in range(self.n_cond):
            h = df.loc[healthy & (cond == c), SENSORS]
            self.mu[c] = h.mean()
            self.sd[c] = h.std().replace(0, np.nan)
        z = self._z(df, cond)
        # keep sensors that actually move as engines approach failure. Use mean |z| rather than
        # |mean z|: in FD003/FD004 the two fault modes push some sensors in opposite directions,
        # which cancels in a plain average. Pure noise gives mean |z| ~ 0.8, so require 1.5.
        near_fail = df.true_rul <= 20
        shift = z.loc[near_fail].abs().mean()
        self.sensors = [s for s in SENSORS if np.isfinite(shift.get(s, np.nan)) and shift[s] >= 1.5]
        # anomaly model: covariance of each engine's deviation from its OWN early-life baseline,
        # measured on still-healthy cycles just after the baseline window
        dev = self._dev(df, z[self.sensors].fillna(0.0))
        ref = (df.cycle > BASELINE) & (df.cycle <= BASELINE + 40)
        self.cov = LedoitWolf().fit(dev.loc[ref].values)
        return self

    @staticmethod
    def _dev(df, z):
        """z minus the engine's own baseline = mean of its first BASELINE cycles. Causal: at
        cycle t (< BASELINE) the baseline only uses cycles 1..t."""
        n = df.groupby("unit").cumcount() + 1
        early = z.mul((n <= BASELINE).astype(float).values, axis=0)
        csum = early.groupby(df.unit).cumsum()
        base = csum.div(np.minimum(n, BASELINE).values, axis=0)
        return z - base

    def _cond(self, df):
        if self.n_cond == 1:
            return pd.Series(0, index=df.index)
        return pd.Series(self.km.predict(df[["op1", "op2", "op3"]]), index=df.index)

    def _z(self, df, cond):
        z = pd.DataFrame(index=df.index, columns=SENSORS, dtype=float)
        for c in range(self.n_cond):
            m = cond == c
            z.loc[m] = ((df.loc[m, SENSORS] - self.mu[c]) / self.sd[c]).values
        return z

    def transform(self, df):
        cond = self._cond(df)
        z = self._z(df, cond)[self.sensors].fillna(0.0)
        out = df[["unit", "cycle", "true_rul", "fail_cycle"]].copy()
        out[self.sensors] = z.values
        d = self._dev(df, z).values - self.cov.location_
        out["anom"] = np.einsum("ij,jk,ik->i", d, self.cov.precision_, d) / len(self.sensors)
        out["anom_ewma"] = out.groupby("unit").anom.transform(lambda s: s.ewm(alpha=EWMA_ALPHA).mean())
        out["in_warmup"] = (df.groupby("unit").cumcount() + 1).values <= BASELINE
        return out


def features(df, sensors):
    """Causal features only: everything at cycle t uses cycles <= t."""
    g = df.groupby("unit")
    f = df[["unit", "cycle", "true_rul", "fail_cycle", "anom", "anom_ewma", "in_warmup"]].copy()
    cols = sensors + ["anom"]
    for w in (10, 30):
        r = g[cols].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        f[[f"{c}_m{w}" for c in cols]] = r.values
    # rolling least-squares slope over the last 20 cycles
    w = 20
    t = df.cycle.astype(float)
    tm = g.cycle.rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
    tt = (t * t).groupby(df.unit).rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
    var_t = (tt - tm ** 2).replace(0, np.nan)
    for c in cols:
        x = df[c]
        xm = g[c].rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
        xt = (x * t).groupby(df.unit).rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
        f[f"{c}_slope"] = ((xt - xm * tm) / var_t).fillna(0).values
    f["anom_ewma_slope"] = f.groupby("unit").anom_ewma.diff(10).fillna(0) / 10
    return f


def make_model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=50, l2_regularization=1.0,
                                         random_state=SEED)


def fit_predict(train_raw, apply_raw):
    """Fit preprocessing + anomaly baseline + RUL model on train_raw, score apply_raw."""
    pp = Preprocessor().fit(train_raw)
    tr = pp.transform(train_raw)
    ap = pp.transform(apply_raw)
    # anomaly threshold: 99.5th pct of smoothed score on early-life (post-warm-up) training cycles
    early = (tr.cycle > BASELINE) & (tr.cycle <= 60)
    thr = float(np.quantile(tr.loc[early, "anom_ewma"], 0.995))
    ftr, fap = features(tr, pp.sensors), features(ap, pp.sensors)
    X = [c for c in ftr.columns if c not in ("unit", "true_rul", "fail_cycle", "in_warmup")]
    m = make_model().fit(ftr[X], np.minimum(ftr.true_rul, RUL_CAP))
    fap["pred_rul"] = np.clip(m.predict(fap[X]), 0, None)
    fap["anom_thr"] = thr
    return fap, pp


def oof_predictions(train_raw, k=5):
    units = train_raw.unit.unique()
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(units), k)
    parts = []
    for f in folds:
        te = train_raw[train_raw.unit.isin(f)]
        tr = train_raw[~train_raw.unit.isin(f)]
        p, _ = fit_predict(tr, te)
        parts.append(p)
    return pd.concat(parts).sort_values(["unit", "cycle"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- alerts
def persistent(flag, unit, k):
    """True at cycle t when flag has been true for the last k consecutive cycles of that unit."""
    run = flag.astype(int).groupby((flag != flag.shift()).cumsum().astype(str) + "_" + unit.astype(str)).cumsum()
    return run >= k


def add_alerts(df, t_plan, t_urgent):
    df = df.copy()
    df["WATCH"] = persistent((df.anom_ewma > df.anom_thr) & ~df.in_warmup, df.unit, PERSIST["WATCH"])
    df["PLAN"] = persistent(df.pred_rul <= t_plan, df.unit, PERSIST["PLAN"])
    df["URGENT"] = persistent(df.pred_rul <= t_urgent, df.unit, PERSIST["URGENT"])
    return df


def evaluate_tier(df, tier):
    """Latched first alert per engine, judged against that engine's real failure cycle.
    caught   : first alert leaves L..W cycles of life (useful, actionable)
    false    : first alert fires with > W cycles left (premature -> unnecessary maintenance)
    late     : alert fires but leaves < L cycles (too little time to act)
    missed   : no alert before the failure
    For truncated test trajectories also:
    quiet_ok : unit never got near failure in the observed data and no alert fired (true negative)
    pending  : unit is inside its window, no alert yet, but data ends before the window closes
    """
    W, L = POLICY[tier]["W"], POLICY[tier]["L"]
    rows = []
    for u, g in df.groupby("unit"):
        end_rul = g.true_rul.iloc[-1]
        hit = g[g[tier]]
        if len(hit):
            lead = int(hit.true_rul.iloc[0])
            outcome = "false" if lead > W else ("caught" if lead >= L else "late")
        else:
            lead = None
            if end_rul < L:
                outcome = "missed"
            elif end_rul > W:
                outcome = "quiet_ok"
            else:
                outcome = "pending"
        rows.append({"unit": int(u), "tier": tier, "first_alert_lead": lead,
                     "end_true_rul": int(end_rul), "outcome": outcome})
    return pd.DataFrame(rows)


def summarize(ev):
    c = ev.outcome.value_counts().to_dict()
    n = {k: int(c.get(k, 0)) for k in ("caught", "late", "missed", "false", "quiet_ok", "pending")}
    reached = n["caught"] + n["late"] + n["missed"]          # failures (inside window) we can judge
    alerts = n["caught"] + n["late"] + n["false"]
    leads = ev.loc[ev.outcome == "caught", "first_alert_lead"].astype(float)
    return {**n,
            "failures_judged": reached,
            "detection_rate": n["caught"] / reached if reached else None,
            "alerts_fired": alerts,
            "false_alert_share": n["false"] / alerts if alerts else None,
            "lead_median": float(leads.median()) if len(leads) else None,
            "lead_p10": float(leads.quantile(0.1)) if len(leads) else None,
            "lead_p90": float(leads.quantile(0.9)) if len(leads) else None}


def sweep_plan(df, grid=range(10, 101, 2)):
    out = []
    for t in grid:
        flag = persistent(df.pred_rul <= t, df.unit, PERSIST["PLAN"])
        ev = evaluate_tier(df.assign(PLAN=flag), "PLAN")
        s = summarize(ev)
        s["T"] = t
        s["cost"] = MISS_COST * (s["missed"] + s["late"]) + s["false"]
        out.append(s)
    return pd.DataFrame(out)


def rul_metrics(pred, true):
    e = pred - true
    score = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1).sum()  # NASA asymmetric score
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "nasa_score": float(score)}


def run(fd):
    tr_raw, te_raw = load(fd)
    oof = oof_predictions(tr_raw)
    sw = sweep_plan(oof)
    # several thresholds often tie on cost; take the middle of the best plateau so the
    # chosen threshold isn't sitting on the edge where it tips into late or premature alerts
    best = sw[sw.cost <= sw.cost.min()]["T"].sort_values().values
    t_plan = int(best[len(best) // 2])
    t_urgent = 20
    oof = add_alerts(oof, t_plan, t_urgent)

    test, pp = fit_predict(tr_raw, te_raw)
    test = add_alerts(test, t_plan, t_urgent)

    res = {"dataset": fd, "sensors_used": pp.sensors, "conditions": pp.n_cond,
           "T_plan": t_plan, "T_urgent": t_urgent, "anom_threshold_test": float(test.anom_thr.iloc[0])}
    evals = []
    for name, d in (("cv_train", oof), ("test", test)):
        res[name] = {}
        for tier in POLICY:
            ev = evaluate_tier(d, tier).assign(split=name, dataset=fd)
            evals.append(ev)
            res[name][tier] = summarize(ev)
        last = d.groupby("unit").tail(1)
        res[name]["rul_last_cycle"] = rul_metrics(last.pred_rul.values, np.minimum(last.true_rul.values, RUL_CAP))
        res[name]["rul_all_cycles"] = rul_metrics(d.pred_rul.values, np.minimum(d.true_rul.values, RUL_CAP))
        # anomaly detector: per-cycle false-alarm rate on clearly healthy cycles (>150 left)
        h = d.true_rul > POLICY["WATCH"]["W"]
        res[name]["watch_cycle_false_rate"] = float(d.loc[h, "WATCH"].mean()) if h.any() else None
    sw["dataset"] = fd
    oof["dataset"], test["dataset"] = fd, fd
    keep = ["dataset", "unit", "cycle", "true_rul", "anom_ewma", "anom_thr", "pred_rul", "WATCH", "PLAN", "URGENT"]
    return res, pd.concat(evals), sw, oof[keep], test[keep]


if __name__ == "__main__":
    fds = sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]
    results, evs, sws, oofs, tests = [], [], [], [], []
    for fd in fds:
        r, e, s, o, t = run(fd)
        results.append(r); evs.append(e); sws.append(s); oofs.append(o); tests.append(t)
        print(fd, json.dumps({k: r[k] for k in ("T_plan", "conditions")}),
              "| test PLAN:", {k: r["test"]["PLAN"][k] for k in ("caught", "late", "missed", "false", "quiet_ok", "pending", "lead_median")},
              "| CV PLAN:", {k: r["cv_train"]["PLAN"][k] for k in ("caught", "late", "missed", "false", "lead_median")},
              "| test RMSE last:", round(r["test"]["rul_last_cycle"]["rmse"], 2), flush=True)
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/results.json", "w"), indent=1, default=float)
    pd.concat(evs).to_csv("results/alert_outcomes.csv", index=False)
    pd.concat(sws).to_csv("results/plan_threshold_sweep.csv", index=False)
    pd.concat(oofs).to_csv("results/cv_cycle_predictions.csv.gz", index=False)
    pd.concat(tests).to_csv("results/test_cycle_predictions.csv.gz", index=False)
