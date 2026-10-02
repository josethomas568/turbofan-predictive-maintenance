"""Build report/index.html and results/fleet_alert_snapshot.csv from the pipeline outputs.

Run after pipeline.py:  python build_report.py
"""
import json
import numpy as np
import pandas as pd

R = "results"
TIERS = ("WATCH", "PLAN", "URGENT")

res = json.load(open(f"{R}/results.json"))
sweep = pd.read_csv(f"{R}/plan_threshold_sweep.csv")
cv = pd.read_csv(f"{R}/cv_cycle_predictions.csv.gz")
test = pd.read_csv(f"{R}/test_cycle_predictions.csv.gz")
ev = pd.read_csv(f"{R}/alert_outcomes.csv")

data = {"results": res}
data["sweep"] = {fd: g[["T", "caught", "late", "missed", "false", "lead_median"]].to_dict("list")
                 for fd, g in sweep.groupby("dataset")}

# example engines for the trajectory chart: the cross-validated engine with median life
examples = {}
for fd in ("FD001", "FD003"):
    x = cv[cv.dataset == fd]
    life = x.groupby("unit").cycle.max()
    u = int(life.sub(life.median()).abs().idxmin())
    g = x[x.unit == u]
    examples[fd] = {
        "unit": u, "cycle": g.cycle.tolist(), "true": np.minimum(g.true_rul, 125).tolist(),
        "pred": g.pred_rul.round(1).tolist(), "anom": g.anom_ewma.round(3).tolist(),
        "thr": float(g.anom_thr.iloc[0]), "fail": int(g.cycle.max()),
        "first": {t: (int(g[g[t]].cycle.iloc[0]) if g[t].any() else None) for t in TIERS},
    }
data["examples"] = examples

data["leads"] = {t: ev[(ev.split == "test") & (ev.tier == t) & (ev.outcome == "caught")]
                 .first_alert_lead.astype(int).tolist() for t in TIERS}
totals = {}
for s in ("cv_train", "test"):
    for t in TIERS:
        e = ev[(ev.split == s) & (ev.tier == t)]
        totals[f"{s}_{t}"] = {**e.outcome.value_counts().to_dict(),
                              "lead_med": float(e[e.outcome == "caught"].first_alert_lead.median())}
data["totals"] = totals

html = open("report/template.html").read().replace("__DATA__", json.dumps(data, default=float))
open("report/index.html", "w").write(html)

# fleet snapshot: alert status of each test engine at its last observed cycle
last = test.groupby(["dataset", "unit"]).tail(1).copy()
last["status"] = np.select([last.URGENT, last.PLAN, last.WATCH], ["URGENT", "PLAN", "WATCH"], "NORMAL")
last = last.rename(columns={"cycle": "last_cycle", "pred_rul": "predicted_rul", "anom_ewma": "anomaly_score",
                            "anom_thr": "anomaly_threshold", "true_rul": "actual_rul_for_eval"})
last[["dataset", "unit", "last_cycle", "status", "predicted_rul", "anomaly_score", "anomaly_threshold",
      "actual_rul_for_eval"]].round(2).to_csv(f"{R}/fleet_alert_snapshot.csv", index=False)
print("wrote report/index.html and results/fleet_alert_snapshot.csv")
