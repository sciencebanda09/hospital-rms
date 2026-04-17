"""
hospital_dashboard.py
=====================
Doctor-facing web dashboard for the Intelligent Hospital Resource Management System.

HOW TO RUN:
    pip install flask
    python hospital_dashboard.py [--scenario surge|pandemic|routine|critical]

Then open: http://localhost:5050

The ACPL simulation runs in the background. The dashboard polls live state
every 2 seconds — no page reload required. Keep hospital_rms.py in the
same folder.

Author: Shashank Dev (HRMS · ACPL Engine)
UI Layer: Doctor Dashboard
"""

import sys
import json
import time
import threading
import argparse
from pathlib import Path

# ── Bootstrap: make sure hospital_rms.py is importable ───────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    import hospital_rms as hrms
except ImportError as e:
    print(f"[ERROR] Cannot import hospital_rms.py — make sure it's in the same folder.\n{e}")
    sys.exit(1)

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("[ERROR] Flask not found. Run: pip install flask")
    sys.exit(1)

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# §1  SIMULATION WRAPPER — runs ticks in background thread
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationRunner:
    TICK_INTERVAL = 0.15          # seconds between ticks (≈6–7 ticks/sec)
    HISTORY_MAX   = 60            # points kept for sparklines

    def __init__(self, cfg: hrms.ExperimentConfig):
        self.cfg     = cfg
        self.sim     = hrms.Simulation(cfg)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SimLoop")
        # Sparkline history
        self._occ_hist  = []
        self._dr_hist   = []
        self._wait_hist = []
        self._pt_hist   = []
        self._thread.start()

    # ── background tick loop ─────────────────────────────────────────────────
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                self.sim.tick()
                # Update sparklines
                occ = list(self.sim.occupancy_hist)[-1] if self.sim.occupancy_hist else 0
                dr  = list(self.sim.discharge_rate_hist)[-1] if self.sim.discharge_rate_hist else 0
                wt  = list(self.sim.wait_time_hist)[-1] if self.sim.wait_time_hist else 0
                pt  = list(self.sim.patient_count_hist)[-1] if self.sim.patient_count_hist else 0
                for lst, val in [
                    (self._occ_hist, occ), (self._dr_hist, dr),
                    (self._wait_hist, wt), (self._pt_hist, pt)
                ]:
                    lst.append(round(val, 1))
                    if len(lst) > self.HISTORY_MAX:
                        lst.pop(0)
            elapsed = time.perf_counter() - t0
            sleep   = max(0, self.TICK_INTERVAL - elapsed)
            time.sleep(sleep)

    def stop(self):
        self._stop.set()
        self.sim.acpl.stop()

    # ── snapshot for the API ─────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            sim = self.sim

            # ── patients ─────────────────────────────────────────────────────
            patients = []
            for p in sim.patients:
                if not p.active:
                    continue
                cond, cond_class = _condition(p.acuity)
                cat_name = hrms._CAT_NAMES[p.category]
                res_name = p.assigned_resource.name_str() if p.assigned_resource else None
                waiting_min = round(p.ticks_waiting * sim.cfg.dt * 60, 1)
                care_hr  = round(p.ticks_in_care * sim.cfg.dt, 1)
                patients.append({
                    "uid":        p.uid,
                    "category":   cat_name,
                    "condition":  cond,
                    "condClass":  cond_class,
                    "acuity":     round(p.acuity * 100),  # 0-100 for UI
                    "waitMin":    waiting_min,
                    "careHr":     care_hr,
                    "resource":   res_name or "Awaiting Bed",
                    "assigned":   res_name is not None,
                    "dangerLevel": p.danger_level,
                    "wave":       p.wave,
                })
            # Sort: critical first
            patients.sort(key=lambda p: -p["acuity"])

            # ── resources ────────────────────────────────────────────────────
            resources = []
            for r in sim.resources:
                status_label, status_class = _res_status(r)
                util_pct = round(r.used / max(1, r.capacity) * 100)
                resources.append({
                    "name":       r.name_str(),
                    "type":       hrms._RES_NAMES[r.rtype],
                    "status":     status_label,
                    "statusClass": status_class,
                    "used":       r.used,
                    "capacity":   r.capacity,
                    "utilPct":    util_pct,
                    "patient":    r.patient.uid if r.patient and r.patient.active else None,
                })

            # ── KPIs ─────────────────────────────────────────────────────────
            active_pts = len(patients)
            critical   = sum(1 for p in patients if p["condClass"] == "critical")
            avail_slots = sum(
                r.available_slots for r in sim.resources
                if r.status != hrms.ResourceStatus.MAINTENANCE
            )
            total = sim.discharges + sim.mortalities
            dr_pct = round(sim.discharges / max(1, total) * 100, 1)
            occ_pct = round(
                sum(r.used for r in sim.resources) /
                max(1, sum(r.capacity for r in sim.resources)) * 100, 1
            )

            # ── ACPL diagnostics ─────────────────────────────────────────────
            acpl = sim.acpl.diagnostics()

            # ── recent events ─────────────────────────────────────────────────
            events = list(sim.events)[:10]

            return {
                "tick":        sim.step_n,
                "simTime":     round(sim.t, 1),
                "wave":        sim.wave,
                "patients":    patients,
                "resources":   resources,
                "kpi": {
                    "activePatients":  active_pts,
                    "criticalCount":   critical,
                    "availableSlots":  avail_slots,
                    "dischargePct":    dr_pct,
                    "discharges":      sim.discharges,
                    "mortalities":     sim.mortalities,
                    "occupancyPct":    occ_pct,
                    "avgWait":         round(float(np.mean([p["waitMin"] for p in patients] or [0])), 1),
                },
                "acpl": {
                    "updates":   acpl["acpl_updates"],
                    "pLoss":     round(acpl["acpl_p_loss"], 4),
                    "meanLam":   round(acpl["acpl_mean_lam"], 3),
                    "meanC":     round(acpl["acpl_mean_C"], 3),
                },
                "sparklines": {
                    "occupancy":     self._occ_hist[-30:],
                    "dischargeRate": self._dr_hist[-30:],
                    "waitTime":      self._wait_hist[-30:],
                    "patientCount":  self._pt_hist[-30:],
                },
                "events": events,
                "scenario": _SCENARIO_LABEL,
            }


# ── helpers ──────────────────────────────────────────────────────────────────

def _condition(acuity: float):
    if acuity >= 0.85:
        return "Critical",  "critical"
    if acuity >= 0.65:
        return "Urgent",    "urgent"
    if acuity >= 0.45:
        return "Monitor",   "monitor"
    return     "Stable",    "stable"


def _res_status(r: hrms.Resource):
    if r.status == hrms.ResourceStatus.MAINTENANCE:
        return "Offline",     "offline"
    if r.status == hrms.ResourceStatus.OVERLOADED:
        return "At Capacity", "overloaded"
    if r.status == hrms.ResourceStatus.ASSIGNED:
        return "In Use",      "inuse"
    return     "Available",   "available"


# ═══════════════════════════════════════════════════════════════════════════════
# §2  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app     = Flask(__name__)
_runner: SimulationRunner = None
_SCENARIO_LABEL = "Routine Operations"


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    return jsonify(_runner.snapshot())


@app.route("/api/restart", methods=["POST"])
def api_restart():
    global _runner, _SCENARIO_LABEL
    data     = request.get_json(silent=True) or {}
    scenario = data.get("scenario", "")
    cfg      = _build_cfg(scenario)
    old      = _runner
    _runner  = SimulationRunner(cfg)
    old.stop()
    return jsonify({"ok": True, "scenario": scenario})


def _build_cfg(scenario: str) -> hrms.ExperimentConfig:
    global _SCENARIO_LABEL
    presets = {
        "surge":    dict(num_resources=8,  max_patients=24, admission_period=50,  label="Mass-Casualty Surge"),
        "pandemic": dict(num_resources=6,  max_patients=30, admission_period=40,  label="Pandemic Overload"),
        "routine":  dict(num_resources=14, max_patients=10, admission_period=130, label="Routine Operations"),
        "critical": dict(num_resources=10, max_patients=20, admission_period=60,  label="Critical Care Pressure"),
    }
    p = presets.get(scenario, dict(num_resources=10, max_patients=16, admission_period=90, label="Standard Mode"))
    _SCENARIO_LABEL = p.pop("label")
    return hrms.ExperimentConfig(export_csv=False, **p)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DASHBOARD HTML (doctor-facing UI)
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CareOS — Hospital Resource Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Nunito:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
/* ── reset & base ─────────────────────────────────────────────────── */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:       #070d14;
  --surface:  #0d1a26;
  --card:     #111f2e;
  --border:   #1c3044;
  --border2:  #234060;
  --text:     #d8eaf5;
  --dim:      #5a7a95;
  --teal:     #00c9a7;
  --teal2:    #00f5c8;
  --blue:     #2a9df4;
  --red:      #ef233c;
  --orange:   #f4962a;
  --yellow:   #f5c518;
  --green:    #1dd195;
  --purple:   #a855f7;
  --font-head: 'Rajdhani', sans-serif;
  --font-body: 'Nunito', sans-serif;
  --radius: 10px;
  --radius-sm: 6px;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-body);font-size:14px;overflow:hidden}

/* ── layout ───────────────────────────────────────────────────────── */
#app{display:grid;grid-template-rows:58px 1fr;height:100vh;gap:0}
#main{display:grid;grid-template-columns:300px 1fr 280px;gap:12px;padding:10px 12px 12px;overflow:hidden}

/* ── topbar ───────────────────────────────────────────────────────── */
#topbar{
  display:flex;align-items:center;gap:16px;
  padding:0 16px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  position:relative;
}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{
  width:34px;height:34px;border-radius:8px;
  background:linear-gradient(135deg,#00c9a7,#2a9df4);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--font-head);font-weight:700;font-size:16px;color:#070d14;
  flex-shrink:0;
}
.logo-text{font-family:var(--font-head);font-size:20px;font-weight:700;letter-spacing:.04em;color:#fff}
.logo-sub{font-size:11px;color:var(--dim);margin-top:-2px;letter-spacing:.06em;text-transform:uppercase}
.topbar-divider{width:1px;height:30px;background:var(--border);margin:0 4px}
#scenario-badge{
  padding:4px 12px;border-radius:20px;
  font-family:var(--font-head);font-size:13px;font-weight:600;letter-spacing:.04em;
  background:#1c3044;color:var(--teal);border:1px solid var(--border2);
}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:12px}
#sim-clock{font-family:var(--font-head);font-size:15px;font-weight:600;color:var(--dim)}
#sim-clock span{color:var(--teal2)}
.scenario-selector{display:flex;gap:6px}
.sc-btn{
  padding:4px 11px;border-radius:6px;border:1px solid var(--border2);
  background:transparent;color:var(--dim);font-family:var(--font-head);font-size:12px;
  font-weight:600;letter-spacing:.04em;cursor:pointer;transition:all .2s;
}
.sc-btn:hover{background:var(--border);color:var(--text)}
.sc-btn.active{background:var(--teal);color:#070d14;border-color:var(--teal)}
#acpl-indicator{
  display:flex;align-items:center;gap:6px;
  padding:4px 10px;border-radius:6px;background:#0a1c30;
  border:1px solid #1c3d5a;font-size:11px;color:var(--dim);
}
#acpl-indicator .dot{width:7px;height:7px;border-radius:50%;background:var(--teal);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── section titles ───────────────────────────────────────────────── */
.section-title{
  font-family:var(--font-head);font-size:13px;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
  margin-bottom:8px;padding-left:2px;
}

/* ── KPI cards ────────────────────────────────────────────────────── */
#left-panel{display:flex;flex-direction:column;gap:10px;overflow-y:auto;overflow-x:hidden}
#left-panel::-webkit-scrollbar{width:4px}
#left-panel::-webkit-scrollbar-track{background:transparent}
#left-panel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

.kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kpi-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:12px;position:relative;overflow:hidden;
}
.kpi-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--teal));
}
.kpi-label{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.kpi-value{
  font-family:var(--font-head);font-size:28px;font-weight:700;line-height:1;
  margin:4px 0 2px;color:var(--accent,var(--teal));
}
.kpi-sub{font-size:10px;color:var(--dim)}
.kpi-card.danger{--accent:var(--red)}
.kpi-card.warn{--accent:var(--orange)}
.kpi-card.ok{--accent:var(--green)}
.kpi-card.info{--accent:var(--blue)}

/* ── sparkline canvas ─────────────────────────────────────────────── */
.sparkline-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.spark-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:10px;
}
.spark-label{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--dim);margin-bottom:6px}
.spark-val{font-family:var(--font-head);font-size:18px;font-weight:700;color:var(--text);margin-bottom:4px}
canvas.spark{width:100%;height:36px;display:block}

/* ── resource panel ───────────────────────────────────────────────── */
.resource-list{display:flex;flex-direction:column;gap:6px}
.res-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius-sm);
  padding:8px 10px;display:flex;align-items:center;gap:10px;
}
.res-icon{
  width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;
  font-size:13px;flex-shrink:0;background:var(--res-color,#1c3044);color:#fff;font-weight:700;
  font-family:var(--font-head);
}
.res-info{flex:1;min-width:0}
.res-name{font-size:12px;font-weight:600;color:var(--text);font-family:var(--font-head);letter-spacing:.03em}
.res-meta{font-size:10px;color:var(--dim);margin-top:1px}
.res-bar-wrap{width:70px;flex-shrink:0}
.res-bar-bg{height:5px;border-radius:3px;background:var(--border);overflow:hidden}
.res-bar-fill{height:100%;border-radius:3px;background:var(--teal);transition:width .4s ease}
.res-bar-fill.warn{background:var(--orange)}
.res-bar-fill.danger{background:var(--red)}
.res-status-dot{width:8px;height:8px;border-radius:50%;background:var(--teal);flex-shrink:0}
.res-status-dot.inuse{background:var(--blue)}
.res-status-dot.overloaded{background:var(--red);animation:blink 1s infinite}
.res-status-dot.offline{background:var(--dim)}

/* ── patient board ────────────────────────────────────────────────── */
#center-panel{display:flex;flex-direction:column;gap:10px;overflow:hidden}
#patient-board{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:10px;
  overflow-y:auto;
  padding-right:4px;
  flex:1;
  align-content:start;
}
#patient-board::-webkit-scrollbar{width:4px}
#patient-board::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

.pat-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:12px;position:relative;overflow:hidden;
  transition:border-color .3s,box-shadow .3s;
}
.pat-card::after{
  content:'';position:absolute;top:0;left:0;bottom:0;width:3px;
  background:var(--status-color,var(--green));border-radius:3px 0 0 3px;
}
.pat-card.stable{--status-color:var(--green)}
.pat-card.monitor{--status-color:var(--yellow)}
.pat-card.urgent{--status-color:var(--orange);border-color:#3a2810}
.pat-card.critical{
  --status-color:var(--red);border-color:#3d1520;
  animation:critical-pulse 2s ease-in-out infinite;
}
@keyframes critical-pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(239,35,60,0)}
  50%{box-shadow:0 0 0 6px rgba(239,35,60,.15)}
}
.pat-id{font-size:10px;color:var(--dim);font-family:var(--font-head);letter-spacing:.06em}
.pat-cat{
  font-family:var(--font-head);font-size:15px;font-weight:700;letter-spacing:.03em;
  color:var(--text);margin:3px 0;
}
.pat-status{
  display:inline-block;padding:2px 8px;border-radius:12px;
  font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  background:rgba(0,0,0,.3);color:var(--status-color);
  border:1px solid var(--status-color);
}
.pat-divider{height:1px;background:var(--border);margin:8px 0}
.pat-info{display:flex;flex-direction:column;gap:4px}
.pat-row{display:flex;justify-content:space-between;align-items:center;font-size:11px}
.pat-row .label{color:var(--dim)}
.pat-row .value{color:var(--text);font-weight:600}
.acuity-bar-bg{height:4px;border-radius:2px;background:var(--border);overflow:hidden;margin-top:6px}
.acuity-bar-fill{height:100%;border-radius:2px;transition:width .5s ease,background .5s}
.pat-resource{
  margin-top:7px;padding:4px 8px;border-radius:4px;
  font-size:10px;font-weight:600;font-family:var(--font-head);letter-spacing:.04em;
  background:var(--surface);color:var(--dim);
}
.pat-resource.assigned{color:var(--teal);background:#071a14}

/* ── right panel ──────────────────────────────────────────────────── */
#right-panel{display:flex;flex-direction:column;gap:10px;overflow-y:auto}
#right-panel::-webkit-scrollbar{width:4px}
#right-panel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

/* ── events feed ──────────────────────────────────────────────────── */
.event-feed{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:10px;overflow:hidden;flex:1;
}
.event-list{display:flex;flex-direction:column;gap:4px;max-height:220px;overflow-y:auto}
.event-item{
  display:flex;gap:6px;align-items:flex-start;
  padding:5px 6px;border-radius:5px;background:var(--surface);
  font-size:11px;line-height:1.4;
}
.event-dot{width:6px;height:6px;border-radius:50%;margin-top:3px;flex-shrink:0}
.event-item.discharge .event-dot{background:var(--green)}
.event-item.mortality .event-dot{background:var(--red)}
.event-item.wave .event-dot{background:var(--blue)}
.event-item.default .event-dot{background:var(--dim)}

/* ── ACPL panel ───────────────────────────────────────────────────── */
.acpl-panel{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:10px;
}
.acpl-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.acpl-item{background:var(--surface);border-radius:6px;padding:8px;text-align:center}
.acpl-item .val{font-family:var(--font-head);font-size:20px;font-weight:700;color:var(--teal)}
.acpl-item .lbl{font-size:9px;color:var(--dim);letter-spacing:.06em;text-transform:uppercase;margin-top:2px}
.acpl-note{font-size:10px;color:var(--dim);margin-top:8px;line-height:1.5;padding:6px;background:var(--surface);border-radius:5px}

/* ── empty state ──────────────────────────────────────────────────── */
.empty-state{
  grid-column:1/-1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:40px;color:var(--dim);gap:8px;
}
.empty-state .icon{font-size:40px;opacity:.3}
.empty-state .msg{font-family:var(--font-head);font-size:16px;letter-spacing:.04em}

/* ── wave badge ───────────────────────────────────────────────────── */
.wave-badge{
  font-family:var(--font-head);font-size:11px;font-weight:700;
  padding:2px 8px;border-radius:12px;background:var(--border);color:var(--dim);
  margin-left:auto;letter-spacing:.04em;
}

/* ── center header ────────────────────────────────────────────────── */
.center-header{display:flex;align-items:center;gap:8px}
</style>
</head>
<body>
<div id="app">

<!-- ── TOP BAR ──────────────────────────────────────────────────────────── -->
<div id="topbar">
  <div class="logo">
    <div class="logo-mark">C+</div>
    <div>
      <div class="logo-text">CareOS</div>
      <div class="logo-sub">Resource Management · ACPL Engine</div>
    </div>
  </div>
  <div class="topbar-divider"></div>
  <div id="scenario-badge">Loading…</div>
  <div class="topbar-divider"></div>
  <div class="scenario-selector">
    <button class="sc-btn" onclick="restart('routine')">Routine</button>
    <button class="sc-btn" onclick="restart('surge')">Surge</button>
    <button class="sc-btn" onclick="restart('pandemic')">Pandemic</button>
    <button class="sc-btn" onclick="restart('critical')">Critical</button>
  </div>
  <div class="topbar-right">
    <div id="sim-clock">T = <span id="sim-t">0.0</span>h &nbsp;|&nbsp; Wave <span id="sim-wave">1</span></div>
    <div id="acpl-indicator">
      <div class="dot"></div>
      <span>ACPL Active</span>
      <span id="acpl-updates" style="color:var(--teal);font-weight:700">—</span>
    </div>
  </div>
</div>

<!-- ── MAIN ─────────────────────────────────────────────────────────────── -->
<div id="main">

  <!-- LEFT: KPIs + Resources -->
  <div id="left-panel">
    <div>
      <div class="section-title">Live Overview</div>
      <div class="kpi-grid">
        <div class="kpi-card info" style="--accent:var(--blue)">
          <div class="kpi-label">Active Patients</div>
          <div class="kpi-value" id="kpi-pts">—</div>
          <div class="kpi-sub" id="kpi-wave-sub">Wave 1</div>
        </div>
        <div class="kpi-card danger">
          <div class="kpi-label">Critical Now</div>
          <div class="kpi-value" id="kpi-crit">—</div>
          <div class="kpi-sub">Needs immediate attention</div>
        </div>
        <div class="kpi-card ok">
          <div class="kpi-label">Discharge Rate</div>
          <div class="kpi-value" id="kpi-dr">—</div>
          <div class="kpi-sub" id="kpi-dr-sub">discharges</div>
        </div>
        <div class="kpi-card warn">
          <div class="kpi-label">Avg Wait</div>
          <div class="kpi-value" id="kpi-wait">—</div>
          <div class="kpi-sub">minutes</div>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Bed & Resource Utilisation</div>
      <div class="resource-list" id="resource-list">
        <div class="res-card" style="justify-content:center;color:var(--dim);font-size:12px">Loading resources…</div>
      </div>
    </div>

    <div>
      <div class="section-title">Trends</div>
      <div class="sparkline-row">
        <div class="spark-card">
          <div class="spark-label">Occupancy</div>
          <div class="spark-val"><span id="sp-occ">—</span><span style="font-size:11px;color:var(--dim)">%</span></div>
          <canvas class="spark" id="spark-occ"></canvas>
        </div>
        <div class="spark-card">
          <div class="spark-label">Patient Count</div>
          <div class="spark-val" id="sp-pts">—</div>
          <canvas class="spark" id="spark-pts"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- CENTER: Patient Board -->
  <div id="center-panel">
    <div class="center-header">
      <div class="section-title" style="margin:0">Patient Board</div>
      <div class="wave-badge" id="center-wave">Wave —</div>
    </div>
    <div id="patient-board">
      <div class="empty-state">
        <div class="icon">🏥</div>
        <div class="msg">Connecting to simulation…</div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Events + ACPL -->
  <div id="right-panel">
    <div>
      <div class="section-title">Recent Events</div>
      <div class="event-feed">
        <div class="event-list" id="event-list">
          <div style="color:var(--dim);font-size:11px;padding:8px">No events yet…</div>
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">AI Engine — ACPL Status</div>
      <div class="acpl-panel">
        <div class="acpl-grid">
          <div class="acpl-item">
            <div class="val" id="acpl-loss">—</div>
            <div class="lbl">Policy Loss</div>
          </div>
          <div class="acpl-item">
            <div class="val" id="acpl-lam">—</div>
            <div class="lbl">Mean λ</div>
          </div>
          <div class="acpl-item">
            <div class="val" id="acpl-c">—</div>
            <div class="lbl">Mean C</div>
          </div>
          <div class="acpl-item">
            <div class="val" id="acpl-upd">—</div>
            <div class="lbl">Updates</div>
          </div>
        </div>
        <div class="acpl-note">
          The <strong style="color:var(--teal)">ACPL engine</strong> continuously learns which resource assignments lead to
          better patient outcomes, reducing mortalities and wait times over time.
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Session Summary</div>
      <div class="kpi-grid">
        <div class="kpi-card ok">
          <div class="kpi-label">Discharged</div>
          <div class="kpi-value" id="sum-dis">0</div>
          <div class="kpi-sub">Total</div>
        </div>
        <div class="kpi-card danger">
          <div class="kpi-label">Mortalities</div>
          <div class="kpi-value" id="sum-mort">0</div>
          <div class="kpi-sub">Total</div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /main -->
</div><!-- /app -->

<script>
/* ── helpers ───────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const CONDITION_COLORS = {
  stable:   'var(--green)',
  monitor:  'var(--yellow)',
  urgent:   'var(--orange)',
  critical: 'var(--red)',
};

const RES_ICONS = {
  'GEN-BED':   'GB', 'ICU-BED': 'IC',
  'EMRG-BAY':  'EB', 'SURG':    'SU',
  'STAFF':     'ST',
};

const RES_COLORS = {
  'GEN-BED':   '#1a3a4a', 'ICU-BED':  '#3a1a1a',
  'EMRG-BAY':  '#3a2a1a', 'SURG':     '#1a1a3a',
  'STAFF':     '#1a3a1a',
};

/* ── sparkline renderer ────────────────────────────────────────────── */
function drawSparkline(canvasId, data, color='#00c9a7') {
  const canvas = $(canvasId);
  if (!canvas || !data || data.length < 2) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth  || canvas.width;
  const H = canvas.offsetHeight || 36;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0,0,W,H);
  const mn = Math.min(...data), mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v,i) => [i/(data.length-1)*W, H - (v-mn)/range*(H-4)-2]);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for(let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  ctx.stroke();
  // fill
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, color+'40');
  grad.addColorStop(1, color+'00');
  ctx.fillStyle = grad;
  ctx.fill();
}

/* ── condition → acuity bar color ──────────────────────────────────── */
function acuityColor(acuity) {
  if (acuity >= 85) return 'var(--red)';
  if (acuity >= 65) return 'var(--orange)';
  if (acuity >= 45) return 'var(--yellow)';
  return 'var(--green)';
}

/* ── render patient card ────────────────────────────────────────────── */
function patientCard(p) {
  const waitLabel = p.waitMin > 0 ? `${p.waitMin}m` : '—';
  const careLabel = p.careHr > 0  ? `${p.careHr}h` : '—';
  const resClass  = p.assigned ? 'assigned' : '';
  return `
<div class="pat-card ${p.condClass}" data-uid="${p.uid}">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div class="pat-id">PT-${String(p.uid).padStart(3,'0')} · W${p.wave}</div>
    <span class="pat-status">${p.condition}</span>
  </div>
  <div class="pat-cat">${_catIcon(p.category)} ${p.category}</div>
  <div class="acuity-bar-bg">
    <div class="acuity-bar-fill" style="width:${p.acuity}%;background:${acuityColor(p.acuity)}"></div>
  </div>
  <div class="pat-divider"></div>
  <div class="pat-info">
    <div class="pat-row">
      <span class="label">Waiting</span><span class="value">${waitLabel}</span>
    </div>
    <div class="pat-row">
      <span class="label">In Care</span><span class="value">${careLabel}</span>
    </div>
  </div>
  <div class="pat-resource ${resClass}">
    ${p.assigned ? '✓ ' : '⏳ '}${p.resource}
  </div>
</div>`;
}

function _catIcon(cat) {
  return {EMERGENCY:'🚨',ICU:'🫀',GENERAL:'🏥',SURGICAL:'⚕️',MATERNITY:'👶'}[cat] || '🏥';
}

/* ── render resource row ────────────────────────────────────────────── */
function resourceRow(r) {
  const icon  = RES_ICONS[r.type] || '??';
  const color = RES_COLORS[r.type] || '#1c3044';
  const fillClass = r.utilPct >= 90 ? 'danger' : r.utilPct >= 70 ? 'warn' : '';
  return `
<div class="res-card">
  <div class="res-icon" style="background:${color}">${icon}</div>
  <div class="res-info">
    <div class="res-name">${r.name}</div>
    <div class="res-meta">${r.status} · ${r.used}/${r.capacity} slots</div>
  </div>
  <div class="res-bar-wrap">
    <div class="res-bar-bg">
      <div class="res-bar-fill ${fillClass}" style="width:${r.utilPct}%"></div>
    </div>
    <div style="font-size:9px;color:var(--dim);margin-top:2px;text-align:right">${r.utilPct}%</div>
  </div>
  <div class="res-status-dot ${r.statusClass}"></div>
</div>`;
}

/* ── render event item ──────────────────────────────────────────────── */
function eventItem(text) {
  let cls = 'default';
  if (text.includes('DISCHARG'))   cls = 'discharge';
  else if (text.includes('MORTAL')) cls = 'mortality';
  else if (text.includes('WAVE'))   cls = 'wave';
  return `<div class="event-item ${cls}"><div class="event-dot"></div><span>${text}</span></div>`;
}

/* ── main update ─────────────────────────────────────────────────────── */
let prevPatients = {};
let scenarioActive = '';

function update(data) {
  const {patients, resources, kpi, acpl, sparklines, events, scenario, tick, simTime, wave} = data;

  // topbar
  $('scenario-badge').textContent = scenario;
  $('sim-t').textContent = simTime.toFixed(1);
  $('sim-wave').textContent = wave;
  $('acpl-updates').textContent = acpl.updates.toLocaleString();
  $('center-wave').textContent = `Wave ${wave}`;

  // KPIs
  $('kpi-pts').textContent   = kpi.activePatients;
  $('kpi-crit').textContent  = kpi.criticalCount;
  $('kpi-dr').textContent    = kpi.dischargePct + '%';
  $('kpi-dr-sub').textContent = `${kpi.discharges} discharged`;
  $('kpi-wait').textContent  = kpi.avgWait.toFixed(0);
  $('kpi-wave-sub').textContent = `Wave ${wave}`;

  const critCard = $('kpi-crit').parentElement;
  critCard.className = 'kpi-card ' + (kpi.criticalCount > 0 ? 'danger' : 'ok');

  // Summary
  $('sum-dis').textContent  = kpi.discharges;
  $('sum-mort').textContent = kpi.mortalities;

  // Sparklines
  $('sp-occ').textContent  = sparklines.occupancy.slice(-1)[0]?.toFixed(0) ?? '—';
  $('sp-pts').textContent  = sparklines.patientCount.slice(-1)[0] ?? '—';
  drawSparkline('spark-occ', sparklines.occupancy, '#2a9df4');
  drawSparkline('spark-pts', sparklines.patientCount, '#00c9a7');

  // Resources
  $('resource-list').innerHTML = resources.map(resourceRow).join('');

  // Patient board
  const board = $('patient-board');
  if (patients.length === 0) {
    board.innerHTML = `<div class="empty-state">
      <div class="icon">🛌</div>
      <div class="msg">No active patients</div>
    </div>`;
  } else {
    board.innerHTML = patients.map(patientCard).join('');
  }

  // ACPL
  $('acpl-loss').textContent = acpl.pLoss.toFixed(4);
  $('acpl-lam').textContent  = acpl.meanLam.toFixed(3);
  $('acpl-c').textContent    = acpl.meanC.toFixed(3);
  $('acpl-upd').textContent  = acpl.updates;

  // Events
  if (events.length) {
    $('event-list').innerHTML = events.map(eventItem).join('');
  }
}

/* ── restart scenario ─────────────────────────────────────────────────── */
async function restart(scenario) {
  document.querySelectorAll('.sc-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.sc-btn[onclick*="${scenario}"]`).classList.add('active');
  $('scenario-badge').textContent = 'Restarting…';
  await fetch('/api/restart', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({scenario})
  });
}

/* ── polling loop ──────────────────────────────────────────────────────── */
let pollTimer;
async function poll() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();
    update(data);
  } catch(e) {
    console.warn('Poll error', e);
  }
  pollTimer = setTimeout(poll, 1800);
}

poll();
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# §4  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CareOS — Doctor-facing Hospital Resource Dashboard"
    )
    parser.add_argument("--scenario", type=str, default="routine",
                        choices=["routine", "surge", "pandemic", "critical"])
    parser.add_argument("--port",     type=int, default=5050)
    parser.add_argument("--host",     type=str, default="127.0.0.1")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     CareOS — Intelligent Hospital Resource Dashboard     ║")
    print("║           Powered by ACPL Engine  · Shashank Dev        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Scenario : {args.scenario.upper():<45} ║")
    print(f"║  Dashboard: http://{args.host}:{args.port:<37} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    cfg      = _build_cfg(args.scenario)
    _runner  = SimulationRunner(cfg)

    # Mark the initial scenario button as active via a startup note
    print(f"\n  → Open http://{args.host}:{args.port} in your browser\n")

    import logging as _logging
    _logging.getLogger('werkzeug').setLevel(_logging.WARNING)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
