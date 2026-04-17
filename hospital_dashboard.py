import sys
import json
import time
import threading
import argparse
from pathlib import Path

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
# §1  SIMULATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationRunner:
    TICK_INTERVAL = 0.15
    HISTORY_MAX   = 60

    def __init__(self, cfg: hrms.ExperimentConfig):
        self.cfg     = cfg
        self.sim     = hrms.Simulation(cfg)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SimLoop")
        self._occ_hist  = []
        self._dr_hist   = []
        self._wait_hist = []
        self._pt_hist   = []
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                self.sim.tick()
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

    def snapshot(self) -> dict:
        with self._lock:
            sim = self.sim
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
                    "acuity":     round(p.acuity * 100),
                    "waitMin":    waiting_min,
                    "careHr":     care_hr,
                    "resource":   res_name or "Awaiting Bed",
                    "assigned":   res_name is not None,
                    "dangerLevel": p.danger_level,
                    "wave":       p.wave,
                })
            patients.sort(key=lambda p: -p["acuity"])

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
            acpl = sim.acpl.diagnostics()
            events = list(sim.events)[:12]

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


def _condition(acuity: float):
    if acuity >= 0.85: return "Critical",  "critical"
    if acuity >= 0.65: return "Urgent",    "urgent"
    if acuity >= 0.45: return "Monitor",   "monitor"
    return               "Stable",    "stable"


def _res_status(r: hrms.Resource):
    if r.status == hrms.ResourceStatus.MAINTENANCE: return "Offline",     "offline"
    if r.status == hrms.ResourceStatus.OVERLOADED:  return "At Capacity", "overloaded"
    if r.status == hrms.ResourceStatus.ASSIGNED:    return "In Use",      "inuse"
    return                                                  "Available",   "available"


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
# DASHBOARD HTML — Upgraded Premium UI
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NexCare OS — Hospital Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── RESET ─────────────────────────────────────────────────────────── */
*{margin:0;padding:0;box-sizing:border-box}

/* ── DESIGN TOKENS ─────────────────────────────────────────────────── */
:root {
  --bg:         #f0f4f8;
  --surface:    #ffffff;
  --card:       #ffffff;
  --card2:      #f7f9fc;
  --border:     #e2e8f0;
  --border2:    #cbd5e1;
  --text:       #0f172a;
  --text2:      #334155;
  --dim:        #94a3b8;
  --dim2:       #64748b;

  --teal:       #0ea5e9;
  --teal-glow:  rgba(14,165,233,0.15);
  --teal-dark:  #0284c7;
  --emerald:    #10b981;
  --em-glow:    rgba(16,185,129,0.15);
  --red:        #ef4444;
  --red-glow:   rgba(239,68,68,0.12);
  --amber:      #f59e0b;
  --amb-glow:   rgba(245,158,11,0.12);
  --violet:     #8b5cf6;
  --vio-glow:   rgba(139,92,246,0.12);
  --slate:      #475569;

  --font-display: 'Syne', sans-serif;
  --font-body:    'Outfit', sans-serif;
  --font-mono:    'DM Mono', monospace;
  --radius:    14px;
  --radius-sm: 8px;
  --radius-xs: 5px;
  --shadow:    0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.1);
}

html,body {
  height:100%;
  background: var(--bg);
  background-image:
    radial-gradient(ellipse 80% 60% at 20% -10%, rgba(14,165,233,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 110%, rgba(139,92,246,0.05) 0%, transparent 60%);
  color:var(--text);
  font-family:var(--font-body);
  font-size:13.5px;
  overflow:hidden;
}

/* ── LAYOUT ────────────────────────────────────────────────────────── */
#app {
  display:grid;
  grid-template-rows:62px 1fr;
  height:100vh;
  gap:0;
}

#main {
  display:grid;
  grid-template-columns:290px 1fr 300px;
  gap:14px;
  padding:12px 16px 14px;
  overflow:hidden;
}

/* ── TOPBAR ────────────────────────────────────────────────────────── */
#topbar {
  display:flex;
  align-items:center;
  gap:14px;
  padding:0 20px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  box-shadow:var(--shadow);
  position:relative;
  z-index:10;
}

.logo {
  display:flex;
  align-items:center;
  gap:10px;
  flex-shrink:0;
}

.logo-mark {
  width:36px;height:36px;
  border-radius:10px;
  background:linear-gradient(135deg,#0ea5e9,#8b5cf6);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 8px rgba(14,165,233,0.4);
}
.logo-mark svg {stroke:#fff;fill:none;width:18px;height:18px;}

.logo-text {
  font-family:var(--font-display);
  font-size:18px;font-weight:800;
  color:var(--text);
  letter-spacing:-.02em;
}
.logo-sub {
  font-size:10px;color:var(--dim);
  letter-spacing:.06em;text-transform:uppercase;
  margin-top:-2px;
}

.top-divider {
  width:1px;height:28px;
  background:var(--border);
  margin:0 2px;
}

/* Scenario badge */
#scenario-badge {
  padding:5px 14px;border-radius:20px;
  font-family:var(--font-body);font-size:12px;font-weight:600;
  background:linear-gradient(135deg,rgba(14,165,233,0.1),rgba(139,92,246,0.08));
  color:var(--teal-dark);
  border:1px solid rgba(14,165,233,0.25);
  letter-spacing:.01em;
}

/* Scenario buttons */
.sc-strip {
  display:flex;gap:4px;
  background:var(--card2);
  border:1px solid var(--border);
  border-radius:10px;
  padding:3px;
}
.sc-btn {
  padding:4px 12px;border-radius:7px;
  border:none;
  background:transparent;
  color:var(--dim2);
  font-family:var(--font-body);font-size:11.5px;font-weight:600;
  cursor:pointer;transition:all .18s;letter-spacing:.01em;
  white-space:nowrap;
}
.sc-btn:hover { background:var(--border);color:var(--text2); }
.sc-btn.active {
  background:var(--surface);color:var(--teal-dark);
  box-shadow:var(--shadow);
}
.sc-btn.active.surge   { color:#dc2626; }
.sc-btn.active.pandemic{ color:#7c3aed; }
.sc-btn.active.critical{ color:var(--amber); }

/* Live clock / wave */
.topbar-right {
  margin-left:auto;
  display:flex;align-items:center;gap:12px;
}
.clock-chip {
  display:flex;align-items:center;gap:8px;
  background:var(--card2);border:1px solid var(--border);
  border-radius:8px;padding:5px 12px;
  font-family:var(--font-mono);font-size:12px;color:var(--dim2);
}
.clock-chip b { color:var(--text);font-weight:500; }

.acpl-chip {
  display:flex;align-items:center;gap:7px;
  background: linear-gradient(135deg,rgba(14,165,233,0.08),rgba(139,92,246,0.06));
  border:1px solid rgba(14,165,233,0.2);
  border-radius:8px;padding:5px 12px;
  font-size:11px;color:var(--teal-dark);font-weight:600;
}
.pulse-dot {
  width:7px;height:7px;border-radius:50%;
  background:var(--teal);
  animation:pulse-ring 2s ease-in-out infinite;
}
@keyframes pulse-ring {
  0%,100%{box-shadow:0 0 0 0 rgba(14,165,233,0.5)}
  50%{box-shadow:0 0 0 5px rgba(14,165,233,0)}
}

/* ── SECTION TITLES ────────────────────────────────────────────────── */
.sec-title {
  font-family:var(--font-display);
  font-size:11px;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin-bottom:9px;
}

/* ── SCROLLABLE PANELS ─────────────────────────────────────────────── */
#left-panel {
  display:flex;flex-direction:column;gap:12px;
  overflow-y:auto;overflow-x:hidden;
}
#left-panel::-webkit-scrollbar { width:3px; }
#left-panel::-webkit-scrollbar-thumb { background:var(--border2);border-radius:4px; }

#right-panel {
  display:flex;flex-direction:column;gap:12px;overflow-y:auto;
}
#right-panel::-webkit-scrollbar { width:3px; }
#right-panel::-webkit-scrollbar-thumb { background:var(--border2);border-radius:4px; }

/* ── KPI CARDS ─────────────────────────────────────────────────────── */
.kpi-grid { display:grid;grid-template-columns:1fr 1fr;gap:8px; }

.kpi-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:14px 14px 12px;
  position:relative;overflow:hidden;
  box-shadow:var(--shadow);
  transition:box-shadow .2s, transform .2s;
}
.kpi-card:hover { box-shadow:var(--shadow-md);transform:translateY(-1px); }

.kpi-card-accent {
  position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--kpi-color, var(--teal));
  border-radius:var(--radius) var(--radius) 0 0;
}
.kpi-icon-bg {
  position:absolute;right:12px;top:12px;
  width:32px;height:32px;border-radius:8px;
  background:var(--kpi-bg,rgba(14,165,233,0.1));
  display:flex;align-items:center;justify-content:center;
  font-size:14px;
}
.kpi-label {
  font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim2);margin-bottom:4px;
}
.kpi-value {
  font-family:var(--font-display);
  font-size:32px;font-weight:800;line-height:1;
  color:var(--kpi-color,var(--teal));
  letter-spacing:-.02em;
}
.kpi-sub { font-size:10.5px;color:var(--dim);margin-top:4px; }

/* color presets */
.kpi-blue  { --kpi-color:var(--teal);    --kpi-bg:var(--teal-glow); }
.kpi-red   { --kpi-color:var(--red);     --kpi-bg:var(--red-glow); }
.kpi-green { --kpi-color:var(--emerald); --kpi-bg:var(--em-glow); }
.kpi-amber { --kpi-color:var(--amber);   --kpi-bg:var(--amb-glow); }
.kpi-occ   { --kpi-color:var(--violet);  --kpi-bg:var(--vio-glow); }

/* alert pulse on critical */
.kpi-card.alert-pulse { animation:kpi-alert 2s ease-in-out infinite; }
@keyframes kpi-alert {
  0%,100%{background:var(--card)}
  50%{background:rgba(239,68,68,0.04)}
}

/* ── OCCUPANCY RING ────────────────────────────────────────────────── */
.occ-ring-wrap {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:14px;
  display:flex;align-items:center;gap:14px;
  box-shadow:var(--shadow);
}
.ring-container { position:relative;width:64px;height:64px;flex-shrink:0; }
.ring-svg { width:64px;height:64px;transform:rotate(-90deg); }
.ring-bg  { fill:none;stroke:var(--border);stroke-width:5; }
.ring-fg  { fill:none;stroke:var(--violet);stroke-width:5;stroke-linecap:round;
            transition:stroke-dashoffset .6s cubic-bezier(.4,0,.2,1); }
.ring-label {
  position:absolute;inset:0;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--font-display);font-size:13px;font-weight:800;
  color:var(--violet);
}
.ring-info { flex:1; }
.ring-title { font-family:var(--font-display);font-size:14px;font-weight:700;color:var(--text); }
.ring-sub   { font-size:11px;color:var(--dim);margin-top:3px; }

/* ── SPARKLINES ────────────────────────────────────────────────────── */
.spark-pair { display:grid;grid-template-columns:1fr 1fr;gap:8px; }
.spark-card {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:10px;
  box-shadow:var(--shadow);
}
.spark-label { font-size:9.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--dim);margin-bottom:3px; }
.spark-val {
  font-family:var(--font-display);font-size:19px;font-weight:800;
  color:var(--text);margin-bottom:5px;letter-spacing:-.01em;
}
canvas.spark { width:100%;height:32px;display:block; }

/* ── RESOURCE SECTION ──────────────────────────────────────────────── */
.res-scroll { display:flex;flex-direction:column;gap:5px; }
.res-card {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:9px 11px;
  display:flex;align-items:center;gap:10px;
  box-shadow:var(--shadow);
  transition:border-color .2s;
}
.res-card:hover { border-color:var(--border2); }

.res-icon {
  width:30px;height:30px;border-radius:7px;
  display:flex;align-items:center;justify-content:center;
  font-size:10px;font-family:var(--font-mono);
  font-weight:500;flex-shrink:0;
  background:var(--icon-bg,#e0f2fe);color:var(--icon-fg,#0284c7);
}
.res-info { flex:1;min-width:0; }
.res-name { font-size:12px;font-weight:600;color:var(--text);letter-spacing:.01em; }
.res-meta { font-size:10px;color:var(--dim);margin-top:1px; }
.res-bar-wrap { width:72px;flex-shrink:0; }
.res-bar-bg  { height:4px;border-radius:2px;background:var(--border);overflow:hidden; }
.res-bar-fill {
  height:100%;border-radius:2px;
  background:var(--emerald);
  transition:width .4s ease,background .4s;
}
.res-bar-fill.warn   { background:var(--amber); }
.res-bar-fill.danger { background:var(--red); }
.res-pct { font-size:9px;color:var(--dim);text-align:right;margin-top:2px; }

.res-dot {
  width:8px;height:8px;border-radius:50%;
  background:var(--emerald);flex-shrink:0;
}
.res-dot.inuse     { background:var(--teal); }
.res-dot.overloaded{ background:var(--red);animation:pulse-ring 1s infinite; }
.res-dot.offline   { background:var(--border2); }

/* ── CENTER PANEL ──────────────────────────────────────────────────── */
#center-panel { display:flex;flex-direction:column;gap:10px;overflow:hidden; }

.center-hdr {
  display:flex;align-items:center;gap:10px;
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:10px 14px;
  box-shadow:var(--shadow);
}
.center-hdr-title {
  font-family:var(--font-display);font-size:14px;font-weight:800;color:var(--text);
}
.wave-pill {
  padding:3px 10px;border-radius:20px;
  background:rgba(14,165,233,0.1);color:var(--teal-dark);
  font-size:11px;font-weight:700;border:1px solid rgba(14,165,233,0.2);
}
.search-box {
  margin-left:auto;
  display:flex;align-items:center;gap:6px;
  background:var(--card2);border:1px solid var(--border);
  border-radius:7px;padding:4px 10px;
}
.search-box input {
  border:none;outline:none;background:transparent;
  font-family:var(--font-body);font-size:12px;color:var(--text);
  width:130px;
}
.search-box input::placeholder { color:var(--dim); }
.cat-filter {
  display:flex;gap:4px;flex-wrap:wrap;
}
.cat-btn {
  padding:3px 9px;border-radius:6px;
  border:1px solid var(--border);background:transparent;
  color:var(--dim2);font-size:11px;font-weight:600;
  cursor:pointer;transition:all .15s;
}
.cat-btn:hover { background:var(--border);color:var(--text2); }
.cat-btn.active {
  background:var(--text);color:#fff;border-color:var(--text);
}

/* ── PATIENT BOARD ─────────────────────────────────────────────────── */
#patient-board {
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(195px,1fr));
  gap:9px;
  overflow-y:auto;
  padding-right:3px;
  flex:1;
  align-content:start;
}
#patient-board::-webkit-scrollbar { width:3px; }
#patient-board::-webkit-scrollbar-thumb { background:var(--border2);border-radius:4px; }

.pat-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:12px;
  position:relative;overflow:hidden;
  box-shadow:var(--shadow);
  transition:box-shadow .2s,transform .2s,border-color .3s;
}
.pat-card:hover { box-shadow:var(--shadow-md);transform:translateY(-1px); }

.pat-stripe {
  position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--stripe-c,var(--emerald));
}

.pat-card.stable  { --stripe-c:var(--emerald); }
.pat-card.monitor { --stripe-c:var(--amber); }
.pat-card.urgent  { --stripe-c:#f97316;border-color:#fde68a; }
.pat-card.critical{
  --stripe-c:var(--red);
  border-color:#fecaca;
  animation:crit-pulse 2s ease-in-out infinite;
}
@keyframes crit-pulse {
  0%,100%{box-shadow:var(--shadow)}
  50%{box-shadow:0 0 0 4px rgba(239,68,68,0.12)}
}

.pat-top { display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px; }
.pat-id { font-family:var(--font-mono);font-size:10px;color:var(--dim); }
.pat-badge {
  padding:2px 7px;border-radius:10px;
  font-size:9.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  background:var(--badge-bg);color:var(--badge-fg);
}
.pat-card.stable  .pat-badge { --badge-bg:rgba(16,185,129,0.1); --badge-fg:#059669; }
.pat-card.monitor .pat-badge { --badge-bg:rgba(245,158,11,0.1);  --badge-fg:#d97706; }
.pat-card.urgent  .pat-badge { --badge-bg:rgba(249,115,22,0.1);  --badge-fg:#ea580c; }
.pat-card.critical .pat-badge{ --badge-bg:rgba(239,68,68,0.1);   --badge-fg:#dc2626; }

.pat-cat {
  font-family:var(--font-body);font-size:14px;font-weight:700;
  color:var(--text);margin-bottom:2px;
}
.pat-wave { font-size:10px;color:var(--dim); }

.acuity-wrap { margin:8px 0; }
.acuity-label {
  display:flex;justify-content:space-between;
  font-size:9.5px;color:var(--dim);margin-bottom:3px;
}
.acuity-bg {
  height:5px;border-radius:3px;background:var(--border);overflow:hidden;
}
.acuity-fill {
  height:100%;border-radius:3px;
  transition:width .5s ease,background .5s;
}

.pat-stats {
  display:grid;grid-template-columns:1fr 1fr;gap:4px;
  font-size:10.5px;margin-top:6px;
}
.pat-stat {color:var(--dim);}
.pat-stat span { color:var(--text2);font-weight:600; }

.pat-res {
  margin-top:7px;padding:4px 8px;border-radius:5px;
  font-size:10px;font-weight:600;font-family:var(--font-mono);
  background:var(--card2);color:var(--dim);
  display:flex;align-items:center;gap:4px;
}
.pat-res.assigned { color:var(--teal-dark);background:rgba(14,165,233,0.07); }

.empty-state {
  grid-column:1/-1;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:48px;color:var(--dim);gap:10px;
}
.empty-state .e-icon { font-size:40px;opacity:.4; }
.empty-state .e-msg {
  font-family:var(--font-display);font-size:15px;font-weight:700;
  letter-spacing:.02em;
}

/* ── EVENT FEED ────────────────────────────────────────────────────── */
.event-container {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;
  box-shadow:var(--shadow);
  flex:1;
  display:flex;flex-direction:column;
}
.event-hdr {
  padding:10px 12px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:6px;
}
.event-hdr .live-dot {
  width:6px;height:6px;border-radius:50%;background:var(--red);
  animation:pulse-ring 1.5s ease-in-out infinite;
}
.event-hdr span {
  font-size:10.5px;font-weight:700;color:var(--text2);
  letter-spacing:.04em;text-transform:uppercase;
}
.event-list {
  display:flex;flex-direction:column;
  gap:1px;overflow-y:auto;
  max-height:210px;padding:6px;
}
.event-list::-webkit-scrollbar { width:3px; }
.event-list::-webkit-scrollbar-thumb { background:var(--border2);border-radius:4px; }

.ev-item {
  display:flex;gap:8px;align-items:flex-start;
  padding:6px 8px;border-radius:6px;
  font-size:11.5px;line-height:1.4;
  transition:background .2s;
}
.ev-item:hover { background:var(--card2); }
.ev-dot {
  width:7px;height:7px;border-radius:50%;
  margin-top:3px;flex-shrink:0;
  background:var(--dim);
}
.ev-item.discharge .ev-dot { background:var(--emerald); }
.ev-item.mortality  .ev-dot { background:var(--red); }
.ev-item.wave       .ev-dot { background:var(--teal); }
.ev-item.discharge  { color:var(--text2); }
.ev-item.mortality  { color:#7f1d1d; }

/* ── ACPL PANEL ────────────────────────────────────────────────────── */
.acpl-panel {
  background:linear-gradient(135deg,rgba(14,165,233,0.05),rgba(139,92,246,0.05));
  border:1px solid rgba(14,165,233,0.2);
  border-radius:var(--radius);padding:12px;
  box-shadow:var(--shadow);
}
.acpl-header {
  display:flex;align-items:center;gap:7px;margin-bottom:10px;
}
.acpl-header .ai-badge {
  padding:2px 8px;border-radius:5px;
  background:linear-gradient(135deg,#0ea5e9,#8b5cf6);
  color:#fff;font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.acpl-header .title {
  font-family:var(--font-display);font-size:13px;font-weight:800;color:var(--text);
}
.acpl-metrics { display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px; }
.acpl-metric {
  background:rgba(255,255,255,0.7);border:1px solid var(--border);
  border-radius:7px;padding:8px;text-align:center;
}
.acpl-val {
  font-family:var(--font-mono);font-size:17px;font-weight:500;
  color:var(--teal-dark);
}
.acpl-lbl { font-size:9px;color:var(--dim);letter-spacing:.06em;text-transform:uppercase;margin-top:2px; }
.acpl-note {
  font-size:10.5px;color:var(--dim2);line-height:1.55;
  padding:8px;background:rgba(255,255,255,0.6);border-radius:6px;
  border-left:2px solid var(--teal);padding-left:10px;
}

/* ── SESSION SUMMARY ───────────────────────────────────────────────── */
.summary-strip {
  display:grid;grid-template-columns:1fr 1fr;gap:8px;
}
.sum-card {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:11px;
  box-shadow:var(--shadow);text-align:center;
}
.sum-val {
  font-family:var(--font-display);font-size:26px;font-weight:800;
  line-height:1;
}
.sum-val.good { color:var(--emerald); }
.sum-val.bad  { color:var(--red); }
.sum-lbl { font-size:10px;color:var(--dim);margin-top:3px;font-weight:600;text-transform:uppercase;letter-spacing:.05em; }

/* ── CHART SECTION ─────────────────────────────────────────────────── */
.chart-wrap {
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:12px 14px;
  box-shadow:var(--shadow);
}
.chart-title {
  font-size:11px;font-weight:700;color:var(--dim2);
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;
}

/* ── ANIMATIONS ────────────────────────────────────────────────────── */
@keyframes fadeSlideIn {
  from{opacity:0;transform:translateY(6px)}
  to{opacity:1;transform:translateY(0)}
}
.pat-card { animation:fadeSlideIn .25s ease; }
.ev-item  { animation:fadeSlideIn .2s ease; }
</style>
</head>
<body>
<div id="app">

<!-- ══════════════════════════════ TOP BAR ══════════════════════════════════ -->
<div id="topbar">
  <div class="logo">
    <div class="logo-mark">
      <svg viewBox="0 0 24 24" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div>
      <div class="logo-text">NexCare OS</div>
      <div class="logo-sub">Hospital Intelligence Platform · ACPL Engine</div>
    </div>
  </div>

  <div class="top-divider"></div>
  <div id="scenario-badge">Loading…</div>
  <div class="top-divider"></div>

  <div class="sc-strip">
    <button class="sc-btn routine" onclick="restart('routine')">Routine</button>
    <button class="sc-btn surge"   onclick="restart('surge')">Surge</button>
    <button class="sc-btn pandemic" onclick="restart('pandemic')">Pandemic</button>
    <button class="sc-btn critical" onclick="restart('critical')">Critical</button>
  </div>

  <div class="topbar-right">
    <div class="clock-chip">
      T = <b id="sim-t">0.0</b>h &nbsp;·&nbsp; Wave <b id="sim-wave">—</b>
    </div>
    <div class="acpl-chip">
      <div class="pulse-dot"></div>
      ACPL Active &nbsp;·&nbsp; <b id="acpl-updates" style="color:var(--violet)">—</b> updates
    </div>
  </div>
</div>

<!-- ══════════════════════════════ MAIN ══════════════════════════════════════ -->
<div id="main">

  <!-- ── LEFT PANEL ──────────────────────────────────────────────── -->
  <div id="left-panel">

    <!-- KPI Row -->
    <div>
      <div class="sec-title">Live KPIs</div>
      <div class="kpi-grid">
        <div class="kpi-card kpi-blue">
          <div class="kpi-card-accent"></div>
          <div class="kpi-icon-bg">🏥</div>
          <div class="kpi-label">Active Patients</div>
          <div class="kpi-value" id="kpi-pts">—</div>
          <div class="kpi-sub" id="kpi-wave-sub">—</div>
        </div>
        <div class="kpi-card kpi-red" id="crit-card">
          <div class="kpi-card-accent"></div>
          <div class="kpi-icon-bg">🚨</div>
          <div class="kpi-label">Critical Now</div>
          <div class="kpi-value" id="kpi-crit">—</div>
          <div class="kpi-sub">Needs immediate care</div>
        </div>
        <div class="kpi-card kpi-green">
          <div class="kpi-card-accent"></div>
          <div class="kpi-icon-bg">✅</div>
          <div class="kpi-label">Discharge Rate</div>
          <div class="kpi-value" id="kpi-dr">—</div>
          <div class="kpi-sub" id="kpi-dr-sub">discharges</div>
        </div>
        <div class="kpi-card kpi-amber">
          <div class="kpi-card-accent"></div>
          <div class="kpi-icon-bg">⏱</div>
          <div class="kpi-label">Avg Wait</div>
          <div class="kpi-value" id="kpi-wait">—</div>
          <div class="kpi-sub">minutes</div>
        </div>
      </div>
    </div>

    <!-- Occupancy ring -->
    <div class="occ-ring-wrap">
      <div class="ring-container">
        <svg class="ring-svg" viewBox="0 0 64 64">
          <circle class="ring-bg" cx="32" cy="32" r="28"/>
          <circle class="ring-fg" id="occ-ring" cx="32" cy="32" r="28"
                  stroke-dasharray="175.9" stroke-dashoffset="175.9"/>
        </svg>
        <div class="ring-label" id="occ-ring-label">0%</div>
      </div>
      <div class="ring-info">
        <div class="ring-title">Bed Occupancy</div>
        <div class="ring-sub">Current utilization across all resources</div>
        <div style="margin-top:8px;font-family:var(--font-mono);font-size:11px;color:var(--dim)">
          Avail slots: <b id="kpi-avail" style="color:var(--emerald)">—</b>
        </div>
      </div>
    </div>

    <!-- Sparklines -->
    <div>
      <div class="sec-title">Trends</div>
      <div class="spark-pair">
        <div class="spark-card">
          <div class="spark-label">Occupancy</div>
          <div class="spark-val"><span id="sp-occ">—</span><span style="font-size:11px;color:var(--dim)">%</span></div>
          <canvas class="spark" id="spark-occ"></canvas>
        </div>
        <div class="spark-card">
          <div class="spark-label">Patient Load</div>
          <div class="spark-val" id="sp-pts">—</div>
          <canvas class="spark" id="spark-pts"></canvas>
        </div>
      </div>
    </div>

    <!-- Resources -->
    <div>
      <div class="sec-title">Resource Allocation</div>
      <div class="res-scroll" id="resource-list">
        <div class="res-card" style="justify-content:center;color:var(--dim);font-size:11px">Loading…</div>
      </div>
    </div>

  </div><!-- /left-panel -->

  <!-- ── CENTER PANEL ─────────────────────────────────────────────── -->
  <div id="center-panel">

    <!-- Header bar -->
    <div class="center-hdr">
      <div class="center-hdr-title">Patient Monitor</div>
      <div class="wave-pill" id="center-wave">Wave —</div>
      <div class="cat-filter" id="cat-filter">
        <button class="cat-btn active" data-cat="ALL" onclick="setFilter('ALL',this)">All</button>
        <button class="cat-btn" data-cat="EMERGENCY" onclick="setFilter('EMERGENCY',this)">🚨 Emergency</button>
        <button class="cat-btn" data-cat="ICU"       onclick="setFilter('ICU',this)">🫀 ICU</button>
        <button class="cat-btn" data-cat="GENERAL"   onclick="setFilter('GENERAL',this)">🏥 General</button>
        <button class="cat-btn" data-cat="SURGICAL"  onclick="setFilter('SURGICAL',this)">⚕️ Surgical</button>
      </div>
      <div class="search-box">
        <span style="color:var(--dim);font-size:12px">🔍</span>
        <input id="pat-search" type="text" placeholder="Search patient…" oninput="applyFilter()"/>
      </div>
    </div>

    <!-- Patient grid -->
    <div id="patient-board">
      <div class="empty-state">
        <div class="e-icon">🏥</div>
        <div class="e-msg">Connecting to simulation…</div>
      </div>
    </div>
  </div><!-- /center-panel -->

  <!-- ── RIGHT PANEL ──────────────────────────────────────────────── -->
  <div id="right-panel">

    <!-- Events -->
    <div>
      <div class="event-container">
        <div class="event-hdr">
          <div class="live-dot"></div>
          <span>Live Event Feed</span>
        </div>
        <div class="event-list" id="event-list">
          <div class="ev-item"><div class="ev-dot"></div><span style="color:var(--dim)">No events yet…</span></div>
        </div>
      </div>
    </div>

    <!-- ACPL Status -->
    <div class="acpl-panel">
      <div class="acpl-header">
        <span class="ai-badge">AI</span>
        <span class="title">ACPL Engine Status</span>
      </div>
      <div class="acpl-metrics">
        <div class="acpl-metric">
          <div class="acpl-val" id="acpl-loss">—</div>
          <div class="acpl-lbl">Policy Loss</div>
        </div>
        <div class="acpl-metric">
          <div class="acpl-val" id="acpl-lam">—</div>
          <div class="acpl-lbl">Mean λ</div>
        </div>
        <div class="acpl-metric">
          <div class="acpl-val" id="acpl-c">—</div>
          <div class="acpl-lbl">Mean C</div>
        </div>
        <div class="acpl-metric">
          <div class="acpl-val" id="acpl-upd">—</div>
          <div class="acpl-lbl">Updates</div>
        </div>
      </div>
      <div class="acpl-note">
        ACPL continuously learns which resource assignments
        lead to better patient outcomes — reducing mortality
        and wait times over time.
      </div>
    </div>

    <!-- Session summary -->
    <div>
      <div class="sec-title">Session Summary</div>
      <div class="summary-strip">
        <div class="sum-card">
          <div class="sum-val good" id="sum-dis">0</div>
          <div class="sum-lbl">Discharged</div>
        </div>
        <div class="sum-card">
          <div class="sum-val bad" id="sum-mort">0</div>
          <div class="sum-lbl">Mortalities</div>
        </div>
      </div>
    </div>

    <!-- Charts -->
    <div class="chart-wrap">
      <div class="chart-title">Discharge Rate %</div>
      <canvas id="chart-dr" height="70"></canvas>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Avg Wait Time (min)</div>
      <canvas id="chart-wait" height="70"></canvas>
    </div>

  </div><!-- /right-panel -->

</div><!-- /main -->
</div><!-- /app -->

<script>
/* ─── helpers ──────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
let _allPatients = [];
let _activeFilter = 'ALL';

const RES_ICONS = {
  'GEN-BED':'GB','ICU-BED':'IC','EMRG-BAY':'EB','SURG':'SU','STAFF':'ST'
};
const RES_ICON_STYLES = {
  'GEN-BED': 'background:#e0f2fe;color:#0284c7',
  'ICU-BED':  'background:#fee2e2;color:#dc2626',
  'EMRG-BAY': 'background:#fef3c7;color:#d97706',
  'SURG':     'background:#ede9fe;color:#7c3aed',
  'STAFF':    'background:#dcfce7;color:#15803d',
};
const CAT_ICONS = {EMERGENCY:'🚨',ICU:'🫀',GENERAL:'🏥',SURGICAL:'⚕️',MATERNITY:'👶'};

/* ─── sparkline ────────────────────────────────────────────────────── */
function drawSparkline(id, data, color='#0ea5e9') {
  const cv = $(id); if(!cv||!data||data.length<2) return;
  const dpr=window.devicePixelRatio||1;
  const W=cv.offsetWidth||cv.clientWidth||100, H=36;
  cv.width=W*dpr; cv.height=H*dpr;
  const ctx=cv.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const mn=Math.min(...data), mx=Math.max(...data), range=mx-mn||1;
  const pts=data.map((v,i)=>[i/(data.length-1)*W, H-(v-mn)/range*(H-6)-3]);
  ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);
  for(let i=1;i<pts.length;i++){
    const cp=[(pts[i-1][0]+pts[i][0])/2,(pts[i-1][1]+pts[i][1])/2];
    ctx.quadraticCurveTo(pts[i-1][0],pts[i-1][1],cp[0],cp[1]);
  }
  ctx.lineTo(pts[pts.length-1][0],pts[pts.length-1][1]);
  ctx.strokeStyle=color; ctx.lineWidth=1.8; ctx.lineJoin='round'; ctx.stroke();
  ctx.lineTo(W,H); ctx.lineTo(0,H); ctx.closePath();
  const grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,color+'50'); grad.addColorStop(1,color+'00');
  ctx.fillStyle=grad; ctx.fill();
}

/* ─── occupancy ring ───────────────────────────────────────────────── */
function updateRing(pct) {
  const circumference = 175.9;
  const offset = circumference - (pct/100)*circumference;
  const ring = $('occ-ring');
  ring.style.strokeDashoffset = offset;
  const color = pct>85?'#ef4444':pct>65?'#f59e0b':'#8b5cf6';
  ring.style.stroke = color;
  $('occ-ring-label').textContent = pct.toFixed(0)+'%';
  $('occ-ring-label').style.color = color;
}

/* ─── charts ────────────────────────────────────────────────────────── */
const chartOpts = (color, label) => ({
  type:'line',
  data:{labels:[],datasets:[{
    data:[],
    borderColor:color,borderWidth:1.5,
    backgroundColor:color+'20',
    fill:true,tension:.4,pointRadius:0
  }]},
  options:{
    animation:false,responsive:true,maintainAspectRatio:true,
    plugins:{legend:{display:false},tooltip:{enabled:false}},
    scales:{
      x:{display:false},
      y:{
        display:true,
        grid:{color:'rgba(0,0,0,0.05)',drawBorder:false},
        ticks:{color:'#94a3b8',font:{size:9},maxTicksLimit:4},
        border:{display:false}
      }
    }
  }
});

const chartDR   = new Chart($('chart-dr'),   chartOpts('#10b981','DR'));
const chartWait = new Chart($('chart-wait'),  chartOpts('#f59e0b','Wait'));
const MAX_CHART = 50;

function pushChart(chart, val) {
  chart.data.labels.push('');
  chart.data.datasets[0].data.push(val);
  if(chart.data.labels.length > MAX_CHART) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

/* ─── patient card ──────────────────────────────────────────────────── */
function acuityColor(a) {
  if(a>=85) return '#ef4444';
  if(a>=65) return '#f97316';
  if(a>=45) return '#f59e0b';
  return '#10b981';
}

function patCard(p) {
  const waitLbl = p.waitMin>0?`${p.waitMin}m`:'—';
  const careLbl = p.careHr >0?`${p.careHr}h`:'—';
  const icon = CAT_ICONS[p.category]||'🏥';
  return `
<div class="pat-card ${p.condClass}">
  <div class="pat-stripe"></div>
  <div class="pat-top">
    <div class="pat-id">PT-${String(p.uid).padStart(3,'0')} · W${p.wave}</div>
    <span class="pat-badge">${p.condition}</span>
  </div>
  <div class="pat-cat">${icon} ${p.category}</div>
  <div class="acuity-wrap">
    <div class="acuity-label"><span>Acuity</span><span>${p.acuity}%</span></div>
    <div class="acuity-bg"><div class="acuity-fill" style="width:${p.acuity}%;background:${acuityColor(p.acuity)}"></div></div>
  </div>
  <div class="pat-stats">
    <div class="pat-stat">Wait <span>${waitLbl}</span></div>
    <div class="pat-stat">Care <span>${careLbl}</span></div>
  </div>
  <div class="pat-res${p.assigned?' assigned':''}">
    ${p.assigned?'✓':'⏳'} ${p.resource}
  </div>
</div>`;
}

/* ─── resource row ──────────────────────────────────────────────────── */
function resCard(r) {
  const icon  = RES_ICONS[r.type] || '??';
  const style = RES_ICON_STYLES[r.type] || '';
  const fill  = r.utilPct>=90?'danger':r.utilPct>=70?'warn':'';
  return `
<div class="res-card">
  <div class="res-icon" style="${style}">${icon}</div>
  <div class="res-info">
    <div class="res-name">${r.name}</div>
    <div class="res-meta">${r.status} · ${r.used}/${r.capacity}</div>
  </div>
  <div class="res-bar-wrap">
    <div class="res-bar-bg"><div class="res-bar-fill ${fill}" style="width:${r.utilPct}%"></div></div>
    <div class="res-pct">${r.utilPct}%</div>
  </div>
  <div class="res-dot ${r.statusClass}"></div>
</div>`;
}

/* ─── event item ────────────────────────────────────────────────────── */
function evItem(text) {
  let cls='';
  if(text.includes('DISCHARG')||text.includes('discharg'))   cls='discharge';
  else if(text.includes('MORTAL')||text.includes('mortal'))  cls='mortality';
  else if(text.includes('WAVE')||text.includes('Wave'))      cls='wave';
  return `<div class="ev-item ${cls}"><div class="ev-dot"></div><span>${text}</span></div>`;
}

/* ─── filter logic ──────────────────────────────────────────────────── */
function setFilter(cat, btn) {
  _activeFilter = cat;
  document.querySelectorAll('.cat-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  applyFilter();
}

function applyFilter() {
  const q = ($('pat-search').value||'').toLowerCase();
  let pts = _allPatients;
  if(_activeFilter!=='ALL') pts = pts.filter(p=>p.category===_activeFilter);
  if(q) pts = pts.filter(p=>String(p.uid).includes(q)||p.category.toLowerCase().includes(q));
  renderPatients(pts);
}

function renderPatients(pts) {
  const board = $('patient-board');
  if(!pts.length) {
    board.innerHTML=`<div class="empty-state"><div class="e-icon">🛌</div><div class="e-msg">No patients match filter</div></div>`;
    return;
  }
  board.innerHTML = pts.map(patCard).join('');
}

/* ─── main update ───────────────────────────────────────────────────── */
function update(d) {
  const {patients,resources,kpi,acpl,sparklines,events,scenario,simTime,wave,tick} = d;

  // topbar
  $('scenario-badge').textContent = scenario;
  $('sim-t').textContent = simTime.toFixed(1);
  $('sim-wave').textContent = wave;
  $('acpl-updates').textContent = acpl.updates.toLocaleString();
  $('center-wave').textContent = `Wave ${wave}`;

  // KPIs
  $('kpi-pts').textContent  = kpi.activePatients;
  $('kpi-wave-sub').textContent = `Wave ${wave} active`;
  $('kpi-crit').textContent = kpi.criticalCount;
  $('kpi-dr').textContent   = kpi.dischargePct + '%';
  $('kpi-dr-sub').textContent = `${kpi.discharges} total discharged`;
  $('kpi-wait').textContent = kpi.avgWait.toFixed(0);
  $('kpi-avail').textContent = kpi.availableSlots;

  const cc = $('crit-card');
  cc.classList.toggle('alert-pulse', kpi.criticalCount > 0);

  // Occupancy ring
  updateRing(kpi.occupancyPct);

  // Sparklines
  $('sp-occ').textContent = (sparklines.occupancy.slice(-1)[0]||0).toFixed(0);
  $('sp-pts').textContent = sparklines.patientCount.slice(-1)[0] ?? '—';
  drawSparkline('spark-occ', sparklines.occupancy, '#8b5cf6');
  drawSparkline('spark-pts', sparklines.patientCount, '#0ea5e9');

  // Resources
  $('resource-list').innerHTML = resources.map(resCard).join('');

  // Patients
  _allPatients = patients;
  applyFilter();

  // ACPL
  $('acpl-loss').textContent = acpl.pLoss.toFixed(4);
  $('acpl-lam').textContent  = acpl.meanLam.toFixed(3);
  $('acpl-c').textContent    = acpl.meanC.toFixed(3);
  $('acpl-upd').textContent  = acpl.updates;

  // Summary
  $('sum-dis').textContent  = kpi.discharges;
  $('sum-mort').textContent = kpi.mortalities;

  // Events
  if(events.length) $('event-list').innerHTML = events.map(evItem).join('');

  // Charts
  pushChart(chartDR,   kpi.dischargePct);
  pushChart(chartWait, kpi.avgWait);
}

/* ─── restart ───────────────────────────────────────────────────────── */
async function restart(scenario) {
  document.querySelectorAll('.sc-btn').forEach(b=>b.classList.remove('active'));
  const btn = document.querySelector(`.sc-btn.${scenario}`);
  if(btn) btn.classList.add('active');
  $('scenario-badge').textContent = 'Restarting…';
  await fetch('/api/restart',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({scenario})
  });
}

/* ─── poll loop ─────────────────────────────────────────────────────── */
async function poll() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();
    update(data);
  } catch(e) { console.warn('Poll error',e); }
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexCare OS — Hospital Intelligence Dashboard")
    parser.add_argument("--scenario", type=str, default="routine",
                        choices=["routine","surge","pandemic","critical"])
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     NexCare OS — Hospital Intelligence Dashboard         ║")
    print("║           Powered by ACPL Engine  · Shashank Dev        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Scenario : {args.scenario.upper():<45} ║")
    print(f"║  Dashboard: http://{args.host}:{args.port:<37} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  → Open http://{args.host}:{args.port} in your browser\n")

    cfg     = _build_cfg(args.scenario)
    _runner = SimulationRunner(cfg)

    import logging as _logging
    _logging.getLogger('werkzeug').setLevel(_logging.WARNING)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)