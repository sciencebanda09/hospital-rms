# Hospital Resource Management System
### Powered by ACPL v2 — an original algorithm by Shashank Dev

A real-time hospital simulation that uses a novel reinforcement learning algorithm — **ACPL (Adaptive Consequence-Penalised Learning)** — to dynamically allocate resources (beds, staff, equipment) across patients with evolving acuity.

---

## What is ACPL?

> **ACPL is an original algorithm developed by me.**

Most resource allocation systems treat all bad outcomes equally. ACPL doesn't — it distinguishes between *expected* bad outcomes and *consequence-amplified* ones (e.g. assigning a wrong resource type to a deteriorating patient). The core idea: penalise actions proportional to their downstream consequence, not just their immediate reward signal.

Key properties of ACPL v2:
- Learns *online* during the simulation — no pre-training required
- Uses a **dual-stream architecture** to separate state value from action advantage
- Uses **n-step returns** to account for the delayed nature of clinical outcomes
- Maintains a **dynamic consequence penalty** that adapts per state via a learned Lagrange multiplier
- Handles **uncertainty** in consequence prediction explicitly

The algorithm is implemented from scratch in NumPy — no PyTorch, no TensorFlow.

---

## System Overview

The simulation models a live hospital with:

| Component | Description |
|---|---|
| **Patients** | 5 categories (Emergency, ICU, General, Surgical, Maternity), dynamic acuity, comorbidity risk, deterioration/improvement rates |
| **Resources** | General beds, ICU beds, Emergency bays, Surgical suites, Staff teams |
| **ACPL Advisor** | Online RL agent that penalises risky assignments before they happen |
| **Demand Forecaster** | EMA-based wave size predictor; flags surges proactively |
| **SLA Tracker** | NHS-style wait-time limits per category with breach logging |
| **Readmission Queue** | 5% of discharged patients return with elevated acuity |
| **Staff Fatigue Model** | Load-dependent burnout that reduces treatment effectiveness |
| **Kalman Filter** | Per-patient acuity trend estimation for predictive assignment |
| **Live Dashboard** | Matplotlib animation — ward map, KPIs, acuity histogram, ACPL diagnostics |

---

## Requirements

```
Python >= 3.9
numpy
scipy
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

---

## Usage

```bash
# Default run
python hospital_rms_v2.py

# Scenarios
python hospital_rms_v2.py --scenario surge        # Mass-casualty surge
python hospital_rms_v2.py --scenario pandemic     # High load, reduced staff
python hospital_rms_v2.py --scenario nightshift   # Night-shift capacity
python hospital_rms_v2.py --scenario critical     # ICU pressure
python hospital_rms_v2.py --scenario routine      # Calm operations

# Headless batch runs (no GUI)
python hospital_rms_v2.py --batch 5 --no-csv

# Disable v2 features
python hospital_rms_v2.py --no-fatigue --no-readmission

# Light theme, wider window
python hospital_rms_v2.py --theme light --width 30
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--seed` | 2025 | RNG seed |
| `--frames` | 2000 | Simulation steps |
| `--scenario` | — | One of: surge, pandemic, routine, critical, nightshift |
| `--batch N` | 1 | Run N seeds headlessly and export summary CSV |
| `--num-resources` | 10 | Number of hospital resources |
| `--max-patients` | 16 | Max concurrent patients |
| `--adm-period` | 90 | Ticks between patient waves |
| `--theme` | dark | `dark` or `light` |
| `--no-csv` | — | Disable CSV event export |
| `--no-fatigue` | — | Disable staff fatigue model |
| `--no-readmission` | — | Disable readmission loop |
| `--output-dir` | hrms_runs | Directory for logs, CSVs, reports |

---

## Output Files

Each run produces (inside `hrms_runs/`):

- `hrms_events_<id>.csv` — per-event log (discharge/mortality, acuity, wait time, ACPL penalty)
- `session_report_<id>.json` — summary with per-category stats and SLA breach rates
- `batch_summary_<id>.csv` — (batch mode) per-seed metrics with 95% CI

---

## Live Dashboard

The real-time dashboard shows:

- **Ward map** — coloured patient positions by category; red stars = critical acuity; heatmap = mortality density
- **Bed occupancy %** — with 80% and 95% warning lines
- **Discharge rate %** — rolling clinical throughput
- **Active patient count**
- **SLA breach rate %** — per-tick wait-time violations
- **Resource utilisation + fatigue** — horizontal bars per resource
- **ACPL v2 diagnostics** — live: update count, buffer size, policy loss, mean λ, mean consequence
- **Patient acuity distribution** — histogram across all active patients

---

## Architecture Notes

```
Simulation.tick()
  ├── Patient.step()          — acuity evolution (det/imp rates + comorbidity + noise)
  ├── AcuityFilter.predict()  — Kalman filter per patient
  ├── Resource.step()         — fatigue update (STAFF_TEAM only)
  ├── ResourceAllocator.assign()
  │     ├── build cost matrix (urgency + skill + wait + velocity + load + fatigue + ACPL penalty)
  │     └── scipy.linear_sum_assignment → optimal assignment
  ├── SLATracker.check()      — every 10 ticks
  ├── ReadmissionQueue.flush() — readmissions due this tick
  └── outcome resolution (discharge / mortality → ACPL.feedback())
```

---

## Copyright

Copyright © 2025 Shashank Dev. All rights reserved.

The **ACPL algorithm** (including its consequence-penalised learning formulation, dual-stream architecture, and n-step consequence aggregation) is an original contribution by the author. Reproduction of the ACPL core for commercial or research purposes without explicit written permission is prohibited.

The simulation framework (patient model, ward layout, visualisation) is open for educational use with attribution.
