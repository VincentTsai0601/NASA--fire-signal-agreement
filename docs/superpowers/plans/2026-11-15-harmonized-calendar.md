# Harmonized calendar implementation plan

Goal: Deliver the approved all-year MODIS-derived VIIRS-equivalent calendar.
Spec: docs/superpowers/specs/2026-11-15-harmonized-calendar-design.md
Architecture: Shared training-only calibration in pipeline.py; separate harmonized
SQLite table with provenance; Streamlit layer/model selection, shared-year color
scale and downloadable evidence. Retain original observations and held-out scores.

Existing workspace policy: work in the existing checkout, preserving the prior
session's in-place editing decision. Git is now functional; no commit or push is requested.
Execute inline as authorized by the user's request to finish the remaining items.

- [x] Test all-year calibration, unchanged held-out estimates, target isolation,
  provenance, missing optional XGBoost, and persistence; implement shared fitting.
- [x] Test calendar layer selection, model labels, year filtering, stable color
  scale and export; implement dashboard while preserving analyst behavior.
- [x] Test integrity verification against corrupt calibration/provenance; implement
  verifier and rebuild supplied archives without changing the published benchmark.
- [x] Run full suite, real-data dashboard checks, Docker build and HTTP health.
  Request independent review; fix material issues and document actual results.
- [x] Verify full GPU deployment if host toolkit becomes available; otherwise
  record exact blocking prerequisite, keeping calendar delivery independent.

Review focus: no training-target leakage from 2025, no fabricated optional-model
outputs, explicit fitted versus held-out labels, absent legacy table, excluded
weeks, float-valued estimates, and independent colors/text contrast.

Verification so far: 102 tests passed; raw-archive rebuild preserved all prior
held-out predictions and metrics exactly. 155 calibrated rows (104 fitted, 51 held
out), offline source/metrics/calibration integrity passed. Independent review found
no material issues. A subsequent label-contrast regression exposed a midrange
contrast ratio of 3.39:1; luminance-based black/white text fixed it (RED then GREEN).
Rebuilt dashboard image and real-data container rendering/HTTP health passed.
Local .env sets port18501 to preserve the existing host app on8501. Host GPU probe
now passes after user installed toolkit. vLLM image pulled and analyst initializing.

Completion: both services reached healthy status. Container live benchmark passed
5/5, demos3/3, scope checks4/4; saved in data/docker-smoke.json. The actual Ask
button matched all three live answers in container AppTest, saved separately in
data/docker-dashboard-check.json. Stopped both services and restarted them with
`docker compose up -d --wait`; both became healthy again, live Qwen connectivity
and HTTP health passed, and 155 harmonized/51 held-out rows persisted. System left
running at http://localhost:18501. Full independent browser operation and video
recording are not claimed. No findings were deferred. No source commit/push made.
