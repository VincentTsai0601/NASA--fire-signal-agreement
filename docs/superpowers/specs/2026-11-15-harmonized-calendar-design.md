# Calibrated burning calendar design

Status: user approved the recommended all-year calibrated calendar approach.

## Goal and scope

Connect the existing MODIS-to-VIIRS calibration to the weekly calendar without
changing the study box, source archives, or published held-out benchmark. Retain
the existing analyst and evidence views. This is a regional retrospective
prototype, not an operational early-warning or distinct-fire-counting system.

## Approaches

Recommended: apply the existing calibration trained on 2023–2024 to all 155
supported weeks. Show MODIS-derived VIIRS-equivalent estimates and observed VIIRS
counts as separate selectable calendar layers. Explicitly identify 2023–2024 as
training-period fitted estimates and 2025 as held-out estimates. This provides a
consistent reference scale without pretending training fits are independent
validation or combining overlapping detections by addition.

Smaller alternative: display only the existing held-out estimates for 2025;
earlier years retain an observations-only calendar. This avoids new fitted outputs
but does not provide calibrated values across the full supported record.

The user approved the recommended approach.

## Recommended implementation

- Fit the existing linear and XGBoost models on the same 104 training weeks and
  preserve their parameters and nonnegative clipping. VIIRS observations are
  training targets only; inference uses Terra and Aqua counts only.
- Persist all-week calibrated estimates separately from the existing weekly
  observations and 51-week held-out predictions. Include model, reference sensor,
  training dates, interpretation, and training/held-out status in provenance and
  exported data. Default to the existing XGBoost demonstration model when present;
  expose linear estimates with explicit labels and no silent substitution.
- Let users choose observed VIIRS, calibrated MODIS, and the model. Display a
  sequential luminance color ramp with numeric labels, a legend, and consistent
  scaling across years for each selected layer/model. Leave excluded weeks blank.
- Keep raw observations alongside the estimates, downloadable harmonized CSV,
  and the existing held-out MAE. Never describe fitted-period error as validation.
- Explain that calibrated values are expected VIIRS-equivalent hotspot counts;
  they can be fractional or nonzero even for zero MODIS records. They are not
  unique fires, burned area, forecasts, or a correction for unknown coverage.
- Verify identical 2025 predictions, no target leakage, complete calendar joins,
  missing-model behavior, metadata/CSV provenance, year selection, and the actual
  rendered calibrated calendar. Rebuild and verify the dashboard Docker image.

## Docker completion

The host RTX 3090 works outside the sandbox. NVIDIA Container Toolkit is missing,
despite an existing Docker runtime entry. Installing the toolkit, configuring the
runtime, and restarting Docker requires user-side sudo authentication. After that,
verify a CUDA container, start both Compose services, run the existing analyst
smoke benchmark, and record actual results. A successful dashboard test alone
does not establish that the full system works.

Completion evidence: the user installed/configured the toolkit; GPU access now
works. Both services are healthy after a one-command restart, data persisted,
the live analyst benchmark passed 5/5, and three live dashboard Ask-button checks
passed in container AppTest. The approved calendar is implemented and verified;
102 application tests pass. This note supersedes the initial blocker above.
