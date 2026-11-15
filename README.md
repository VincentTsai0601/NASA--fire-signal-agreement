# Fire Signal Agreement

Compare weekly Terra/Aqua MODIS and SNPP VIIRS satellite hotspot detections within 37–42°N, 125–119°W. Includes a calendar, held-out calibration charts, a raw evidence panel, and an optional bounded vLLM analyst.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py --modis fire_archive_M-C61_848110.csv --viirs fire_archive_SV-C2_848111.csv
streamlit run app.py
```

Explicit paths allow renamed archives. Standard archives are required; NRT versions and unexpected satellite IDs are rejected. SQLite defaults to `data/fire.sqlite`; use `--output` and `FIRE_DB` to change it. The app works without a model server. Install XGBoost to enable its comparison, then rebuild; missing dependencies appear as unavailable, never fabricated scores.

## Verified results on supplied archives

| Input | Raw rows | Actual date range | Study-box rows (all source dates) |
|---|---:|---|---:|
| MODIS | 420,160 | 2023-01-01–2026-08-31 | 18,380 |
| SNPP VIIRS | 1,641,209 | 2023-01-01–2026-07-31 | 63,990 |

Only 2023–2025 complete weeks enter evaluation: 104 training weeks and 51 held-out weeks. Source hashes, included row counts, full model parameters, error sign conventions and package versions are in `data/build-report.json`.

| Model | 2025 MAE | Signed mean error (estimate − observed) |
|---|---:|---:|
| Training median | 184.2941 | -82.8824 |
| Linear regression | 163.9575 | 36.9573 |
| XGBoost | 80.8117 | -1.9552 |

XGBoost 3.4.1 was executed against the original archives. On the 51 held-out 2025 weeks, its MAE is 80.8117 versus 163.9575 for linear regression (about 50.7% lower). This comparison concerns weekly satellite hotspot detections within the study box.

Direct raw-CSV reconciliation for 2025-08-04–2025-08-10: Terra **41**, Aqua **22**, VIIRS **318**, matching SQLite. Largest held-out absolute linear error: week starting **2025-07-07**, observed VIIRS **1,631**, estimate **760.7201**, observed-minus-estimated gap **870.2799**.

## End-to-end data verification

```bash
python verify_project.py
```

This reads the original archives without modifying them, verifies their recorded SHA-256 hashes, checks the complete weekly calendar, reconciles the default week (2025-08-04–2025-08-10) directly against raw CSV rows, matches held-out observations with weekly counts, and recomputes MAE, signed mean error and signed gaps from saved predictions. Use `--db`, `--modis`, `--viirs`, `--week` and `--output` for other paths or an included Monday.

The command writes `data/verification.json` and exits nonzero if any offline integrity check fails. It distinguishes `offline_passed` from `full_plan_complete`: successful data checks do not imply live model, browser, recording or challenge-brief verification. XGBoost execution is reported separately. The current report passes offline integrity checks and correctly records the full plan as incomplete.

## Optional local analyst

Serve a locally available instruction model with vLLM's OpenAI-compatible API, then configure its exact served ID:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=your-served-model-id
python smoke_test.py
streamlit run app.py
```

The client checks `/models` and uses JSON-schema-constrained responses from `/chat/completions`. Native tool calls are not required. Qwen first selects one typed task: largest model error, MAE comparison, weekly counts, documentation explanation, or unsupported scope. Python validates and executes the task using bounded read-only SQLite queries or TF-IDF retrieval. Numeric results and citations are rendered directly from tables, with no model synthesis step. For explanations, one retrieval supplies source passages to one constrained selection of received document IDs. The application displays the corresponding curated paraphrases and source links; model-written methodological prose is ignored. No free-form tool loop runs in the live vLLM path. The model never receives the prediction table to calculate errors itself. The legacy bounded loop remains supported for clients without task routing.

Supported numeric operations are rows, totals, largest absolute error, and MAE comparison. Scope refusals use fixed application text. Unknown tasks and fields are rejected, and dates must use YYYY-MM-DD within the supported window. Up to one evidence operation runs per question. Live routing still depends on the model classifying the question correctly; smoke tests must pass before claiming validated analyst behavior. Raw CSV rows are never sent to the endpoint.

Validated demo model: **Qwen/Qwen2.5-1.5B-Instruct**. The user-executed live smoke run at **2026-11-15T08:06:39Z** passed all three demo cases. Numeric answers were independently reconciled against SQLite, and the sensor explanation matches the selected curated NASA passages. Evidence is saved in `data/live-smoke.json` and `data/live-review.json`. An earlier automated pass contained a false generated explanation; methodological output now displays curated source notes. Validation covers these three examples. Later user-side scope probes incorrectly returned study-box totals for unsupported requests. Application checks now reject spelled-out US state names and explicit years outside 2023–2025 before model or data access. Quantity phrases such as “2026 hotspot detections” are distinguished from a requested year. These conservative checks are not a general geography parser; other scope interpretation remains model-dependent. See `data/scope-verification.json` for offline rejection checks. The user then confirmed correct refusals for Texas and 2026 in the dashboard. The expanded automated benchmark passed on the user’s machine at 2026-11-15T08:58:13Z: five-case benchmark **5/5**, three original demo cases passed, and four application scope checks passed. The saved baseline is `data/qwen-1.5b-benchmark.json`. Charts and evidence remain usable when the model server is unavailable.

## Verification and demo

```bash
python -m pytest -q
python smoke_test.py  # optional; requires a running model; exits nonzero on failure
```

The suite uses synthetic archives and temporary SQLite databases. It covers ingestion rejection, content-based archive identification, week and geographic boundaries, train/test target isolation, exact SQLite answers, normalized dates, geography rejection, retrieval, tool and citation restrictions, grounded numeric rendering, live smoke failure detection, Streamlit loading/year filtering, missing data and analyst-unavailable states. Latest verification: **91 passed**; installed Streamlit/Altair dependencies emit deprecation warnings.

Verified reference answers to all three demo questions are available in the dashboard and [docs/demo-results.md](docs/demo-results.md), without a language model. The live smoke test saves `data/live-smoke.json`, checks the first two answers against computed results, checks source linkage for the third, checks four unsupported-scope requests without model dispatch, and exits nonzero on a failed case. Methodological prose still needs human accuracy review. Each case includes `model_replies`, the model’s completion JSON in order, to diagnose rejected tool arguments or repeated calls. Prompts are not recorded.

The report also includes a versioned five-case `benchmark_cases` group: the two numeric paraphrases, the hotspot-versus-fire definition, and the Texas/2026 refusal checks. Refusal results are reused from `scope_checks`. Numeric expectations are computed from SQLite, and the definition must include the curated MODIS pixel passage and its source link. Any failed benchmark case fails the overall run. `benchmark_summary` reports passed/total; `model` identifies the served model. Scope checks measure application behavior, not model scope reasoning.

For a named baseline report, keep Qwen running and use:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
python smoke_test.py --output data/qwen-1.5b-benchmark.json
```

A successful run reports `"passed": true` and `"benchmark_summary": {"passed": 5, "total": 5}`. Reports contain model replies and computed answers, but no prompts or raw archive rows. This is a small benchmark, not a general accuracy estimate. A 2026 question is a refusal probe; no 2026 data is ingested.

Live model comparison: Qwen2.5-1.5B scored **5/5** and Qwen2.5-7B scored **4/5** on benchmark version 1. The 7B model falsely refused the hotspot-definition question. Keep 1.5B as the working demo baseline; this small benchmark does not establish general model superiority. See [docs/model-comparison.md](docs/model-comparison.md) for results and limitations.

Detailed launch commands, narration, and recording checks: [docs/demo-walkthrough.md](docs/demo-walkthrough.md).

Walkthrough script:

1. Start with 2025 in the weekly calendar. Explain that Monday cells represent whole UTC weeks, and zeros describe supplied records only.
2. Show weekly sensor counts and the published held-out model metrics.
3. Inspect the week of 2025-07-07 in raw predictions and its signed linear gap.
4. With a verified live server, ask “Which 2025 week had the largest calibration error?”, “Did XGBoost beat linear regression?”, and “Why can the sensors report different counts?”. Compare numeric answers with the evidence tables and methodological answers with NASA links.

Live model smoke validation is included, and the user confirmed all three demo answers in the running dashboard against the reference results. No walkthrough recording is included. The challenge brief was not supplied; the local plan and official NASA notes are indexed instead. See `docs/sources.md` for provenance and limitations.

Environment note: earlier server launches were blocked by sandbox socket permissions (`Operation not permitted`). The user has since supplied a screenshot of the dashboard running at `localhost:8501`. Automated checks use Streamlit's in-process AppTest; an independent browser walkthrough and live model validation remain pending. Blank calendar cells for adjacent months no longer inherit observation colors.

## GitHub source distribution

This repository includes source, tests, documentation and small benchmark reports. Raw FIRMS CSVs, generated SQLite databases, local credential/configuration directories, Python caches and recordings are excluded by `.gitignore`. After cloning, install requirements, supply the two standard archive CSVs locally and run the pipeline command above before launching Streamlit. Benchmark JSON reports alone do not provide the database needed by the dashboard. Publishing this repository does not host the Streamlit or vLLM servers.
