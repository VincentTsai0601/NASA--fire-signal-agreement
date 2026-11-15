# Implementation record

## Docker deployment and author credit

Added Dockerfile, a build-context allowlist, and Compose services for Streamlit
and Qwen2.5-1.5B via vLLM. SQLite is mounted read-only; the Hugging Face model cache
uses a persistent volume. Dashboard startup waits for analyst readiness. Deployment,
database rebuild, smoke-test and image-export instructions are in `docs/docker.md`.
README now credits Min-Sheng Tsai as requested.

Verified: Compose configuration and internal API/model/mount checks, dashboard image
build, 91 passing host tests, and real-data container AppTest (155 weekly rows;
2025 totals 1,881 / 2,104 / 16,620). A temporary dashboard-only deployment passed its
HTTP health check. Independent deployment review found no material issues.
Full-system execution remains unverified: Docker GPU probes fail with
`could not select device driver "" with capabilities: [[gpu]]` even though an
NVIDIA runtime appears in Docker info. Host GPU/container runtime setup needs to
work before starting the analyst and running the container smoke benchmark.

Plan: `2026-11-15-fire-signal-agreement.md`

- Pipeline: implemented and rebuilt on both original standard archives; 155 weeks, 104 train / 51 test. One weekly count reconciled directly against raw rows.
- Calibration: median and linear metrics verified. XGBoost implementation present, runtime unavailable because dependency download was blocked. No substitute model used.
- Dashboard: calendar, weekly comparison, held-out metrics/gaps, year filtering, raw evidence and provenance implemented. In-process Streamlit test passes. HTTP launch blocked by sandbox socket policy.
- Analyst: read-only bounded SQLite tools, local TF-IDF retrieval, JSON tool allowlist, three-tool budget, endpoint/model validation, stub tests implemented. Live server and response quality not verified because socket access is blocked. No model has been selected as passing the examples.
- Documentation: README, official NASA paraphrases, source links, hashes, metrics and demo script included. Separate challenge brief and walkthrough recording were unavailable; neither is claimed as completed.

Ruling: use the supplied plan as implementation authority; do not re-design an already specified project. No functional Git repository exists in the mounted `.git` directory, so changes are written directly in the authorized workspace without commits or worktrees.
Ruling: mark absent XGBoost scores unavailable so pipeline and charts remain usable without inventing comparison results.
Ruling: count all supplied archive detection types and confidence levels, since the plan specifies no exclusion; document this limitation.

Verification: `python -m pytest -q` passed 46 tests (dependency deprecation warnings). Compile check passed. Live socket operations denied by the environment. See `data/build-report.json` for source hashes and real results.

## Continued implementation and review

- Fixed a third-tool off-by-one error: three tools may run, followed by one final response; a fourth tool is rejected without execution.
- Added final-answer citation validation and structured numeric claims. Numeric prose from the model is ignored; allowlisted operations render values directly from received evidence. Empty evidence is rejected. Scope refusals are fixed application text.
- Fixed smoke-test false success: endpoint/model or task-output failures produce a failed JSON report and nonzero process status. Numeric outputs are checked against computed evidence; methodological source linkage is checked, with human factual review still required.
- Added verified demo reference answers in the dashboard and `docs/demo-results.md`; these work without a live model.
- Processing-version validation now rejects unknown/missing versions and identifies standard archives by content, including misleading filenames.
- Provenance now includes model parameters, software versions, sign conventions and actually included row counts. Rebuilt outputs from both original archives.
- Normalized accepted ISO dates before SQL binding to avoid empty results for compact date inputs. Dashboard tests now use isolated synthetic data and cover missing database and unavailable analyst behavior.
- Independent review verified both Important findings were resolved. Final suite: 46 passed; installed dependency deprecation warnings remain. No local XGBoost installation or cached wheel was found.

Remaining environment/input work: install and execute XGBoost; configure and validate a live instruction model; launch/render the HTTP dashboard and record the walkthrough outside the socket-restricted sandbox; add the separate challenge brief when supplied.

## Reproducible end-to-end verification

Added `verify_project.py` and six tests covering real source reconciliation, tampered source detection, corrupted weekly counts, corrupted scores, missing database handling and excluded-week handling. The command is read-only with respect to sources and SQLite; it writes only its JSON report.

Real-archive result: `offline_passed: true`, `full_plan_complete: false`. Verified both recorded source hashes; 104 training / 51 held-out complete weeks; Terra 41, Aqua 22 and SNPP VIIRS 318 for 2025-08-04–2025-08-10; and published MAE, mean errors, gap signs and held-out observations against stored predictions. XGBoost remains unavailable and external integrations remain explicitly not run.

Latest suite: `python -m pytest -q` — 52 passed, dependency deprecation warnings only. User input requested for a live endpoint/model ID and challenge-brief URL; none supplied yet.

## Continuation: XGBoost execution and calendar correction

XGBoost 3.4.1 is now installed in the active Python environment. Rebuilt `data/fire.sqlite` and `data/build-report.json` from the original archives. XGBoost held-out MAE 80.8117322734758; signed mean error −1.9551792518765319. Regenerated demo answers, including its largest absolute error on 2025-07-14 (467.93 detections). `python verify_project.py` exits 0 with offline integrity and XGBoost execution passed.

Calendar root cause: the lookup applied a neighboring month's Monday count before hiding that date's label. Restricting the lookup to dates in the displayed month keeps padding cells uncolored. Added an AppTest regression checking a counted Monday remains colored while its duplicate in February padding does not. Latest suite: `python -m pytest -q` — 53 passed, dependency deprecation warnings only.

The user's screenshot establishes that the dashboard has launched on their machine; it does not establish full browser-flow or live analyst verification. Updated the original plan's checkboxes for verified tasks; kept mixed tasks with missing external steps open.

Ruling: continue editing the existing non-Git workspace, preserving the earlier workspace decision. No commits or worktree operations are possible here; file changes and this record remain reviewable directly.

Remaining: live vLLM base URL and served model ID, separate challenge brief, full browser walkthrough and recording. Previous notes about missing XGBoost are historical and superseded by this entry.

Final continuation review: independent read-only reviewer found no Important issues. Confirmed calendar padding behavior, four passing dashboard tests, and consistency of SQLite metrics with build report, verification report, README and demo answers. Regression also verified RED with the guard removed, then GREEN after restoring it; full suite 53 passed. Real-data AppTest loaded successfully and matched screenshot totals (Terra 1,881; Aqua 2,104; VIIRS 16,620). Compile check passed.

## Live Qwen validation: protocol failure diagnosis

User reached `/v1/models` with `Qwen/Qwen2.5-1.5B-Instruct`, then supplied a failed live smoke report: two unsupported tool/argument responses and a three-tool budget exhaustion. This verifies server reachability but does not qualify the model as passing. The rejected JSON was not retained, so the exact mismatch is still unknown. Direct server access from this agent fails with `Operation not permitted`.

Added per-case `model_replies` to the smoke report via a recording client. This preserves completion JSON in order, without recording prompts or changing tool validation/budgets. Two regression assertions failed before implementation and passed afterward; full suite 53 passed. Next step: user reruns `python smoke_test.py` with the existing Qwen server and shares the trace. Do not relax allowlists or claim a model fix without inspecting those responses.

## Qwen protocol correction

User trace identifies the mismatch: `get_model_metrics` received an invented `window` argument for both numeric questions, while the methodological question repeated searches beyond the tool budget. The previous `json_object` response format guaranteed JSON syntax only, leaving those protocol choices unconstrained.

Changed vLLM requests to `response_format.type=json_schema`, with closed tool argument objects and allowlisted response forms. After three recorded assistant tool requests, the response schema excludes all tool forms. Application validation, claim computation, citations and tool limits remain unchanged. Added explicit instructions for immediate use of supplied predictions, empty metric arguments, and synthesis after sufficient retrieval.

A new request-boundary regression failed with the old JSON-object format, then passed with schema constraints. It checks the exact user's invalid window argument and repeated fourth-tool shape against the actual transmitted schemas. Full suite: 54 passed, dependency deprecation warnings only. Live vLLM schema compilation and Qwen task accuracy still require a user-side smoke rerun; do not claim live success from stub tests.

## Qwen second live trace: repeated tools and invented citation IDs

The next user-side run accepted metric arguments but repeated metric/search calls. Numeric synthesis invented row-level citation IDs; one completion failed JSON parsing (the report cannot establish whether truncation caused it). The methodological answer passed automated source linkage, but its archive-latency explanation still requires factual review.

Updated response schemas per turn from application-added evidence: citation IDs are exact received IDs, method citations use document IDs, and data claim sources use received tables. Remove get_model_metrics and search_docs after their first successful round, while leaving bounded weekly window queries available. Data answer prose is constrained to an empty string because numeric text is computed by the application. Doubled the completion budget to 1600 tokens and bounded method prose to 2400 characters to reduce truncation exposure without claiming truncation was proven. Existing runtime allowlists/citation checks remain intact.

Regression test reproduces the invented citation ID and repeated tool shapes against transmitted schemas, failing before this change and passing afterward. Full suite: 55 passed; compile check passed. Live quality remains unverified until the user reruns the smoke test.

## Replace live free-form tool loop with structured task routing

Third user-side trace demonstrates semantic failure, beyond argument formatting: Qwen1.5B chose irrelevant metric/count tools, emitted timestamp strings as dates, and selected scope while writing numeric claims. Repeated schema tightening did not validate the model.

Ruling: use the plan's explicit structured-routing option for the live vLLM client. Route a question once into largest_error, compare_mae, counts, explain or unsupported. Python validates exact fields and executes one evidence operation, then computes numeric answers directly; it never sends prediction tables for the model to interpret. Method questions retrieve once and request a method-only answer with document citation IDs. Costs: smaller supported task set; model intent classification still needs live verification. The legacy loop remains for existing clients without routing support. No new model or fabricated pass result is introduced.

RecordingClient forwards routing capabilities and records route/explain stages separately. Tests cover computed largest error, unavailable comparison, bounded counts, rejected plan fields/models/dates, method evidence isolation, and smoke-test dispatch through the actual recorder. They failed before the new path was implemented. Additional HTTP-boundary checks verify route and method-only generation schemas. Live smoke report remains failed until a user-side rerun passes; this sandbox cannot reach the user's model endpoint.

Structured-routing verification: full suite 65 passed, dependency deprecation warnings only; compile check passed. Independent read-only review found no Important implementation issues, with 33 targeted tests passing. Model classification and live methodological prose still require a successful live smoke run and factual review.

## Manual review of first automated live pass

The user-side Qwen smoke report at 2026-11-15T08:03:08Z passes automated checks. Inspected the saved `data/live-smoke.json`: largest linear error and XGBoost comparison match the published data. The methodological text is factually wrong despite valid citation linkage: it reverses the resolution relationship when claiming MODIS can count each fire within a VIIRS pixel. NASA's VIIRS description supports improved response to small fires and mapping of large perimeters; hotspot records are pixel detections, not distinct-fire counts. Preserved the original report and recorded this review separately in `data/live-review.json`.

Ruling: render curated source passage text for methodological answers, selected by the model's validated document IDs; do not expose generated methodological prose. This narrows synthesis flexibility but prevents the observed unsupported claim without pretending a citation proves prose accuracy. Live selection schema now requires answer="". Both numeric routing and computations remain unchanged. Regression reproduces the user's false sentence and verifies it is absent while the selected curated passage appears. Test updated to verify the method-only request disallows generated prose. Full suite 66 passed, dependency deprecation warnings only. Corrected live selection still needs a user-side smoke rerun; endpoint reachability and numeric validation are established.

## Live demo validation accepted

Saved user-executed smoke run at 2026-11-15T08:06:39.620734+00:00 passes all three cases with Qwen/Qwen2.5-1.5B-Instruct. Independently checked both numeric answer strings using render_claim against the current SQLite tables. Checked that the final methodological answer contains the exact curated VIIRS resolution and MODIS pixel-definition passages and their source URLs. Updated `data/live-review.json` to record accepted manual review, preserving the earlier false-prose finding as history. Checked the plan's live endpoint/model integration task and updated README. No code changed in this acceptance step, so no suite rerun was needed.

Remaining: supplied challenge brief, the three questions in the browser dashboard, full browser walkthrough and recording. No general open-ended routing quality claim is made.

## Dashboard walkthrough preparation

No desktop/browser or screen-recording tool is attached to this session; the agent cannot operate or record the user's Chrome window. Used the cloud-desktop fallback workflow. Streamlit AppTest replayed the exact saved successful Qwen routing and citation-selection responses through the real dashboard Ask button. All three rendered answers match the saved live results. Verified 2025 totals and switching 2024→2025. Report: `data/dashboard-replay.json`, explicitly tagged as replay without live network, browser or recording.

Prepared `docs/demo-walkthrough.md` with Qwen launch environment, expected values, narration and playback checks. Linked it from README. Browser verification and recording remain open and require the user to run the supplied steps on their local desktop. No application code changed.

## User-confirmed dashboard demo checks

User supplied the actual Ask the analyst outputs for all three demo questions in the running dashboard. All match the saved reference results: week 2025-07-07, linear absolute error 870.28; XGBoost MAE 80.81 versus linear 163.96; curated sensor-resolution and pixel-definition notes. Checked the plan's three-question dashboard task. Evidence origin is user-reported browser outputs; no agent-operated browser or video capture is claimed. Remaining external items: challenge brief and walkthrough recording.

## Scope robustness failure and application guard

User reported success on two numeric paraphrases and the hotspot-versus-fire question. Subsequent outputs, supplied in response to the Texas and 2026 scope probes, returned 2025 study-box totals instead of refusals; an additional output repeated the 2025 MAE comparison. The original three-case demo pass is narrower than robustness validation. Scope cannot rely only on Qwen's task choice.

Added `explicit_scope_violation` before either routing path or evidence lookup. It conservatively rejects spelled-out US states (the box is not a whole-state dataset) and explicit calendar years outside 2023–2025. A quantity phrase such as “2026 hotspot detections” is not treated as a year. This is explicitly not a universal geography parser; other locations and ambiguous wording still depend on model interpretation. State-name discussion may be conservatively refused even when methodological.

Sixteen regression cases reproduce state/year failures across routed and legacy clients, and assert no model/data call is reached. They failed before implementation. Added quantity-disambiguation coverage and a smoke report regression. Expanded smoke tests include four scope probes; these report separately as `scope_checks` and contribute to overall passed status. All 84 tests pass, dependency deprecation warnings only; compile check passed. Independently invoked all four probes with objects lacking any model/evidence methods, confirming early refusals; evidence in `data/scope-verification.json`. Revised live smoke and user-side UI check remain pending; earlier live results are preserved.

Independent scope review found an Important ambiguity in the first quantity exception: “Compare 2026 fire detections with 2025” and “January 2026 hotspot detections” could bypass the year check. Added both regressions for routed and legacy clients; all four failed, then passed after restricting quantity exceptions to explicit quantity constructions such as “Do … mean …”. General detection nouns no longer imply quantities. Final suite: 88 passed, dependency deprecation warnings only.

## Five-case automated analyst benchmark

User confirmed the two scope refusals after the application guard; all five manual robustness probes now match expected behavior. User authorized automating those exact checks. Extended the existing smoke script, preserving original demo cases and scope checks. Added `benchmark_cases` for two numeric paraphrases, the 100-hotspots definition question and the Texas/2026 refusals (reuse existing refusal results). Added model ID, benchmark version and passed/total summary. Expectations use current SQLite computations and the exact curated MODIS pixel definition plus source URL. Failures contribute to overall exit status. Scope checks are identified as application checks, not LLM reasoning scores.

Three new tests failed before implementation: missing benchmark, wrong paraphrase not failing the report, and an irrelevant valid citation falsely passing. They now pass. Existing fake clients were extended to handle the additional prompts. Full suite: 91 passed, dependency deprecation warnings only; compile check passed. No live model run is claimed; user can write the named baseline with `python smoke_test.py --output data/qwen-1.5b-benchmark.json`. Original live evidence is preserved. Dataset and calibration remain restricted to 2023–2025; out-of-range years occur only as refusal probes.

## Qwen baseline benchmark accepted

User executed `python smoke_test.py --output data/qwen-1.5b-benchmark.json` at 2026-11-15T08:58:13.348747+00:00. Inspected the saved report: model Qwen/Qwen2.5-1.5B-Instruct; benchmark version 1; benchmark 5/5; all three original demo cases and all four scope checks pass. Scope checks have no model replies, confirming application-side refusals. Baseline file preserved for future comparison. This establishes the fixed example benchmark only, not general language-model accuracy. No code changed during result acceptance.

## Second-model comparison

User launched Qwen/Qwen2.5-7B-Instruct on RTX3090. The first attempt could not connect during startup; the completed second run at 2026-11-15T09:01:53Z passed demo 3/3 and application scope 4/4, but benchmark 4/5. It incorrectly classified the supported hotspot-versus-fire question as unsupported. Inspected both saved reports: benchmark version and question list match; 1.5B scores 5/5. Preserved the failed result without tuning prompts to make the larger model win. Added `docs/model-comparison.md` and linked README. Recommend the 1.5B working baseline for this workflow; no general model ranking, latency claim, or model-only scope score is implied. Documentation-only acceptance; no code changes or test rerun needed.

## GitHub preparation

User asked about uploading the working prototype. Confirmed raw archive CSVs and generated SQLite were already ignored. Extended `.gitignore` to exclude .env files, local .aws/.codex/.agents directories, SQLite sidecars and screen recordings. Verified patterns in an isolated temporary Git repository: raw data/config/caches/recording paths ignored; Python source, tests, docs and benchmark JSON retained. Added README distribution instructions explaining local source requirements and that GitHub publication does not host the application. No remote repository was created and no push was performed; destination URL is not supplied.
