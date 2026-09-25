# GEMINI.md

Guidance for Gemini Code working in this repository.

## What this project is

A FastAPI service that extracts 15 identity fields from UAE National IDs, passports and residence
visas, and in parallel locates, crops, de-skews and rotates each physical document in the upload.
Clients call it over HTTP (Postman for manual testing - see `POSTMAN_GUIDE.md`); there is no UI.

Model: **`gemini-2.5-flash`** ($2.00 / 1M input, $10.00 / 1M output).

---

## Token & Behavior Rules

- Do NOT run tests automatically after code changes. Wait for me to explicitly say "run tests" or
  "test this".
- Do NOT explore/scan the whole codebase for simple tasks — only touch files directly relevant to
  the request.
- Skip verbose explanations, summaries, or recaps unless I ask for them. Just make the change.
- Don't re-read files you've already read in this session unless they changed.
- For simple/small tasks, skip deep reasoning — act directly.
- Ask before running expensive operations (full test suites, builds, installs).

---

## Cost discipline — read this first

**Every live request spends the user's money.** Each document in an `/ocr` upload makes up to
**four** model calls — extraction, box detection, orientation, and self-verification — and costs
about **$0.021**, so a batch costs that much per document; `DETECTION_ENABLED`/`X-Detection` and
`VALIDATION_ENABLED`/`X-Validation` each drop one pipeline (two calls, or one) when switched off. A
batch of two or more documents makes **one further call** (cross-verification) for the whole
request, not per document. A National ID/passport/visa uploaded as two separate incomplete files
adds **two more calls** for that pair specifically: one pairing confirmation
(`verify_document_pair`), and one extra extraction call on the merged file - on top of the two
extraction calls already spent on each original file individually. A single code-execution
experiment measured **$0.41**. These add up fast during debugging.

### Do not make live API calls to test

Do not run the pipeline against the real API to check that a change works. There is no situation
where "let me just run it once to see" is the default. Verify with the tools below instead, which
cost nothing:

| Instead of | Use |
| --- | --- |
| Running a real extraction | `pytest` — 160+ tests, ~5s, no network |
| Checking a service method works | `unittest.mock.patch.object(httpx, "post", ...)` or a mocked `GeminiService.client` |
| Checking prompt/schema shape | Read the constant and assert on it directly |
| Getting sample pipeline output | Read the existing `logs/` and `detections/` files — four days of real runs are already on disk |

Offline verification patterns that already work in this repo:

```bash
# Full suite
./.venv/Scripts/python.exe -m pytest -q

# Exercise GeminiService with a fake client — no network
# svc.client = MagicMock(); svc.client.messages.create = AsyncMock(return_value=fake_response)
```

### When a live call is genuinely necessary

Sometimes only the real API can answer a question (does this model accept this parameter; is this
answer stable across runs). In that case:

1. **Ask first.** State what you want to run, why offline testing cannot answer it, and the
   estimated cost.
2. Use the **smallest** input that settles the question — an existing crop from `detections/`,
   not a fresh full-page upload; `max_tokens=4` for a capability probe.
3. Run the **minimum number of iterations** that gives an answer. Three runs establish stability;
   ten do not add much.
4. **Report the actual spend** afterwards.

Never loop the live API to gather statistics without explicit approval.

---

## Reducing token usage generally

Applies to Gemini Code's own work in this repo, not just the OCR service.

**Read narrowly.**
- `app/services/detection.py` is ~800 lines and `app/services/gemini.py` ~800. Do not `cat` them. Use
  `grep -n` to locate, then `sed -n 'START,ENDp'` to read only the region that matters.
- Never re-read a file you just edited. `Edit` fails loudly if it did not apply.
- Read a log file's *last* entries (`tail`, or slice in Python), not the whole file. `logs/` holds
  hundreds of JSON lines across rotated files.

**Compute, don't dump.**
- When analysing logs, write a short Python script that prints an aggregate (counts, means, a
  table). Do not print raw log lines and reason over them in context.
- Same for JSON: print the two fields you need, not the whole document.

**Prefer surgical edits.**
- `Edit` on the exact block beats rewriting a file with `Write`.
- For multi-point edits, a single Python patch script with `str.replace` and `assert old in text`
  is cheaper and safer than several round trips.

**Reuse what is already on disk.**
- `logs/rotation.log`, `logs/hits.log`, `logs/ocr-api.log`, `logs/token-breakdown.log` and
  `detections/*.json` contain real pipeline output going back days. Almost any question about
  behaviour, accuracy or cost can be answered from them without running anything.

**Don't re-verify what is already verified.**
- Do not run the test suite unless asked (see Token & Behavior Rules above). When asked, run it
  once for the whole change set, not per edit.
- Do not re-run a check that passed unless something it depends on changed.

---

## Commands

```bash
# API
uvicorn app.main:app --reload

# Tests
./.venv/Scripts/python.exe -m pytest -q
```

The venv is `.venv/` (Windows layout: `.venv/Scripts/python.exe`). `ocr-dev-venv/` is a stale
second environment — ignore it.

---

## Architecture notes

Standard FastAPI layout - routes stay thin, the work lives in `app/services/`:

- `app/main.py` — `create_app()`: logging, `OcrSecurityMiddleware`, exception handlers, routers.
- `app/core/` — `config.py` (Settings, instantiated at import), `logging.py` (six log files, three
  timezones per line), `exceptions.py` (`ValidationStopped` + the handlers that give `/ocr` its
  errorInfo error shape), `middleware.py` (HTTPS + rate limit, exact path `/ocr` only).
- `app/api/routes/` — `health.py`, `ocr.py` (`POST /ocr`: request ID, `X-Detection`/`X-Validation`
  overrides, field-count checks, then `run_ocr_batch`), `leasing.py` (`POST /ocr/leasing` and
  `/ocr/leasing/upload`, both `response_model=LeasingOcrResponse`).
- `app/schemas/` — `ocr.py` (DATA_FIELDS, FRONT_BACK_TYPES, OcrResponse), `leasing.py`.
- `app/services/gemini.py` — every model call (extraction, box detection, orientation,
  self-/cross-verification, front/back pairing confirmation), prompts, schemas, usage logging.
  `get_gemini_service()` returns the one shared instance; tests monkeypatch methods on it.
- `app/services/detection.py` — everything OpenCV: page rendering, corner refinement, perspective
  crop, rotation, enhancement, crop/metadata output, `save_merged_document`.
- `app/services/ocr_pipeline.py` — the shared pipeline steps: `extract_and_type_check`,
  `resolve_incomplete_pairs` (merges exactly one complementary front/back pair per type after
  `verify_document_pair` confirms it, via `_build_merged_pdf` + re-extraction),
  `run_self_verification`, `merge_extracted_data`, and `run_ocr_batch` (`/ocr`'s whole flow).
  `/ocr` is all-or-nothing and stops at the first problem: a validation/provider failure raises
  `HTTPException`; a type mismatch, unresolved incompleteness, rejected pairing or verification
  finding raises `ValidationStopped` carrying the blank-`data` errorInfo response. Cross-
  verification runs only after every document passed; crop/rotate only after the whole batch did.
- `app/services/leasing.py` — `/ocr/leasing` batch logic on top of the same pipeline steps, but
  every document is checked in full (pairing per document type) and each failure is reported on
  its own document; `form_data` is still all-or-nothing. Outputs go to `LEASING_OUTPUT_DIR`.
- `app/services/post_processing.py` — parses the model's field lines, builds /ocr error payloads.
- `app/services/validation.py` — upload validation and the in-memory rate limiter.

### Gotchas learned the hard way

- **Repeatability comes from constraining the output, not from sampling settings.** `top_edge`
  is a four-value enum and the quarter-turn arithmetic lives in code. Gemini does expose
  `temperature`, but pinning it would not on its own make an answer correct - the enum is what
  removed the coin flip.
- **The orientation call returns `top_edge`, never an angle.** `TOP_EDGE_ROTATIONS` in
  `app/services/detection.py` maps the edge to degrees. Do not move that arithmetic back into the
  prompt — the model got it wrong half the time when it did it.
- **`hits.log` and `token-breakdown.log` only record the extraction call.** Box detection and
  orientation appear solely as INFO lines in `ocr-api.log`. Any cost analysis that reads only
  `hits.log` undercounts by roughly 60%.
- **`results.log` writes extracted KYC values in plaintext** (Emirates ID, DOB, passport numbers).
  Development only — see the retention warning in `README.md`.
- **`app/core/config.py` instantiates `Settings` at import**, so anything importing project modules needs
  `GEMINI_API_KEY` and `GEMINI_MODEL` present (from `.env` or the environment).
- **Bash heredocs with quoted delimiters sometimes fail** in this environment on content with
  apostrophes. Write the script to a file with the `Write` tool and execute it instead.
