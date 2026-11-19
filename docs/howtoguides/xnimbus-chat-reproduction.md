# Reproducing the grounded XNIMBUS chat

This guide rebuilds, tests, runs and evaluates the grounded conversational interface for Explainable
NIMBUS (XNIMBUS) from a clean checkout. Each step names the command, where to run it, and what you
should see. The design is described in
[Grounded conversational XNIMBUS](../explanation/xnimbus-conversational-architecture.md).

## 1. What you can reproduce

| Artifact | Command (section) | Expected result |
|---|---|---|
| Backend chat tests | §5 | `158 passed` (chat files) |
| Web UI chat tests | §5 | `6 passed` in the two chat test files |
| Benchmark snapshot `v3` | §6 | byte-identical file, SHA-256 `ad7e402c0545d2c53975edb69e521cafb162e58215683c59816d4448e230f005` |
| Benchmark dry run | §7 | `unsafe_answer_rate 0.00`, `abstention_recall 1.00`, `action_block_rate 1.00` |
| Model benchmark | §7 | a JSON audit log and a Markdown summary, both with provenance |
| One real chat turn via the API | §8 | `status: verified` |
| The chat in the web UI | §9 | answers below the XNIMBUS explanations |

## 2. Versions used

Results in the thesis were produced with these versions. Python and JavaScript dependencies are
pinned by `uv.lock` and `webui/package-lock.json`.

| Component | Version |
|---|---|
| Code | `salmanprottoy/DESDEO`, branch `feature/xnimbus-chat-on-gui-xnimbus` (based on `giomara-larraga/DESDEO` `GUI_XNIMBUS` @ `8554f1f0`) |
| Machine | Apple M4, 16 GB RAM, macOS 27.2.1 (arm64) |
| Python / uv | 3.13.13 / uv 0.13.8 |
| Node / npm | 22.23.2 / 10.9.8 |
| Ipopt | 3.14.20 (used to solve the benchmark snapshot) |
| Ollama | 0.40.16 |
| Prompt version | `xnimbus-grounded-v10` |

Local models (Ollama tag, size, quantization, digest prefix). Every benchmark report records the
full digest of each local model it evaluated (§7).

| Tag | Parameters | Quantization | Digest |
|---|---|---|---|
| `ministral-3:8b` (default) | 8.9B | Q4_K_M | `1922accd5827` |
| `qwen3.5:9b` | 9.7B | nvfp4 | `d63f830f28d5` |
| `gemma4:12b` | 12.4B | nvfp4 | `c7597fc90b86` |
| `llama3.1:8b` | 8.0B | Q4_K_M | `46e0c10c039e` |

## 3. Get the code and install

```bash
git clone https://github.com/salmanprottoy/DESDEO.git
cd DESDEO
git checkout feature/xnimbus-chat-on-gui-xnimbus
uv sync
cd webui && npm ci && cd ..
```

The repository folder must be called `DESDEO`: some test problems locate their data by that name.

Install [Ipopt](https://coin-or.github.io/Ipopt/) (`brew install ipopt` on macOS) and
[Ollama](https://ollama.com), then pull the models:

```bash
ollama pull ministral-3:8b
ollama pull qwen3.5:9b
ollama pull gemma4:12b
ollama pull llama3.1:8b
```

A tag can point to newer weights later. Compare `ollama list` with the digests above; if they
differ, the model is not the one used in the thesis.

## 4. Configure the models

The server reads model profiles from environment variables; the browser can only choose among them.
For local models only:

```bash
export XNIMBUS_LOCAL_MODELS=qwen3.5:9b,gemma4:12b,ministral-3:8b,llama3.1:8b
export XNIMBUS_LLM_DEFAULT_PROFILE=local:ministral-3:8b
```

Optional settings: `XNIMBUS_LOCAL_BASE_URL` (default `http://localhost:11434/v1`),
`XNIMBUS_LOCAL_TIMEOUT_SECONDS` (default 90). Cloud providers are enabled by their API keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`) and the
matching `*_MODELS` variable; see the README section "XNIMBUS grounded chat providers". Never commit
keys; keep them in an untracked `.env`.

Decoding is fixed in code: temperature 0, and thinking is turned off for the Qwen 3 and Gemma 4
families (`desdeo/api/llm_config.py`).

## 5. Run the tests

Backend (from the repository root):

```bash
uv run pytest tests/test_anthropic.py tests/test_chat_persistence.py tests/test_deepseek.py tests/test_evaluation.py tests/test_evidence.py tests/test_experiment_context.py tests/test_google_ai.py tests/test_grounded_chat.py tests/test_llm_config.py tests/test_openai_compatible.py tests/test_xnimbus_benchmark.py desdeo/api/tests/test_chat_schema.py desdeo/api/tests/test_xnimbus_chat.py -n 8
```

Web UI (from `webui/`):

```bash
npx vitest --config vitest.config.ts --run src/lib/components/custom/xnimbus src/lib/components/custom/preferences-bar
```

Known failures that exist on `GUI_XNIMBUS` before the chat was added, and are unrelated to it:
`desdeo/api/tests/test_models.py::test_rpm_state`,
`test_models.py::test_nimbus_initialize_w_multidimensional_constraints`,
`test_routes.py::test_add_problem_json`, `test_routes.py::test_get_available_solvers`,
`test_routes.py::test_gdm_score_bands`, and `webui/src/routes/page.svelte.test.ts` (imports a
missing `+page.svelte`).

## 6. Rebuild the benchmark snapshot

The benchmark is one XNIMBUS iteration on the four-objective river pollution problem, solved with
Ipopt, plus 40 labelled questions about it. The committed file is
`datasets/xnimbus_chat_benchmark_v3.json`. To check it:

```bash
uv run python scripts/xnimbus_eval/build_benchmark.py --out /tmp/v3-rebuild.json
shasum -a 256 datasets/xnimbus_chat_benchmark_v3.json /tmp/v3-rebuild.json
```

Both lines should show `ad7e402c…f005`. A different Ipopt build or platform can change solver output
in the last digits; then the checksums differ. Use the committed file for comparisons and report the
rebuilt numbers separately.

How the snapshot is made: starting point from `generate_starting_point`, then a classification that
improves WQCity (f_1) by 0.1 and lets RFish (f_3) worsen by 1.0, solved for 2 solutions (NIMBUS and
STOM). Multipliers, trade-offs and active objectives are computed with the same functions as the
XNIMBUS explanation panel (`desdeo/api/routers/xnimbus.py`).

## 7. Run the benchmark

Dry run (no model; checks the pipeline in seconds):

```bash
uv run python scripts/xnimbus_eval/run_benchmark.py --dry-run --repetitions 1
```

Model run (Ollama running, §4 variables set). The thesis protocol is 3 repetitions per model:

```bash
uv run python scripts/xnimbus_eval/run_benchmark.py --providers local --repetitions 3
```

Output goes to `scripts/xnimbus_eval/results/` as `<benchmark>-<UTC time>.json` (every case,
attempt, raw model answer, verifier reason, latency and token use) and a `.md` summary. Both record
**provenance**: git branch and commit, whether tracked files had uncommitted changes, the benchmark
SHA-256, Python version, platform, and each local model's Ollama digest. Only report numbers from runs
with `uncommitted changes: False`.

Run time on the machine above: roughly 17–42 s per question on average for the local models, so one
model with 3 repetitions takes about 30–90 minutes.

## 8. One real chat turn through the API

Uses an in-memory database and never touches `test.db`:

```bash
uv run python scripts/xnimbus_eval/run_api_smoke.py --model ministral-3:8b
```

Expected: a JSON response with `"status": "verified"` and the objective values named, for example
`f_1 (WQCity)`, plus the audit record saved for that answer.

## 9. Run the application

```bash
uv run python desdeo/api/db_init_xnimbus.py
```

This **drops and recreates** the tables in `test.db` and seeds the river pollution problem for the
test users (log in as `analyst` / `analyst`, from `desdeo/api/config.toml`). It also creates the chat
tables. For an existing database without them, run
`uv run python scripts/init_xnimbus_chat_schema.py`, which only adds missing tables.

API (with the §4 variables in the same shell):

```bash
uv run uvicorn desdeo.api.app:app --port 8000
```

Web UI: create `webui/.env` with `API_BASE_URL="http://localhost:8000"` and `VITE_API_URL="/api"`, then:

```bash
cd webui && npm run dev -- --port 5173
```

Open <http://localhost:5173>, log in, choose **Explainable NIMBUS**, set a classification (for example
RFish "impair until" 5), ask for 2 solutions and press **Iterate**. The chat appears at the bottom of
the **Explanations** panel for the selected solution. Enter sends; Shift+Enter adds a line.

## 10. Behaviour to expect

- The chat quotes the same multipliers, trade-offs and active objectives as the XNIMBUS panel.
- In the running application, trade-off questions are answered with an abstention
  (`actual_solver_unknown`): the request does not carry which solver produced the solution, and the
  chat does not quote trade-offs without that. The benchmark passes the solver explicitly.
- Questions asking for the exact value of a new solution ("if I set RFish to 7, what exactly would
  WQCity be?") are declined: a local trade-off cannot give it.
- Requests to change the session (solve, save, finalize) are blocked; the chat is read-only.
- Results measured on `datasets/xnimbus_chat_benchmark_v2.json` (upstream `master` code) are not
  comparable: v3 uses the panel's scaled multipliers and this branch's objective names.

## 11. Where things are

| Path | Contents |
|---|---|
| `desdeo/api/evidence.py` | Evidence facts with stable ids, built from the stored solver results |
| `desdeo/api/experiment_context.py` | Facts about the whole decision experiment (iterations, history) |
| `desdeo/api/grounded_chat.py` | Intent routing, policy, prompt, verifier, rendering, fallback |
| `desdeo/api/llm_config.py` and provider modules | Server-owned model profiles and provider adapters |
| `desdeo/api/routers/xnimbus_chat.py` | `/method/xnimbus/chat` and `/chat/models`, persistence, audit |
| `desdeo/api/evaluation.py` | Benchmark model, metrics, provenance |
| `scripts/xnimbus_eval/` | Benchmark builder, runner, API smoke test |
| `datasets/xnimbus_chat_benchmark_v3.json` | Benchmark used for this branch |
| `webui/src/lib/components/custom/xnimbus/xnimbus-chat.svelte` | Chat panel |
