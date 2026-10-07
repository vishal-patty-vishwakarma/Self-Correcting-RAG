<h1 align="center">Self-Correcting RAG</h1>

<p align="center">
  <strong>A closed-loop evidence assessment and recovery layer built on top of BookRAG</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.03413">
    <img src="https://img.shields.io/badge/Base%20paper-arXiv%3A%202512.03413-b31b1b?style=flat-square" alt="Base paper">
  </a>
  <a href="./BOOKRAG_VLDB_2026_full.pdf">
    <img src="https://img.shields.io/badge/Base%20paper%20PDF-VLDB%202026-7c3aed?style=flat-square" alt="Base paper PDF">
  </a>
  <a href="./LICENSE_SCOPE.md">
    <img src="https://img.shields.io/badge/license-AGPL%20%2F%20Apache--2.0-111827?style=flat-square" alt="License">
  </a>
</p>

---

## 1. Overview

This repository is an **in-progress research extension of [BookRAG](https://github.com/sam234990/BookRAG)** (accepted at VLDB 2026).

BookRAG answers questions over long, highly structured documents (manuals, handbooks, reports) by building a **BookIndex**: a hierarchical document tree + an entity-relation graph + vector indexes, then retrieving with an agent-style planner over that index.

Retrieval still fails silently sometimes: the pipeline returns evidence that *looks* plausible but is incomplete, off-topic, or contradictory — and then generates an answer anyway.

**Self-Correcting RAG** adds a *closed feedback loop between retrieval and answer generation*:

> **retrieval → assess evidence → decide → recover → re-assess → … (bounded)**

It measures the evidence the retriever produced, diagnoses *why* it is weak, picks a targeted recovery strategy, expands the evidence set, and only then generates an answer. If the evidence is already sufficient it stops immediately and does no extra work.

Everything in the loop (assessment, decision, recovery) is currently **deterministic and LLM-free** — see [Limitations](#9-current-limitations).

---

## 2. What Self-Correcting RAG Does

| Step | Component | Responsibility |
| --- | --- | --- |
| 1 | **Retriever** (BookRAG GBC) | Produces an initial evidence set of tree/graph nodes. |
| 2 | **EAM** — Evidence Assessment Module | Scores the evidence and emits a status + diagnosis. |
| 3 | **Retry Controller** | Maps diagnosis → strategy, or decides to STOP. |
| 4 | **Recovery Executor** | Expands the evidence set using one of four strategies (S1–S4). |
| 5 | **Orchestrator** | Runs 2–4 as a bounded loop, merges/dedups evidence, returns the final set. |
| 6 | **Answer generator** | Runs only after the loop finishes. |

### Evidence statuses and diagnoses

`EvidenceStatus`: `SUFFICIENT` · `PARTIAL` · `INSUFFICIENT` · `CONTRADICTORY`

`EvidenceDiagnosis`: `NONE` · `NO_PROGRESS` · `LOW_RELEVANCE` · `LOW_COVERAGE` · `WEAK_SUPPORT` · `MISSING_ENTITY` · `MULTI_HOP_GAP` · `CONTRADICTION`

### Recovery strategies

| ID | Name | What it expands to | Needs KG? |
| --- | --- | --- | --- |
| **S1** | `LOCAL_CONTEXT` | node + parent + direct children + previous/next siblings | no |
| **S2** | `ENTITY_GUIDED` | tree evidence → linked / query-matching graph entities → their source tree nodes | yes |
| **S3** | `GRAPH_MULTI_HOP` | seed entities → bounded BFS over the KG (2 hops, ≤ 20 entities) → source tree nodes | yes |
| **S4** | `STRUCTURAL_SCOPE` | wider structural window: anchor subtree + nearby sibling sections | no |

S1/S4 are always available. S2/S3 fall back automatically when the knowledge graph is empty or unavailable.

### Stop conditions

The controller stops on any of: `SUFFICIENT`, `MAX_ATTEMPTS` (default 4), `NO_APPLICABLE_STRATEGY`, `NO_PROGRESS`, `CONTRADICTION_UNRESOLVED`.

Contradictory evidence gets one forced S4 pass; if the conflict remains, the loop **stops rather than guessing** at a resolution.

### Where it hooks in

Self-correction is **not a separate retrieval strategy**. It is a flag on the existing `gbc` strategy:

```yaml
rag:
  strategy: gbc
  self_correcting_enabled: true   # <- turn the loop on/off
  max_recovery_attempts: 4
  recovery_max_nodes: 25
```

It runs **after retrieval and before answer generation**, for simple queries and for each complex retrieval sub-question. Filter/count/list ("global") queries do not run the loop.

Per-iteration trace is written into `retrieval_res.json` under `self_correcting_assessments`, `self_correcting_decisions`, `self_correcting_final_status`, `self_correcting_stop_reason`.

---

## 3. Architecture

```text
                         ┌──────────────────────────────────────────┐
   PDF ──► MinerU ──────►│  OFFLINE INDEX CONSTRUCTION (BookRAG)     │
                         │  tree ── graph ── VDB ── reranker index   │
                         └───────────────────┬──────────────────────┘
                                             │ BookIndex
                                             ▼
   query ──► planner ──► retriever (GBC) ──► initial evidence
                                             │
                       ┌─────────────────────▼─────────────────────┐
                       │   SELF-CORRECTING CLOSED LOOP  (this repo) │
                       │                                            │
                       │   EAM ──► Controller ──► Recovery (S1-S4)   │
                       │    ▲                              │        │
                       │    └────────── merged evidence ◄──┘        │
                       │          (stop when SUFFICIENT             │
                       │           or budget exhausted)             │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                        answer generator ──► result.json
```

**Key modules**

| Path | Purpose |
| --- | --- |
| `Core/rag/self_correcting/eam.py` | Evidence Assessment Module (heuristic scorer) |
| `Core/rag/self_correcting/controller.py` | Retry controller: diagnosis → strategy / stop |
| `Core/rag/self_correcting/recovery.py` | Recovery executor: S1–S4 expansion logic |
| `Core/rag/self_correcting/strategies.py` | Diagnosis → strategy selector |
| `Core/rag/self_correcting/orchestrator.py` | The bounded closed loop |
| `Core/rag/self_correcting/types.py` | Statuses, diagnoses, assessment/result dataclasses |
| `Core/rag/gbc_rag.py` | BookRAG GBC retriever + the hook that invokes the loop |
| `Core/Index/` | Document tree, knowledge graph, GBC index |
| `Core/pipelines/` | PDF → tree / KG / VDB index builders |
| `Core/provider/` | LLM, VLM, embedding, reranker, vector-DB providers |
| `main.py` | CLI entry point (`index` / `rag`) |
| `experiments/` | Benchmark and diagnostic scripts (see §7) |
| `Eval/` | Dataset evaluation + answer extraction |

Preserved baseline copies (`*_before_self_correcting.py`, `*_original.py`, `main_original.py`) are kept for reference and are **not imported** by the live code.

---

## 4. Repository Structure

```text
BookRAG/
├── Core/                     # indexing, retrieval, generation, self-correcting loop
│   ├── rag/self_correcting/  # EAM, controller, recovery, orchestrator
│   ├── Index/                # document tree + knowledge graph
│   ├── pipelines/            # index construction pipelines
│   ├── provider/             # LLM / VLM / embedding / reranker / VDB
│   ├── prompts/              # prompt templates
│   └── configs/              # pydantic config schema
├── config/                   # YAML run configurations (*.example.yaml are portable)
├── experiments/              # controlled benchmark + diagnostics
├── Eval/                     # dataset evaluation (Qasper / m3docvqa / MMLongBench)
├── Scripts/                  # example index/rag/eval shell scripts + dataset cfg
├── datasets/                 # filtered QA files used in the paper experiments
├── tests/                    # unit tests for the self-correcting loop
├── test_data/                # controlled benchmark queries (12 questions)
├── assets/                   # README figures
├── LICENSES/                 # full license texts
├── main.py                   # CLI entry point
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 5. Requirements & Setup

**Python 3.12.**

```bash
# Option A - conda
conda env create -f environment.yml
conda activate gbc-rag

# Option B - pip
pip install -r requirements.txt
```

Sanity check (no models, no network):

```bash
python main.py --help
```

Notes:

- PDF parsing uses [MinerU](https://github.com/opendatalab/MinerU). The default configs use the `pipeline` backend; a `vlm-sglang-client` backend needs a running MinerU/SGLang service and a valid `mineru.server_url`.
- Graph extraction uses the local spaCy model `en_core_web_sm` (pinned in `requirements.txt`).
- The embedding and reranker models (`Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`) are downloaded locally on first use.
- If MinerU warns that `sglang` is not installed, it is harmless unless you use the `sglang-engine` backend.

---

## 6. Configuring Credentials Safely

> **This is a public repository. Never put a real API key in a tracked file.**

### How credentials are read

The pipeline reads `llm.api_key` / `vlm.api_key` **directly from the YAML config** and passes it to an OpenAI-compatible client. There is no environment-variable fallback in the core pipeline.

The evaluation answer extractor (`Eval/utils/extract_answer.py`) reads a separate 3-line file, **`Eval/utils/api.txt`**:

```text
line 1: base_url
line 2: api_key
line 3: model_name
```

### What is git-ignored

The following are **never** committed (see `.gitignore`):

- `**/api.txt` and any `*.env`
- the six configs that held a real key locally:
  `self_correcting.yaml`, `self_correcting_kg_fallback.yaml`, `sc_stress.yaml`,
  `baseline_stress.yaml`, `baseline_kg_fallback.yaml`, `foresight_ai_jd.yaml`

### Setup after cloning

Each of those six ships as a portable template instead:

```bash
# Windows
copy config\self_correcting.example.yaml config\self_correcting.yaml
# macOS / Linux
cp config/self_correcting.example.yaml config/self_correcting.yaml
```

Then edit the three `TODO` values in your private copy:

```yaml
pdf_path: TODO   # path to your input PDF
save_path: TODO  # where the index and run outputs are written
llm:
  api_key: TODO  # your Gemini / OpenAI-compatible key
```

Your private copy stays untracked, so a plain `git add -A` will never pick it up.

### Extra safeguards already in the code

`main.py` redacts `api_key`, `access_token`, `auth_token`, `password`, `secret`, `client_secret` from run logs and from the `run_config.yaml` snapshots it writes (replacing them with `***REDACTED***`). The legacy `main_original.py` does **not** do this.

---

## 7. Running the Project

All commands are run from the repository root.

### 7.1 Build the index

```bash
python main.py -c config/self_correcting.yaml index --stage all
```

`--stage` choices: `tree`, `graph`, `vdb`, `mm_reranker`, `rebuild_graph_vdb`, `all` (default).

Batch indexing over a dataset, split across machines/processes:

```bash
python main.py -c config/gbc.yaml -d Scripts/cfg/example-Qasper.yaml \
    --nsplit 2 --num 1 index --stage graph
```

The equivalent shell wrapper is `Scripts/example-index.sh`.

### 7.2 Run a RAG query

**With self-correction** (any config where `rag.self_correcting_enabled: true`):

```bash
python main.py -c config/self_correcting.yaml rag -q "Who manufactures Valve X?"
```

**Baseline / no self-correction** (flag `false` or absent):

```bash
python main.py -c config/baseline_stress.yaml rag -q "Who manufactures Valve X?"
```

Single-query outputs are written to `<save_path>/eval_single_query_sc_on_gbc_standard/` (or `..._sc_off_...`), each containing `result.json` and `retrieval_res.json`.

Batch mode:

```bash
python main.py -c config/default.yaml -d Scripts/cfg/example-MMLongBench.yaml \
    --nsplit 2 --num 1 rag
```

Shell wrapper: `Scripts/example-rag.sh`.

### 7.3 Run the controlled benchmark

`experiments/run_controlled_benchmark.py` runs the 12 questions in `test_data/controlled_benchmark_queries.json` through **both** configurations and compares them:

```bash
python experiments/run_controlled_benchmark.py --mode both --repeat-count 1
```

| Flag | Meaning |
| --- | --- |
| `--mode {baseline,sc,both}` | which arm(s) to run (default `both`) |
| `--repeat-count N` | repeats per query |
| `--query-id Q07` | run only specific IDs (repeatable) |
| `--resume` | skip queries that already have valid results |

It requires `config/baseline_kg_fallback.yaml` and `config/self_correcting_kg_fallback.yaml` (create both from their `.example.yaml` templates first).

Results land in `test_output/controlled_manual/final_benchmark/run_NNN/`:

```text
benchmark_results.json / .csv          # run-level aggregates
Qxx/baseline/benchmark_result.json     # per-query, per-arm
Qxx/baseline/retrieval_res.json        # full retrieval trace + SC diagnostics
Qxx/sc/...
```

Correctness is decided **heuristically** by matching `expected_identifiers` / `absence_phrases` in the answer — it is not human grading.

### 7.4 Other experiments and diagnostics

| Script | Purpose |
| --- | --- |
| `run_local_baseline_vs_sc.py` | LLM-free, tree-only 4-case baseline vs. self-correcting comparison |
| `run_local_sc_evaluation.py` | Same harness, PASS/FAIL against expected status & diagnosis |
| `compare_fixed_evidence_baseline_vs_s3.py` | Fixed-evidence A/B of baseline vs. full S3 recovery |
| `replay_real_s2_with_answer.py` / `replay_real_s3_with_answer.py` | Replay a specific recovery strategy on fixed evidence, then answer |
| `replay_real_s3_from_baseline.py` | Orchestrator-only trace from a fixed starting node |
| `screen_retrieval_candidates.py` | Does baseline retrieval produce evidence weak enough to trigger recovery? |
| `screen_baseline_evidence.py` | Offline EAM screening of a saved baseline run |
| `diagnose_entity_vdb.py` / `rebuild_entity_vdb.py` | Entity vector-DB inspection / rebuild |
| `diagnose_s2_eam_false_positive.py` | Investigates an S2 false-positive case |
| `validate_real_graph_recovery.py` | Exercises S1–S4 against the real tree + graph |
| `replay_saved_eam_after_fix.py` | Re-runs current EAM over a saved run and diffs assessments |

Most of these read a pre-built `tree.pkl` from your local `save_path` and therefore **must be run after indexing**.

### 7.5 Tests

```bash
python -m unittest tests.test_self_correcting
```

Covers EAM scoring/diagnosis, the retry controller, S1–S4 expansion, and the closed loop (~25 tests). **The suite currently loads a `tree.pkl` from a local workspace path** (see `tests/test_self_correcting.py`), so it only runs once you have built an index.

### 7.6 Evaluation on public datasets

```bash
python Eval/evaluation.py -d Scripts/cfg/example-Qasper.yaml --method gbc_standard
```

Supported dataset names: `qasper`, `m3docrag`, `MMLongBench`. Metrics are computed by the official/ upstream scripts under `Eval/utils/`. The answer extractor needs `Eval/utils/api.txt` populated (§6).

---

## 8. Research & Evaluation Status

**Status: work in progress. The self-correcting loop is implemented and instrumented; accuracy benefits are not yet demonstrated.**

What exists:

- Full closed loop (EAM → controller → S1–S4 → orchestrator) wired into the GBC retriever, switchable by one config flag.
- A 12-question controlled single-document benchmark with paired baseline/self-correcting arms, per-query traces, and run-level aggregates.
- Instrumentation of every loop decision written into `retrieval_res.json`.

Results so far (from `test_output/controlled_manual/final_benchmark/`, scored by the heuristic identifier checker):

| Run | Queries | Baseline correct | Self-correcting correct | Notes |
| --- | --- | --- | --- | --- |
| `run_002` | 2 | 2 / 2 | 2 / 2 | recovery never triggered |
| `run_003` | 12 | 12 / 12 | 12 / 12 | SC stopped immediately on 11/12, ran recovery on 1 |
| `run_001`, `run_006`–`run_008` | 2 / 4 / 1 / 1 | 0 | 0 | **every API call failed** (Gemini `503 UNAVAILABLE`) — *these are failures, not accuracy numbers* |
| `run_004` | 4 (×3 repeats) | — | — | per-query repeats only, no run-level aggregate |

Honest reading of the above:

- On the one complete 12-query run, **baseline and self-correcting both scored 12/12**; the loop improved 0 queries and degraded 0 queries. There is no measured accuracy gain yet.
- The benchmark is small (one document, 12 questions) and scored by a keyword/identifier heuristic, not by humans or an LLM judge.
- Several runs are unusable because of upstream API outages; they must not be quoted as accuracy.
- No multi-document study, no repeated-seed statistics, no cost/latency ablation yet.

---

## 9. Current Limitations

- **The EAM is a deterministic heuristic.** Its own docstring states it does not yet use an LLM, NLI, embeddings, a cross-encoder, or the BookRAG knowledge graph. It estimates relevance, coverage, support, contradiction and progress from lexical overlap and conflict markers.
- **Thresholds are hard-coded in `Core/rag/self_correcting/eam.py`** (`relevance 0.25`, `coverage 0.70`, `support 0.65`, `contradiction 0.60`, `progress 0.10`) and are **not** configurable from YAML.
- **Recovery never calls an LLM.** Seed entities come from graph links plus a purely lexical query↔entity matcher, so entity recall is limited.
- **Contradictions are not resolved** — the loop deliberately stops (`CONTRADICTION_UNRESOLVED`) instead of picking a winner.
- **Global queries** (filter / count / list) skip the self-correcting loop entirely.
- **S2 and S3 need a populated knowledge graph**; without one they are unavailable and the controller falls back to S1/S4.
- **Gemini rate limiting is hard-coded** (≈13 s minimum interval, `max_retries=0`, `timeout=60`), reflecting an observed ~5 requests/minute project quota.
- **Absolute paths** remain in some experiment scripts and in the local (git-ignored) configs; adjust `pdf_path`, `save_path` and any `save_path`-derived paths to your machine.

---

## 10. Not Included in This Repository

Generated and heavyweight artifacts are **git-ignored and must be rebuilt locally**:

| Not included | How to obtain it |
| --- | --- |
| Built indexes — `Tree_vdb/`, `kg_vdb*/`, `entity_vdb/`, `*.pkl`, `*.pickle` | `python main.py -c <config> index --stage all` |
| Local model weights — `*.safetensors`, `*.bin`, `*.pt`, `*.gguf` | downloaded automatically by `modelscope` / `transformers` on first run |
| Model caches — `.cache/`, `.modelscope/`, `.huggingface/` | created automatically |
| Vector DB contents and Chroma data | created during indexing |
| Run outputs — `test_output/`, `eval_single_query_*/`, `run_config.yaml` | produced by `rag` and the benchmark |
| Logs — `*.log` | produced at runtime |
| Input PDFs | supply your own; `pdf_path` in your private config |
| API keys — `api.txt`, `*.env`, the six private configs | see §6 |

> **A fresh clone will not run out of the box.** You must supply an API key, point `pdf_path` at a document, build an index, and then run queries.

Small data assets that **are** included: `datasets/*.json` (filtered QA files), `test_data/controlled_benchmark_queries.json`, `assets/*` (README figures), and `BOOKRAG_VLDB_2026_full.pdf` (the base paper).

---

## 11. Quick Start for Teammates

```bash
git clone https://github.com/vishal-patty-vishwakarma/Self-Correcting-RAG.git
cd Self-Correcting-RAG

conda env create -f environment.yml
conda activate gbc-rag

# 1. create your private config from the template
cp config/self_correcting.example.yaml config/self_correcting.yaml
#    ...then edit pdf_path, save_path and llm.api_key

# 2. build the index
python main.py -c config/self_correcting.yaml index --stage all

# 3. ask a question
python main.py -c config/self_correcting.yaml rag -q "Who manufactures Valve X?"

# 4. optional: controlled benchmark (needs both *_kg_fallback configs)
cp config/self_correcting_kg_fallback.example.yaml config/self_correcting_kg_fallback.yaml
cp config/baseline_kg_fallback.example.yaml config/baseline_kg_fallback.yaml
python experiments/run_controlled_benchmark.py --mode both --repeat-count 1
```

---

## 12. License & Attribution

This project builds on **BookRAG** by Shu Wang, Yingli Fang and Yixiang Zhou.

The root [`LICENSE`](./LICENSE) provides the GNU Affero General Public License v3.0 text for repository-level license identification. License terms for individual source files are specified by their SPDX identifiers; the root license does not automatically apply to every file.

- Independently authored components (including `Core/` and `main.py` upstream parts) are available under **Apache-2.0 or AGPL-3.0-only**.
- Files derived from or adapted for **MinerU** are **AGPL-3.0-only**; a distribution that includes them must comply with AGPL-3.0.
- `Eval/utils/m3doc_eval.py` is derived from the DROP evaluation implementation and is **not** covered by the BookRAG license grants.
- Datasets, evaluation resources, the paper PDF, figures and third-party materials remain under their own terms.

See [`LICENSE_SCOPE.md`](./LICENSE_SCOPE.md) and [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for details.

The BookRAG paper was accepted to **PVLDB Volume 19 / VLDB 2026** and is governed by PVLDB publication terms (CC BY-NC-ND 4.0), not by the software licenses above.

---

## 13. Citation

If you use the base BookRAG system, please cite:

```bibtex
@article{wang2025bookrag,
  title   = {BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents},
  author  = {Wang, Shu and Zhou, Yingli and Fang, Yixiang},
  journal = {arXiv preprint arXiv:2512.03413},
  year    = {2025},
  eprint  = {2512.03413},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2512.03413}
}
```

**Links:** [arXiv](https://arxiv.org/abs/2512.03413) · [Base paper PDF](./BOOKRAG_VLDB_2026_full.pdf) · [Upstream BookRAG](https://github.com/sam234990/BookRAG)
