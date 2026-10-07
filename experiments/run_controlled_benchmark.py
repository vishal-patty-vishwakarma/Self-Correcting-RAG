import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.configs.llm_config import LLMConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.Index.GBCIndex import GBC
from Core.rag import create_rag_agent
from Core.rag.gbc_utils import SubStep, GBCRAGContext
from Core.provider.TokenTracker import TokenTracker

QUERIES_PATH = PROJECT_ROOT / "test_data" / "controlled_benchmark_queries.json"
BASELINE_CONFIG = "config/baseline_kg_fallback.yaml"
SC_CONFIG = "config/self_correcting_kg_fallback.yaml"
OUTPUT_BASE = PROJECT_ROOT / "test_output" / "controlled_manual" / "final_benchmark"


def find_typed_config(obj, target_type, max_depth=5):
    visited = set()
    def walk(value, depth):
        if value is None:
            return None
        if isinstance(value, target_type):
            return value
        if depth > max_depth:
            return None
        value_id = id(value)
        if value_id in visited:
            return None
        visited.add(value_id)
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)):
            children = value
        else:
            try:
                children = vars(value).values()
            except TypeError:
                return None
        for child in children:
            found = walk(child, depth + 1)
            if found is not None:
                return found
        return None
    return walk(obj, 0)


def build_rag_agent(config_path):
    system_cfg = load_system_config(config_path)
    gbc = GBC.load_gbc_index(system_cfg)
    rag_cfg = find_typed_config(system_cfg, GBCRAGConfig)
    llm_cfg = find_typed_config(system_cfg, LLMConfig)
    vlm_cfg = find_typed_config(system_cfg, VLMConfig)
    if rag_cfg is None:
        raise RuntimeError(f"Could not locate GBCRAGConfig in {config_path}")
    if llm_cfg is None:
        raise RuntimeError(f"Could not locate LLMConfig in {config_path}")
    if vlm_cfg is None:
        raise RuntimeError(f"Could not locate VLMConfig in {config_path}")
    rag = create_rag_agent(
        strategy_config=rag_cfg, llm_config=llm_cfg,
        vlm_config=vlm_cfg, gbc_index=gbc,
    )
    return rag


def check_correctness(answer, query_def):
    expected_ids = query_def.get("expected_identifiers", [])
    answer_type = query_def.get("expected_answer_type", "fact")
    if answer_type == "absence":
        absence_phrases = query_def.get("absence_phrases", [])
        answer_lower = answer.lower()
        for phrase in absence_phrases:
            if phrase.lower() in answer_lower:
                return "CORRECT"
        return "NEEDS_MANUAL_REVIEW"
    if not expected_ids:
        return "NEEDS_MANUAL_REVIEW"
    answer_upper = answer.upper()
    found = [eid for eid in expected_ids if eid.upper() in answer_upper]
    if len(found) == len(expected_ids):
        return "CORRECT"
    elif len(found) > 0:
        return "PARTIAL"
    else:
        return "INCORRECT"


def run_single_query(rag, query, query_output_dir, is_sc=False):
    query_output_dir.mkdir(parents=True, exist_ok=True)
    tracker = TokenTracker.get_instance()
    tracker.reset()
    start_time = time.perf_counter()
    try:
        answer, retrieval_ids = rag.generation(
            query=query, query_output_dir=query_output_dir,
        )
    except Exception as exc:
        return {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runtime_seconds": round(time.perf_counter() - start_time, 3),
        }
    elapsed = time.perf_counter() - start_time
    usage = tracker.get_usage()
    result = {
        "final_answer": answer,
        "retrieved_node_ids": retrieval_ids,
        "runtime_seconds": round(elapsed, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }
    if is_sc:
        sc_data = _extract_sc_trace(query_output_dir)
        result.update(sc_data)
    return result


def _extract_sc_trace(query_output_dir):
    res_file = query_output_dir / "retrieval_res.json"
    if not res_file.exists():
        return {}
    try:
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    iterations = data.get("iterations", [])
    if not iterations:
        return {}
    step = iterations[0]
    sc_assessments = step.get("self_correcting_assessments", [])
    sc_decisions = step.get("self_correcting_decisions", [])
    initial_status = ""
    initial_diagnosis = ""
    initial_relevance = 0.0
    initial_coverage = 0.0
    initial_support = 0.0
    initial_contradiction = 0.0
    if sc_assessments:
        first = sc_assessments[0]
        initial_status = first.get("status", "")
        initial_diagnosis = first.get("diagnosis", "")
        initial_relevance = first.get("relevance_score", 0.0)
        initial_coverage = first.get("coverage_score", 0.0)
        initial_support = first.get("support_score", 0.0)
        initial_contradiction = first.get("contradiction_score", 0.0)
    final_status = step.get("self_correcting_final_status", "")
    stop_reason = step.get("self_correcting_stop_reason", "")
    strategies = [d.get("strategy") for d in sc_decisions if d.get("strategy")]
    attempt_count = len(sc_assessments) - 1 if sc_assessments else 0
    return {
        "self_correcting_initial_status": initial_status,
        "self_correcting_initial_diagnosis": initial_diagnosis,
        "initial_relevance_score": initial_relevance,
        "initial_coverage_score": initial_coverage,
        "initial_support_score": initial_support,
        "initial_contradiction_score": initial_contradiction,
        "self_correcting_final_status": final_status,
        "attempt_count": attempt_count,
        "strategies_executed": strategies,
        "stop_reason": stop_reason,
    }


def classify_answer_change(baseline_correct, sc_correct):
    if baseline_correct == "CORRECT" and sc_correct == "CORRECT":
        return "BOTH_CORRECT"
    if baseline_correct == "INCORRECT" and sc_correct == "CORRECT":
        return "SC_IMPROVED"
    if baseline_correct == "CORRECT" and sc_correct in ("INCORRECT", "PARTIAL"):
        return "SC_DEGRADED"
    if baseline_correct == "PARTIAL" and sc_correct == "CORRECT":
        return "SC_IMPROVED"
    if baseline_correct in ("INCORRECT", "PARTIAL") and sc_correct in ("INCORRECT", "PARTIAL"):
        return "BOTH_INCORRECT"
    return "MANUAL_REVIEW"


def classify_recovery(baseline_correct, sc_result):
    sc_executed = sc_result.get("attempt_count", 0) > 0
    if not sc_executed:
        return "NO_RECOVERY"
    sc_correct = sc_result.get("correctness", "") == "CORRECT"
    if sc_correct and baseline_correct == "CORRECT":
        return "NOT_NEEDED"
    if sc_correct and baseline_correct != "CORRECT":
        return "POSSIBLE"
    if not sc_correct:
        return "FAILED_RECOVERY"
    return "UNKNOWN"

def compute_aggregates(comparisons):
    total = len(comparisons)
    baseline_correct = sum(1 for c in comparisons if c["baseline_correctness"] == "CORRECT")
    baseline_partial = sum(1 for c in comparisons if c["baseline_correctness"] == "PARTIAL")
    baseline_incorrect = sum(1 for c in comparisons if c["baseline_correctness"] == "INCORRECT")
    baseline_review = sum(1 for c in comparisons if c["baseline_correctness"] == "NEEDS_MANUAL_REVIEW")
    sc_correct = sum(1 for c in comparisons if c["sc_correctness"] == "CORRECT")
    sc_partial = sum(1 for c in comparisons if c["sc_correctness"] == "PARTIAL")
    sc_incorrect = sum(1 for c in comparisons if c["sc_correctness"] == "INCORRECT")
    sc_review = sum(1 for c in comparisons if c["sc_correctness"] == "NEEDS_MANUAL_REVIEW")
    baseline_runtimes = [c["baseline_runtime_seconds"] for c in comparisons if c.get("baseline_runtime_seconds")]
    sc_runtimes = [c["sc_runtime_seconds"] for c in comparisons if c.get("sc_runtime_seconds")]
    baseline_tokens = [c["baseline_total_tokens"] for c in comparisons if c.get("baseline_total_tokens")]
    sc_tokens = [c["sc_total_tokens"] for c in comparisons if c.get("sc_total_tokens")]
    sc_immediate_stop = sum(1 for c in comparisons if c.get("sc_stop_reason") == "SUFFICIENT")
    sc_recovery_executed = sum(1 for c in comparisons if c.get("sc_attempt_count", 0) > 0)
    strategy_counts = {}
    for c in comparisons:
        for s in c.get("sc_strategies", []):
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
    sc_improved = sum(1 for c in comparisons if c.get("answer_change") == "SC_IMPROVED")
    sc_degraded = sum(1 for c in comparisons if c.get("answer_change") == "SC_DEGRADED")
    both_correct = sum(1 for c in comparisons if c.get("answer_change") == "BOTH_CORRECT")
    both_incorrect = sum(1 for c in comparisons if c.get("answer_change") == "BOTH_INCORRECT")
    return {
        "total_queries": total,
        "baseline": {
            "correct": baseline_correct, "partial": baseline_partial,
            "incorrect": baseline_incorrect, "manual_review": baseline_review,
            "avg_runtime_seconds": round(sum(baseline_runtimes) / len(baseline_runtimes), 3) if baseline_runtimes else 0,
            "avg_total_tokens": round(sum(baseline_tokens) / len(baseline_tokens)) if baseline_tokens else 0,
        },
        "self_correcting": {
            "correct": sc_correct, "partial": sc_partial,
            "incorrect": sc_incorrect, "manual_review": sc_review,
            "avg_runtime_seconds": round(sum(sc_runtimes) / len(sc_runtimes), 3) if sc_runtimes else 0,
            "avg_total_tokens": round(sum(sc_tokens) / len(sc_tokens)) if sc_tokens else 0,
            "immediate_stop_count": sc_immediate_stop,
            "recovery_executed_count": sc_recovery_executed,
            "strategy_counts": strategy_counts,
        },
        "comparison": {
            "both_correct": both_correct,
            "sc_improved": sc_improved,
            "sc_degraded": sc_degraded,
            "both_incorrect": both_incorrect,
        },
    }


def save_csv(comparisons, csv_path):
    if not comparisons:
        return
    fieldnames = [
        "query_id", "category", "question", "expected_identifiers",
        "baseline_correctness", "sc_correctness",
        "baseline_answer", "sc_answer",
        "baseline_node_ids", "sc_initial_node_ids", "sc_final_node_ids",
        "sc_initial_status", "sc_initial_diagnosis",
        "sc_final_status", "sc_stop_reason",
        "sc_strategies", "sc_attempt_count",
        "baseline_runtime_seconds", "sc_runtime_seconds",
        "baseline_total_tokens", "sc_total_tokens",
        "recovery_executed", "recovery_help_classification",
        "answer_change", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in comparisons:
            row = {}
            for k in fieldnames:
                v = c.get(k, "")
                if isinstance(v, list):
                    v = "; ".join(str(x) for x in v)
                row[k] = v
            writer.writerow(row)


def load_queries(query_ids=None):
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if query_ids:
        queries = [q for q in queries if q["id"] in query_ids]
    return queries


def main():
    parser = argparse.ArgumentParser(description="Benchmark: Baseline vs Self-Correcting RAG")
    parser.add_argument("--query-id", action="append", dest="query_ids", help="Run only specific query IDs (repeatable)")
    parser.add_argument("--repeat-count", type=int, default=1, help="Number of repeats per query (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Skip queries with existing valid results")
    parser.add_argument("--mode", choices=["baseline", "sc", "both"], default="both", help="Run baseline, sc, or both")
    args = parser.parse_args()

    queries = load_queries(args.query_ids)
    if not queries:
        print("No queries to run.")
        return

    run_number = 1
    run_dir = OUTPUT_BASE / f"run_{run_number:03d}"
    while run_dir.exists():
        run_number += 1
        run_dir = OUTPUT_BASE / f"run_{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONTROLLED BENCHMARK: BASELINE vs SELF-CORRECTING RAG")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Queries: {len(queries)}")
    print(f"Repeat count: {args.repeat_count}")
    print(f"Mode: {args.mode}")
    print()

    baseline_rag = None
    sc_rag = None

    if args.mode in ("baseline", "both"):
        print("Building baseline RAG agent...")
        baseline_rag = build_rag_agent(BASELINE_CONFIG)
        print(f"  Self-correcting enabled: {baseline_rag.self_correcting_enabled}")
        print()

    if args.mode in ("sc", "both"):
        print("Building self-correcting RAG agent...")
        sc_rag = build_rag_agent(SC_CONFIG)
        print(f"  Self-correcting enabled: {sc_rag.self_correcting_enabled}")
        print()

    all_comparisons = []

    for qi, qdef in enumerate(queries):
        qid = qdef["id"]
        question = qdef["question"]
        expected = qdef.get("expected_identifiers", [])

        print(f"[{qi+1}/{len(queries)}] {qid}: {question}")
        print(f"  Expected: {expected}")

        comparison = {
            "query_id": qid,
            "category": qdef.get("category", ""),
            "question": question,
            "expected_identifiers": expected,
        }

        for repeat_idx in range(args.repeat_count):
            suffix = f"_r{repeat_idx+1}" if args.repeat_count > 1 else ""

            baseline_result = None
            sc_result = None

            if baseline_rag:
                bdir = run_dir / qid / f"baseline{suffix}"
                if args.resume and (bdir / "retrieval_res.json").exists():
                    print(f"  [resume] baseline exists, skipping")
                    with open(bdir / "benchmark_result.json", "r", encoding="utf-8") as f:
                        baseline_result = json.load(f)
                else:
                    print(f"  Running baseline...")
                    baseline_result = run_single_query(baseline_rag, question, bdir, is_sc=False)
                    with open(bdir / "benchmark_result.json", "w", encoding="utf-8") as f:
                        json.dump(baseline_result, f, indent=2, ensure_ascii=False)

                if "error_type" in baseline_result:
                    print(f"  BASELINE ERROR: {baseline_result['error_type']}: {baseline_result['error_message']}")
                    comparison["baseline_correctness"] = "ERROR"
                    comparison["baseline_answer"] = ""
                    comparison["baseline_runtime_seconds"] = baseline_result.get("runtime_seconds", 0)
                    comparison["baseline_total_tokens"] = 0
                    comparison["baseline_node_ids"] = []
                else:
                    bc = check_correctness(baseline_result["final_answer"], qdef)
                    comparison["baseline_correctness"] = bc
                    comparison["baseline_answer"] = baseline_result["final_answer"]
                    comparison["baseline_runtime_seconds"] = baseline_result["runtime_seconds"]
                    comparison["baseline_total_tokens"] = baseline_result["total_tokens"]
                    comparison["baseline_node_ids"] = baseline_result["retrieved_node_ids"]
                    print(f"  Baseline: {bc} ({baseline_result['runtime_seconds']}s, {baseline_result['total_tokens']} tokens)")

            if sc_rag:
                sdir = run_dir / qid / f"sc{suffix}"
                if args.resume and (sdir / "retrieval_res.json").exists():
                    print(f"  [resume] SC exists, skipping")
                    with open(sdir / "benchmark_result.json", "r", encoding="utf-8") as f:
                        sc_result = json.load(f)
                else:
                    print(f"  Running self-correcting...")
                    sc_result = run_single_query(sc_rag, question, sdir, is_sc=True)
                    with open(sdir / "benchmark_result.json", "w", encoding="utf-8") as f:
                        json.dump(sc_result, f, indent=2, ensure_ascii=False)

                if "error_type" in sc_result:
                    print(f"  SC ERROR: {sc_result['error_type']}: {sc_result['error_message']}")
                    comparison["sc_correctness"] = "ERROR"
                    comparison["sc_answer"] = ""
                    comparison["sc_runtime_seconds"] = sc_result.get("runtime_seconds", 0)
                    comparison["sc_total_tokens"] = 0
                    comparison["sc_initial_node_ids"] = []
                    comparison["sc_final_node_ids"] = []
                else:
                    sc_c = check_correctness(sc_result["final_answer"], qdef)
                    comparison["sc_correctness"] = sc_c
                    comparison["sc_answer"] = sc_result["final_answer"]
                    comparison["sc_runtime_seconds"] = sc_result["runtime_seconds"]
                    comparison["sc_total_tokens"] = sc_result["total_tokens"]
                    comparison["sc_initial_node_ids"] = []
                    comparison["sc_final_node_ids"] = sc_result["retrieved_node_ids"]
                    comparison["sc_initial_status"] = sc_result.get("self_correcting_initial_status", "")
                    comparison["sc_initial_diagnosis"] = sc_result.get("self_correcting_initial_diagnosis", "")
                    comparison["sc_final_status"] = sc_result.get("self_correcting_final_status", "")
                    comparison["sc_stop_reason"] = sc_result.get("stop_reason", "")
                    comparison["sc_strategies"] = sc_result.get("strategies_executed", [])
                    comparison["sc_attempt_count"] = sc_result.get("attempt_count", 0)
                    print(f"  SC: {sc_c} ({sc_result['runtime_seconds']}s, {sc_result['total_tokens']} tokens)")
                    print(f"    SC trace: {comparison['sc_initial_status']}/{comparison['sc_initial_diagnosis']} -> {comparison['sc_final_status']} ({comparison['sc_stop_reason']})")
                    print(f"    Strategies: {comparison['sc_strategies']}, Attempts: {comparison['sc_attempt_count']}")

        bc = comparison.get("baseline_correctness", "N/A")
        sc = comparison.get("sc_correctness", "N/A")
        comparison["answer_change"] = classify_answer_change(bc, sc)
        comparison["recovery_executed"] = comparison.get("sc_attempt_count", 0) > 0
        comparison["recovery_help_classification"] = classify_recovery(bc, comparison)
        comparison["notes"] = qdef.get("notes", "")

        all_comparisons.append(comparison)
        print(f"  Change: {comparison['answer_change']}, Recovery: {comparison['recovery_help_classification']}")
        print()

    aggregates = compute_aggregates(all_comparisons)

    results_path = run_dir / "benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": f"run_{run_number:03d}",
            "queries": all_comparisons,
            "aggregates": aggregates,
        }, f, indent=2, ensure_ascii=False)

    csv_path = run_dir / "benchmark_results.csv"
    save_csv(all_comparisons, csv_path)

    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(json.dumps(aggregates, indent=2))
    print()
    print(f"Results saved to: {results_path}")
    print(f"CSV saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
