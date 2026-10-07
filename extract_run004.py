import json
import glob
import os

base = 'test_output/controlled_manual/final_benchmark/run_004'
files = sorted(glob.glob(os.path.join(base, '*', '*', 'benchmark_result.json')))
print(f"{len(files)} files")
for p in files:
    parts = p.replace('/', '\\').split('\\')
    qid, cfg = parts[-3], parts[-2]
    with open(p) as f:
        d = json.load(f)
    if 'error_type' in d:
        print(f"{qid} | {cfg} | ERROR {d['error_type']} | rt={d['runtime_seconds']}")
        continue
    line = (f"{qid} | {cfg} | rt={d['runtime_seconds']} | tok={d['total_tokens']} "
            f"(p={d['prompt_tokens']},c={d['completion_tokens']}) | nodes={d['retrieved_node_ids']}")
    if 'self_correcting_initial_status' in d:
        line += (f" | SC {d['self_correcting_initial_status']}/{d['self_correcting_initial_diagnosis']}"
                 f" -> {d['self_correcting_final_status']} stop={d['stop_reason']}"
                 f" att={d['attempt_count']} strat={d['strategies_executed']}"
                 f" rel={d.get('initial_relevance_score')} cov={d.get('initial_coverage_score')}"
                 f" sup={d.get('initial_support_score')} contra={d.get('initial_contradiction_score')}")
    line += f" | ans={d['final_answer'][:80]!r}"
    print(line)
