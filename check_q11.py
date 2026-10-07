import json

for cfg in ['baseline_r1', 'sc_r1']:
    d = json.load(open(f'test_output/controlled_manual/final_benchmark/run_004/Q11/{cfg}/benchmark_result.json'))
    a = d['final_answer'].upper()
    print(cfg, '| AX-4:', 'AX-4' in a, '| AX-3:', 'AX-3' in a)

r3 = json.load(open('test_output/controlled_manual/final_benchmark/run_003/benchmark_results.json'))
q11 = [q for q in r3['queries'] if q['query_id'] == 'Q11'][0]
print('run_003 Q11:', q11['baseline_correctness'], q11['sc_correctness'],
      '| b_nodes:', len(q11['baseline_node_ids']),
      '| sc_final:', len(q11['sc_final_node_ids']),
      '| sc_stop:', q11['sc_stop_reason'], q11['sc_attempt_count'])
