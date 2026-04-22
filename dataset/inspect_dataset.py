import json
from collections import defaultdict
import statistics


def process_and_generate_examples(filepath):
    with open(filepath, 'r') as f:
        ds = []
        for line in f:
            data = json.loads(line)

            processed_rules = []
            if data.get('rules'):
                for rule in data['rules']:
                    if isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], list):
                        processed_rules.append({"premises": rule[0], "conclusion": rule[1]})
                    else:
                        raise ValueError("Malformed rule structure")
            data['rules'] = processed_rules
            ds.append(data)
    return ds


path = "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced.jsonl"
ds = process_and_generate_examples(path)

stats = defaultdict(lambda: {'count': 0, 'max_preds': 0, 'pred_counts': [], 'seq_lengths': [], 'min_premises': 9000, 'max_premises': 0, 'max_rules': 0, 'sum_rules': 0})

for d in ds:
    depth = d['depth']
    label = d['label']
    # key = (depth, label)
    key = (depth, -1)
    stats[key]['count'] += 1

    pred_len = len(d['preds'])
    stats[key]['pred_counts'].append(pred_len)
    stats[key]['max_preds'] = max(stats[key]['max_preds'], pred_len)
    stats[key]['max_rules'] = max(stats[key]['max_rules'], len(d['rules']))
    stats[key]['sum_rules'] += len(d['rules'])
    if len(d['rules']) > 0:
        stats[key]['min_premises'] = min(stats[key]['min_premises'], min([len(x['premises']) for x in d['rules']]))
        stats[key]['max_premises'] = max(stats[key]['max_premises'], max([len(x['premises']) for x in d['rules']]))
    seq_len = sum([1 + len(rule['premises']) for rule in d['rules']]) + len(d['facts']) + 2
    stats[key]['seq_lengths'].append(seq_len)

for key, data in stats.items():
    if data['count'] > 0:
        data['avg_rules'] = data['sum_rules'] / data['count']
        data['preds_avg_pm_std'] = f"{statistics.mean(data['pred_counts']):.2f} ± {statistics.stdev(data['pred_counts']):.2f}"
        data['seq_len_avg_pm_std'] = f"{statistics.mean(data['seq_lengths']):.2f} ± {statistics.stdev(data['seq_lengths']):.2f}"
        del data['pred_counts']
        del data['seq_lengths']

print(f"{'Depth':<10} | {'Label':<10} | Data")
print("-" * 120)
sorted_stats = sorted(stats.items(), key=lambda x: (str(x[0][0]), str(x[0][1])))
for (depth, label), data in sorted_stats:
    print(f"{str(depth):<10} | {str(label):<10} | {str(data)}")

# mainly for confirming data generators worked as hoped
