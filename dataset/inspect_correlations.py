from heuristics import counterfactual_heuristics, is_reachable, calculate_depths
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset
from processor import revert_rules_format


def _count_paths_recursive(predicate, known_facts, rules_by_conclusion, memo):
    """Recursive helper for counting proof paths with cycle detection."""
    # Base case 1: The predicate is a known fact.
    if predicate in known_facts:
        return 1

    # Base case 2: We have already computed the result for this predicate.
    if predicate in memo:
        # If the value is -1, we are currently in the process of computing it (cycle detected).
        # A cycle means this path cannot be grounded in facts, so it contributes 0 proofs.
        if memo[predicate] == -1:
            return 0
        return memo[predicate]

    memo[predicate] = -1
    total_paths = 0
    if predicate in rules_by_conclusion:
        for premises, _ in rules_by_conclusion[predicate]:
            paths_for_this_rule = 1
            possible = True
            for premise in premises:
                premise_paths = _count_paths_recursive(premise, known_facts, rules_by_conclusion, memo)
                if premise_paths == 0:
                    possible = False
                    break
                paths_for_this_rule *= premise_paths

            if possible:
                total_paths += paths_for_this_rule

    memo[predicate] = total_paths
    return total_paths


def count_proof_paths(facts, rules, query):
    """
    Calculates the number of unique proof paths (proof redundancy).
    Handles cycles and complex dependencies using dynamic programming (memoization).
    """
    if not is_reachable(facts, rules, query):
        return 0

    known_facts = set(facts)
    rules_by_conclusion = defaultdict(list)
    for premises, conclusion in rules:
        rules_by_conclusion[conclusion].append((premises, conclusion))

    memo = {}
    return _count_paths_recursive(query, known_facts, rules_by_conclusion, memo)


def backward_chain_(u, depth, rules, facts, max_depth, ances):
    INF = 100000000
    if u in facts:
        return INF
    if u in ances or depth == max_depth:
        return depth

    res = depth
    for rule in [x for x in rules if x[1] == u]:
        head, _ = rule
        tmp = INF
        for lit in head:
            ances.add(u)
            tmp = min(tmp, backward_chain_(lit, depth + 1, rules, facts, max_depth, ances))
            ances.remove(u)
        res = max(res, tmp)
    return res


def backward_chain(query, rules, facts, max_depth):
    return backward_chain_(query, 0, rules, facts, max_depth, set())


def get_backward_chaining_depth(facts, rules, query):
    """
    Calculates the depth after which 'query' gets a final,
    unchangeable provability status (either provable or unprovable).
    """
    fact_set = set(facts)
    provable_depths = calculate_depths(fact_set, rules)

    if query in provable_depths:
        return provable_depths[query]

    return backward_chain(query, rules, fact_set, 9)


def calculate_features(example):
    rules, facts, query = example["rules"], example["facts"], example["query"]
    num_rules = len(rules)
    num_facts = len(facts)
    ratio_rules_facts = num_rules / (num_facts + 1e-8)

    all_predicates_in_rules = set()
    total_rule_lengths = 0
    for head, tail in rules:
        all_predicates_in_rules.update(head)
        all_predicates_in_rules.add(tail)
        total_rule_lengths += len(head) + 1
    num_distinct_predicates_in_rules = len(all_predicates_in_rules)
    num_distinct_predicates_total = len(all_predicates_in_rules.union(set(facts)))
    total_predicates_in_rule_heads = sum(len(head) for head, tail in rules)

    query_as_rule_tail_count = sum(1 for head, tail in rules if tail == query)
    query_in_rule_head_count = sum(1 for head, tail in rules if query in head)
    query_total_occurrences = facts.count(query) + sum(head.count(query) for head, tail in rules) + query_as_rule_tail_count
    avg_rule_head_len = total_predicates_in_rule_heads / (num_rules + 1e-8)

    bf_numerator = num_facts + total_rule_lengths
    bf_denominator = num_facts + num_rules
    branching_factor = bf_numerator / (bf_denominator + 1e-8)
    label = is_reachable(facts, rules, query)

    return {
        'depth': example.get("depth", -1),
        'num_rules': num_rules,
        'num_facts': num_facts,
        'ratio_rules_facts': ratio_rules_facts,
        'num_distinct_predicates_rules': num_distinct_predicates_in_rules,
        'num_distinct_predicates_total': num_distinct_predicates_total,
        'query_as_rule_tail_count': query_as_rule_tail_count,
        'query_in_rule_head_count': query_in_rule_head_count,
        'query_total_occurrences': query_total_occurrences,
        'avg_rule_head_len': avg_rule_head_len,
        'branching_factor': branching_factor,
        'label': label
    }


def get_correlation_with_label(df):
    if df.empty or 'label' not in df.columns or len(df) < 2:
        return pd.Series(dtype='float64')

    numeric_df = df.select_dtypes(include=np.number)
    if 'label' not in numeric_df.columns:
        return pd.Series(dtype='float64')

    return numeric_df.corr()['label']


def analyze_correlations(examples_list, title_prefix, depths_to_analyze):
    print(f"\n\n{'#' * 50}\n# Correlation Analysis: {title_prefix}\n{'#' * 50}")
    results_data = [calculate_features(ex) for ex in examples_list]
    main_df = pd.DataFrame(results_data)
    correlation_dict = {}
    for depth in sorted(depths_to_analyze):
        depth_df = main_df[main_df['depth'] == depth]
        correlation_series = get_correlation_with_label(depth_df)
        correlation_dict[f'Depth {depth}'] = correlation_series

    overall_correlation_series = get_correlation_with_label(main_df)
    correlation_dict['Overall'] = overall_correlation_series
    final_table = pd.DataFrame(correlation_dict).sort_values(by='Overall', ascending=False)
    rows_to_drop = ['label', 'depth']
    final_table.drop(index=rows_to_drop, inplace=True, errors='ignore')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(final_table.to_string(float_format='{:.3f}'.format))


def analyze_structure(examples):
    """Calculates structural metrics for a list of examples."""
    results = []
    for ex in examples:
        facts, rules, query = ex["facts"], ex["rules"], ex["query"]
        results.append({
            'proof_redundancy': count_proof_paths(facts, rules, query),
            'proof_depth': calculate_depths(set(facts), rules).get(query, np.inf),
            'problem_depth': max(calculate_depths(set(facts), rules).values()),
            'backward_chaining_depth': get_backward_chaining_depth(set(facts), rules, query),
            'num_rules': len(rules),
            'num_facts': len(facts),
            'label': ex['label'],
            'depth': ex['depth']
        })
    return pd.DataFrame(results)


def run_structural_analysis():
    print("Starting structural analysis for LP and RP datasets...")
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    lp_examples = [revert_rules_format(x) for x in full_dataset_dict["validation_lp"]]
    rp_examples = [revert_rules_format(x) for x in full_dataset_dict["validation_rp_balanced"]]

    # Generate augmented datasets
    print("Generating r2-augmented examples for LP...")
    lp_r2_heuristics = counterfactual_heuristics(lp_examples, ["r2"], balance=True, preserve_depth=False)
    print("\nGenerating r2-augmented examples for RP...")
    rp_r2_heuristics = counterfactual_heuristics(rp_examples, ["r2"], balance=True, preserve_depth=True)

    all_datasets = {
        ('LP dataset', 'Original'): analyze_structure(lp_examples),
        ('LP dataset', 'r2'): analyze_structure(lp_r2_heuristics),
        ('RP dataset', 'Original'): analyze_structure(rp_examples),
        ('RP dataset', 'r2'): analyze_structure(rp_r2_heuristics),
    }

    depth_data = {}
    for label_val in [0, 1]:
        label_str = 'Label 0 (False)' if label_val == 0 else 'Label 1 (True)'
        for (dset_type, version), df in all_datasets.items():

            df_filtered = df[df['label'] == label_val]
            if df_filtered.empty:
                continue
            for metric in ['problem_depth']:
                counts = df_filtered[metric].value_counts().sort_index()
                counts = counts[~np.isinf(counts.index)]

                if not counts.empty:
                    nine_plus_sum = counts[counts.index >= 9].sum()
                    counts = counts[counts.index < 9]
                    if nine_plus_sum > 0:
                        counts.loc[9] = nine_plus_sum
                metric_name = 'Problem depth'
                depth_data[(metric_name, label_str, dset_type, version)] = counts
    depth_df = pd.DataFrame(depth_data)

    old_col_order_tuples = [
        ('Problem depth', 'LP dataset', 'Original'), ('Problem depth', 'LP dataset', 'r2'),
        ('Problem depth', 'RP dataset', 'Original'), ('Problem depth', 'RP dataset', 'r2'),
    ]

    new_col_order = []
    for metric, dset, version in old_col_order_tuples:
        key0 = (metric, 'Label 0 (False)', dset, version)
        key1 = (metric, 'Label 1 (True)', dset, version)
        if key0 in depth_df.columns:
            new_col_order.append(key0)
        if key1 in depth_df.columns:
            new_col_order.append(key1)

    depth_df = depth_df.reindex(columns=new_col_order)
    depth_df = depth_df.sort_index()
    depth_df = depth_df.rename(index={9: '9+'})
    depth_df_display = depth_df.fillna('--').astype(str).replace(r'\.0$', '', regex=True)
    depth_df_display.index.name = "Depth"

    print("\n\n" + "=" * 80)
    print(" " * 20 + "Depth distribution across datasets")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 100000)
    print(depth_df_display)
    print("=" * 80)


def compare_depth_metrics(distribution="rp", heuristics=False):
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    rp_examples = [revert_rules_format(x) for x in full_dataset_dict[f"validation_{distribution}"]]
    if heuristics:
        rp_examples.extend(counterfactual_heuristics(rp_examples, ["r2"], balance=True, preserve_depth=True))
    rp_examples = [x for x in rp_examples if x["label"] == 0]
    analysis_df = analyze_structure(rp_examples)

    problem_depth_all_counts = analysis_df['problem_depth'].value_counts()
    backward_chaining_depth_all_counts = analysis_df['backward_chaining_depth'].value_counts()

    bins_0_to_8 = range(9)
    problem_depth_counts = problem_depth_all_counts.reindex(bins_0_to_8, fill_value=0).sort_index()
    settling_depth_counts = backward_chaining_depth_all_counts.reindex(bins_0_to_8, fill_value=0).sort_index()
    problem_depth_9plus = problem_depth_all_counts[problem_depth_all_counts.index >= 9].sum()
    settling_depth_9plus = backward_chaining_depth_all_counts[backward_chaining_depth_all_counts.index >= 9].sum()
    problem_depth_counts.loc[9] = problem_depth_9plus
    settling_depth_counts.loc[9] = settling_depth_9plus

    x_labels = [str(i) for i in bins_0_to_8] + ['9+']
    x = np.arange(len(x_labels))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(4, 2))
    width = 0.35
    rects1 = ax.bar(x - width / 2, problem_depth_counts, width, label='Forward-chaining', color='skyblue')
    rects2 = ax.bar(x + width / 2, settling_depth_counts, width, label='Backward-chaining', color='salmon')

    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Depth', fontsize=12)
    # ax.set_title('Problem depth vs. proof settling depth')
    ax.set_xticks(x)
    ax.set_ylim(0, 2800)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=12)
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig(f"problem_vs_proof_depth_{distribution}.pdf", format='pdf', bbox_inches='tight')


# # Appendix: Mitigating superficial statistical cues
# depths = [0, 1, 2, 3, 4, 5, 6]
#
# full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
# data = [revert_rules_format(x) for x in full_dataset_dict["validation_rp_balanced"]]
# heuristics = counterfactual_heuristics(data, ["r2"], balance=True, algo="RP")
# analyze_correlations(data, "original RP balanced", depths)
# analyze_correlations(heuristics, "heuristics RP balanced", depths)
# analyze_correlations(data + heuristics, "original + heuristics RP balanced", depths)
#
# full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
# data = [revert_rules_format(x) for x in full_dataset_dict["validation_lp"]]
# heuristics = counterfactual_heuristics(data, ["r2"], balance=True, algo="LP")
# analyze_correlations(data, "original LP", depths)
# analyze_correlations(heuristics, "heuristics LP", depths)
# analyze_correlations(data + heuristics, "original + heuristics LP", depths)

# Problem depth and proof settling depth
#
# compare_depth_metrics("rp")
# compare_depth_metrics("rp_balanced")
# compare_depth_metrics("lp")
# compare_depth_metrics("lp_balanced")
