from scipy.optimize import fsolve, minimize
import numpy as np


def scores_for_depth(raw_scores, target_depth):
    scores = {}
    for key, accs in raw_scores.items():
        dataset = key[0]
        config_tags = frozenset(key[2:])
        if config_tags not in scores:
            scores[config_tags] = {}
        if target_depth in accs:
            scores[config_tags][dataset] = accs[target_depth]
    return scores


# =====================================================================
# Core functions
# =====================================================================
def recovery_complexity(s, n, head_dim):
    """
    m = C * s log n/s best models empirical results
    C = (s log n/s) / m
    we assume C is roughly constant across models, serving as a 'good enough' empirical heuristic
    :return C
    alternatives from https://arxiv.org/abs/2602.11246
    """
    return (s * np.log(n / s)) / head_dim
    # return (s ** 2 / np.log(s) * np.log(n / s)) / head_dim
    # return s ** 2 * np.log(n) / head_dim


def solve_s(target_val, n, head_dim, guess=2):
    def equation(x):
        return recovery_complexity(x, n, head_dim) - target_val

    return fsolve(equation, guess)[0]


def get_dim_and_heads(tags):
    dim = 1
    heads = 1
    for tag in tags:
        if isinstance(tag, str):
            if tag.startswith('dim='):
                dim = int(tag.split('=')[1])
            elif tag.startswith('heads='):
                heads = int(tag.split('=')[1])
    return dim, heads


def get_single_head_acc(total_acc, heads):
    error_rate = max(0.0, 1.0 - total_acc)
    if error_rate == 0.0:
        return 1.0
    return 1.0 - np.power(error_rate, 1.0 / heads)


def log_acc(a):
    return -np.log(np.clip(a, 1e-10, 1.0 - 1e-10))


def generate_latex_cross_val_table(scores, dataset_s_map, n_fixed, c1, c2):
    eval_targets = [
        'validation_rp_balanced_1_premise',
        'validation_rp_balanced_1_2_premise',
        'validation_rp_balanced',
        'validation_rp_balanced_2_premise',
        'validation_rp_balanced_2_3_premise',
        'validation_rp_balanced_3_premise'
    ]

    header_names = []
    for ds in eval_targets:
        name = ds.replace('validation_rp_balanced', 'RP').replace('_premise', '').replace('_', '-')
        if name == 'RP': name = 'RP 1.5'
        header_names.append(name.strip('- '))

    latex = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Cross-validated predicted active feature counts ($s$) anchored to different base complexities. Using exponential decay capacity limits.}",
        "\\label{tab:cross_validation_s}",
        "\\begin{tabular}{l" + "c" * len(eval_targets) + "}",
        "\\toprule",
        "\\textbf{Model Configuration} & " + " & ".join([f"\\textbf{{{h}}}" for h in header_names]) + " \\\\",
    ]

    for base_dataset, base_s in dataset_s_map.items():
        anchor_name = base_dataset.replace('validation_rp_balanced', 'RP').replace('_premise', '').replace('_', '-')
        if anchor_name == 'RP': anchor_name = 'RP 1.5'
        anchor_name = anchor_name.strip('- ')

        latex.extend([
            "\\midrule",
            f"\\multicolumn{{{len(eval_targets) + 1}}}{{c}}{{\\textbf{{Anchor: {anchor_name} (Assumed $s={base_s:.1f}$)}}}} \\\\",
            "\\midrule"
        ])

        for tags, datasets in scores.items():
            if base_dataset not in datasets:
                continue
            dim, heads = get_dim_and_heads(tags)
            head_dim = dim / heads
            target_base = recovery_complexity(c1 * base_s + c2, n_fixed, head_dim)
            log_a_base = log_acc(get_single_head_acc(datasets[base_dataset], heads))

            tag_str = ", ".join(sorted([t for t in tags if t not in ['rp', 'r2', 'width', 'corrective', 'bidir']]))
            safe_tag_str = tag_str.replace('_', '\\_')
            row_vals = [safe_tag_str]
            for ds in eval_targets:
                if ds not in datasets:
                    row_vals.append("—")
                    continue
                if ds == base_dataset:
                    row_vals.append(f"\\textbf{{{base_s:.2f}}}")
                    continue

                acc = datasets[ds]
                a_target = get_single_head_acc(acc, heads)
                log_a_target = log_acc(a_target)
                target_val = target_base * (log_a_target / log_a_base)
                predicted_s_eff = solve_s(target_val, n_fixed, head_dim)
                predicted_s = (predicted_s_eff - c2) / c1
                row_vals.append(f"{predicted_s:.2f}")
            latex.append(" & ".join(row_vals) + " \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    return "\n".join(latex)


def calculate_error_for_fixed_c(n_guess_array, scores, dataset_s_map, fixed_c1, fixed_c2):
    n_guess = n_guess_array[0]
    total_squared_error = 0.0
    predictions_count = 0
    for base_dataset, base_s in dataset_s_map.items():
        for tags, datasets in scores.items():
            if base_dataset not in datasets:
                continue

            dim, heads = get_dim_and_heads(tags)
            head_dim = dim / heads
            target_base = recovery_complexity(fixed_c1 * base_s + fixed_c2, n_guess, head_dim)
            log_a_base = log_acc(get_single_head_acc(datasets[base_dataset], heads))
            for target_ds, true_s in dataset_s_map.items():
                if target_ds not in datasets or target_ds == base_dataset:
                    continue
                acc_target = datasets[target_ds]
                a_target = get_single_head_acc(acc_target, heads)
                target_val = target_base * (log_acc(a_target) / log_a_base)
                predicted_s_eff = solve_s(target_val, n_guess, head_dim, guess=fixed_c1 * base_s + fixed_c2)
                predicted_s = (predicted_s_eff - fixed_c2) / fixed_c1
                total_squared_error += (predicted_s - true_s) ** 2
                predictions_count += 1
    return total_squared_error / predictions_count


def find_optimal_parameters(scores, dataset_s_map):
    print("Scanning C1, C2, and N...")
    c1_candidates = [1, 2, 3, 4, 5]
    c2_candidates = [0, 1, 2, 3, 4, 5]
    for c1 in c1_candidates:
        for c2 in c2_candidates:
            result = minimize(calculate_error_for_fixed_c, x0=[100.0], args=(scores, dataset_s_map, c1, c2), bounds=[(10.0, 500000.0)], method='L-BFGS-B')
            print(f"Testing C1 = {c1}, C2 = {c2} | Optimal N = {result.x[0]:8.2f} | Error = {result.fun:.4f}")
