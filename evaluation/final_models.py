import math

from conf import *
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy import stats
from dataset.processor import pad, train_curriculum, special_tokens, prepare_ds_inference, process
from model.type_llama_no_ffn import ModelArgs, Transformer
from dataset.data_preprocessing import pad_collate
from dataset.eval import eval_model
from datasets import load_dataset
from torch.utils.data import DataLoader

from model_repository import ModelRepository
from training.train_loop import load_weights_and_init

def get_means(scores_dict, key):
    if key not in scores_dict:
        return np.nan, np.nan
    data = scores_dict[key]
    vals_6 = [data[d] for d in range(7) if d in data]
    mean_6 = np.mean(vals_6) * 100 if vals_6 else np.nan
    vals_12 = [data[d] for d in range(7, 13) if d in data]
    mean_12 = np.mean(vals_12) * 100 if vals_12 else np.nan
    return mean_6, mean_12


def parse_key_metadata(key):
    eval_domain = key[0].upper()
    is_cot = key[1]
    remainder = [str(x).lower() for x in key[2:]]
    known_domains = {'lp', 'rp'}
    train_domain = "UNKNOWN"
    for item in reversed(remainder):
        if item in known_domains:
            train_domain = item.upper()
            break
    method_group = 'cot' if is_cot else 'direct'

    # Priority tags for consistent naming, but we capture ALL tags now
    # We reconstruct the label based on the actual components found in the key
    # to ensure unique labels for every run configuration.

    standard_tags = ['corrective', 'r2', 'bidir', 'ffn', 'universal']
    found_tags = []
    for tag in standard_tags:
        if tag in remainder:
            found_tags.append(tag)
    ignore_set = set(known_domains) | {'cot', 'direct'} | set(standard_tags) | {x.lower() for x in known_domains}
    extras = [x for x in remainder if x not in ignore_set]
    all_tags = found_tags + extras
    label = f"{method_group} w. " + "+".join(sorted(all_tags))
    return eval_domain, train_domain, method_group, label


def process_and_pivot(scores_30, scores_60):
    all_keys = set(scores_30.keys()) | set(scores_60.keys())
    processed_data = []

    for key in all_keys:
        eval_domain, train_domain, method, label = parse_key_metadata(key)
        acc_30_6, acc_30_12 = get_means(scores_30, key)
        acc_60_6, acc_60_12 = get_means(scores_60, key)

        processed_data.append({
            'method': method,
            'train_domain': train_domain,
            'label': label,
            'tags': frozenset(sorted([method] + list(key[2:]))),
            'eval_domain': eval_domain,
            'acc_30_6': acc_30_6,
            'acc_30_12': acc_30_12,
            'acc_60_6': acc_60_6,
            'acc_60_12': acc_60_12
        })

    df = pd.DataFrame(processed_data)
    pivot_df = df.pivot_table(
        index=['method', 'train_domain', 'label', 'tags'],
        columns=['eval_domain'],
        values=['acc_30_6', 'acc_30_12', 'acc_60_6', 'acc_60_12']
    )
    pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
    pivot_df = pivot_df.sort_index(axis=1)

    desired_domains = ['LP', 'LP_STAR', 'RP']
    present_domains = [d for d in desired_domains if d in pivot_df.columns.get_level_values(0)]
    metric_order = ['acc_30_6', 'acc_30_12', 'acc_60_6', 'acc_60_12']
    new_columns = pd.MultiIndex.from_product([present_domains, metric_order])
    pivot_df = pivot_df.reindex(columns=new_columns)

    return pivot_df, present_domains


def get_sort_key(label):
    score = 0
    if 'direct' in label: score += 0
    if 'cot' in label: score += 1000
    parts = label[1]
    score += len(parts) * 10
    if 'r2' in label: score += 1
    if 'corrective' in label: score += 2
    if 'ffn' in label: score += 50
    if 'bidir' in label: score += 60
    if 'universal' in label: score += 500
    return score


def generate_latex_table_str(sub_df, caption, label_tag, present_domains, highlight_label=None):
    col_def = "l" + ("cccc" * len(present_domains))

    header_1 = "\\toprule\n& "
    cmid_1 = ""
    start_idx = 2
    for domain in present_domains:
        clean_domain = domain.replace('_STAR', '*')
        header_1 += f"\\multicolumn{{4}}{{c}}{{{clean_domain}}} & "
        cmid_1 += f"\\cmidrule(lr){{{start_idx}-{start_idx + 3}}} "
        start_idx += 4
    header_1 = header_1.rstrip(" & ") + " \\\\"

    header_2 = "& "
    cmid_2 = ""
    start_idx = 2
    for _ in present_domains:
        header_2 += "\\multicolumn{2}{c}{30 Preds} & \\multicolumn{2}{c}{60 Preds} & "
        cmid_2 += f"\\cmidrule(lr){{{start_idx}-{start_idx + 1}}} \\cmidrule(lr){{{start_idx + 2}-{start_idx + 3}}} "
        start_idx += 4
    header_2 = header_2.rstrip(" & ") + " \\\\"

    header_3 = "Model & "
    for _ in present_domains:
        header_3 += "$\\le 6$ & $\\le 12$ & $\\le 6$ & $\\le 12$ & "
    header_3 = header_3.rstrip(" & ") + " \\\\"

    def get_mean(x):
        if isinstance(x, str) and "{" in x:
            x = x.split("{")[-1]
        if isinstance(x, str) and '\\pm' in x:
            return float(x.split('\\pm')[0])
        return float(x)

    max_means = {}
    for col in sub_df.columns:
        max_means[col] = sub_df[col].apply(get_mean).max()

    data_rows = ""
    for label, row in sub_df.iterrows():
        row_str = f"{label[0]} "
        for domain in present_domains:
            metrics = ['acc_30_6', 'acc_30_12', 'acc_60_6', 'acc_60_12']
            for m in metrics:
                val = row[(domain, m)]
                col_max = max_means[(domain, m)]
                val_mean = get_mean(val)
                is_max = (val_mean == col_max) and (val_mean != -float('inf'))

                if pd.isna(val) or val == '-':
                    row_str += "& - "

                elif isinstance(val, str):
                    mean_part, std_part = val.split('\\pm')
                    mean_part = mean_part.strip()
                    std_part = std_part.strip()
                    if highlight_label and is_max:
                        formatted_cell = f"\\textbf{{{mean_part}}} \\pm {std_part}"
                    else:
                        formatted_cell = f"{mean_part} \\pm {std_part}"
                    row_str += f"& ${formatted_cell}$ "
                else:
                    if is_max:
                        row_str += f"& \\textbf{{{val:.1f}}} "
                    else:
                        row_str += f"& {val:.1f} "

        data_rows += row_str + "\\\\\n"

    return (
        "\\begin{table}[ht!]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label_tag}}}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"\\begin{{tabular}}{{{col_def}}}\n"
        f"{header_1}\n{cmid_1}\n"
        f"{header_2}\n{cmid_2}\n"
        f"{header_3}\n"
        "\\midrule\n"
        f"{data_rows}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}"
    )


def generate_main_tables(scores_30, scores_60, methods=['direct', 'cot']):
    pivot_df, present_domains = process_and_pivot(scores_30, scores_60)
    latex_tables = []

    for method in methods:
        for train_domain in ['RP']:
            sub_df = pivot_df.xs((method, train_domain), level=['method', 'train_domain'])
            sorted_index = sorted(sub_df.index, key=get_sort_key)
            sub_df = sub_df.reindex(sorted_index)
            caption = (f"Main Results: {method.upper()} model trained on {train_domain}. "
                       f"Best column-wise results are bolded.")
            label_tag = f"tab:main_{method}_{train_domain.lower()}"
            latex_tables.append(generate_latex_table_str(sub_df, caption, label_tag, present_domains))

    return "\n\n".join(latex_tables)

def filter_scaling_scores(raw_scores, take_layers=None):
    filtered = {}
    for key, vals in raw_scores.items():
        remainder = [str(x).lower() for x in key[2:]]
        has_scaling = 'scaling' in remainder
        has_layers = any(x.startswith('layers=') for x in remainder)
        has_heads = any(x.startswith('heads=') for x in remainder)
        if has_scaling and not has_layers or has_heads:
            continue
        if take_layers is not None and has_layers:
            layers_val = int(next((x for x in remainder if x.startswith('layers=')), None).split('=')[1])
            if layers_val not in take_layers:
                continue
        new_key = list(key)
        if not has_scaling:
            new_key.append('layers=8')
        filtered[tuple(new_key)] = vals
    return filtered

def evaluate_gap_closure(scores, margin=0.01):
    pairs = {}
    for key, length_accs in scores.items():
        eval_domain = key[0]
        is_cot = key[1]
        config_tags = frozenset([str(x).lower() for x in key[2:]])
        pair_key = (eval_domain, config_tags)
        if pair_key not in pairs:
            pairs[pair_key] = {}
        pairs[pair_key]['cot' if is_cot else 'direct'] = length_accs

    results = []
    for (eval_domain, tags), methods in pairs.items():
        if 'cot' in methods and 'direct' in methods:
            cot_accs = methods['cot']
            dir_accs = methods['direct']
            layer_tag = next((t for t in tags if t.startswith('layers=')), 'layers=8')
            layer_count = int(layer_tag.split('=')[1])
            for depth_limit, depth_label in [(6, '<=6'), (12, '<=12')]:
                common_lengths = sorted([l for l in set(cot_accs.keys()) & set(dir_accs.keys()) if l <= depth_limit])
                if len(common_lengths) < 2:
                    continue
                cot_array = [cot_accs[l] for l in common_lengths]
                dir_array = [dir_accs[l] for l in common_lengths]
                diffs = np.array(cot_array) - np.array(dir_array)
                is_closed = bool(np.max(diffs) <= margin)
                results.append({
                    'eval_domain': eval_domain.upper(),
                    'layers': layer_count,
                    'depth_group': depth_label,
                    'gap_closed': is_closed,
                    'tags': ", ".join(tags)
                })

    df_results = pd.DataFrame(results)
    return df_results

def generate_gap_summary_table(gap_df_30, gap_df_60):
    def get_mins(df):
        results = {}
        for (domain, depth), group in df.groupby(['eval_domain', 'depth_group']):
            group = group.sort_values('layers')
            sustained_min = "—"
            layers = group['layers'].values
            closed = group['gap_closed'].values
            # Check the current model layer.
            # If it, AND all deeper evaluated models are closed, we found our sustained minimum.
            for i in range(len(layers)):
                if all(closed[i:]):
                    sustained_min = str(layers[i])
                    break
            results[(domain, depth)] = sustained_min
        return results

    mins_30 = get_mins(gap_df_30)
    mins_60 = get_mins(gap_df_60)
    domains = ['LP', 'LP_STAR', 'RP']
    latex = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Minimum layers required for Direct model performance to sustain strict non-inferiority (within 1\\% margin) compared to CoT. The reported layer count indicates the point at which the gap closes and remains closed for all subsequently evaluated model depths.}",
        "\\label{tab:gap_closure_summary}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{30 Preds}} & \\multicolumn{2}{c}{\\textbf{60 Preds}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "\\textbf{Eval Domain} & \\textbf{$\\le 6$} & \\textbf{$\\le 12$} & \\textbf{$\\le 6$} & \\textbf{$\\le 12$} \\\\",
        "\\midrule"
    ]

    for domain in domains:
        val_30_6 = mins_30.get((domain, '<=6'), "—")
        val_30_12 = mins_30.get((domain, '<=12'), "—")
        val_60_6 = mins_60.get((domain, '<=6'), "—")
        val_60_12 = mins_60.get((domain, '<=12'), "—")
        clean_domain = domain.replace('_STAR', '$^*$')
        latex.append(f"{clean_domain} & {val_30_6} & {val_30_12} & {val_60_6} & {val_60_12} \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex)

def generate_differential_summary(scores_30, scores_60):
    full_pivot, present_domains = process_and_pivot(scores_30, scores_60)
    rp_data = full_pivot.xs('RP', level='train_domain')
    target_tags = ['r2', 'corrective', 'ffn', 'bidir']
    def format_diff_row(diff_list, row_label, columns):
        raw_differences = np.array(diff_list)
        k_pairs = len(raw_differences)
        mean_diff = np.mean(raw_differences, axis=0)
        std_diff = np.std(raw_differences, axis=0, ddof=1) if k_pairs > 1 else np.zeros(raw_differences.shape[1])
        formatted_cells = []
        for m, s, col in zip(mean_diff, std_diff, columns):
            n = k_pairs
            t_stat = (m / (s / math.sqrt(n))) if s > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if n > 1 else 1.0
            cell_text = f"{m:.1f} \\pm {s:.1f}"
            if p_value < 0.05:
                if m > 0:
                    cell_text = f"\\textcolor{{green!70!black}}{{{cell_text}}}"
                elif m < 0:
                    cell_text = f"\\textcolor{{red}}{{{cell_text}}}"
            formatted_cells.append(cell_text)
        return pd.Series(formatted_cells, index=columns, name=(row_label,))

    all_latex_tables = []
    for method in ['direct', 'cot']:
        if method not in rp_data.index:
            continue

        method_data = rp_data.loc[method]
        method_rows = []
        for tag in target_tags:
            diffs = []
            for (_, current_tags), row_with in method_data.iterrows():
                if tag in current_tags:
                    mask_without = method_data.index.get_level_values('tags') == (current_tags - frozenset([tag]))
                    row_without = method_data[mask_without]
                    if not row_without.empty:
                        diffs.append((row_with.values - row_without.values[0], current_tags))

            if not diffs:
                continue

            if method == 'direct' and tag == 'r2':
                with_corr = [d[0] for d in diffs if 'corrective' in d[1]]
                wout_corr = [d[0] for d in diffs if 'corrective' not in d[1]]
                for d_list, label in [(with_corr, 'r2 (w/ corrective)'), (wout_corr, 'r2 (w/o corrective)')]:
                    row = format_diff_row(d_list, label, method_data.columns)
                    if row is not None: method_rows.append(row)
            else:
                all_tag_diffs = [d[0] for d in diffs]
                row = format_diff_row(all_tag_diffs, tag, method_data.columns)
                if row is not None: method_rows.append(row)

        if method_rows:
            summary_df = pd.DataFrame(method_rows)
            caption = (f"Marginal Performance Impact: {method.upper()} models trained on RP. "
                       f"Values denote the percentage point difference ($Avg_{{with}} - Avg_{{without}} \\pm Std_{{diff}}$). "
                       f"\\textcolor{{green!70!black}}{{Green}} indicates a statistically significant improvement, while \\textcolor{{red}}{{red}} indicates a significant degradation ($p < 0.05$).")
            table_str = generate_latex_table_str(summary_df, caption, f"tab:diff_rp_{method}", present_domains)
            all_latex_tables.append(table_str)
    return "\n\n".join(all_latex_tables)

def get_aggregated_stats(pivot_df):
    tags_idx = pivot_df.index.names.index('tags')
    new_idx_tuples = []
    for idx in pivot_df.index:
        idx_list = list(idx)
        clean_tags = frozenset(t for t in idx_list[tags_idx] if 'seed' not in str(t).lower())
        idx_list[tags_idx] = clean_tags
        new_idx_tuples.append(tuple(idx_list))
    agg_df = pivot_df.copy()
    agg_df.index = pd.MultiIndex.from_tuples(new_idx_tuples, names=pivot_df.index.names)
    grouped = agg_df.groupby(level=agg_df.index.names)
    return grouped.mean(), grouped.std(ddof=1).fillna(0)

def generate_universal_appendix(baseline_30, baseline_60, universal_30, universal_60):
    scores_30 = {**baseline_30, **universal_30}
    scores_60 = {**baseline_60, **universal_60}
    pivot_df, present_domains = process_and_pivot(scores_30, scores_60)
    pivot_df_clean = pivot_df.reset_index(level='label', drop=True)
    mean_df, std_df = get_aggregated_stats(pivot_df_clean)

    def format_diff_row(diff_list, row_label, columns):
        raw_differences = np.array(diff_list)
        k_pairs = len(raw_differences)
        mean_diff = np.mean(raw_differences, axis=0)
        std_diff = np.std(raw_differences, axis=0, ddof=1) if k_pairs > 1 else np.zeros(raw_differences.shape[1])
        formatted_cells = []
        df = k_pairs - 1
        critical_t = stats.t.ppf(1 - 0.05, df) if df > 0 else float('inf')
        for m, s, col in zip(mean_diff, std_diff, columns):
            cell_text = f"{m:.1f} \\pm {s:.1f}"
            t_stat = (m / (s / math.sqrt(k_pairs))) if s > 0 else 0
            if df > 0:
                if t_stat > critical_t:
                    cell_text = f"\\textcolor{{green!70!black}}{{{cell_text}}}"
                # elif t_stat < -critical_t:
                #     cell_text = f"\\textcolor{{red}}{{{cell_text}}}"
            formatted_cells.append(cell_text)
        return pd.Series(formatted_cells, index=columns, name=(row_label,))

    def get_seed_data(target_clean_idx):
        res = {}
        for idx, row in pivot_df.iterrows():
            idx_method, idx_domain, idx_label, idx_tags = idx
            clean_tags = frozenset(t for t in idx_tags if 'seed' not in str(t).lower())
            current_clean_idx = (idx_method, idx_domain, clean_tags)
            if current_clean_idx == target_clean_idx:
                import re
                match = re.search(r'seed(\d+)', str(idx_label).lower())
                seed = f"seed{match.group(1)}" if match else "no_seed"
                res[seed] = row.values
        return res

    def get_base_row(idx, name):
        m = mean_df.loc[idx]
        s = std_df.loc[idx]
        cells = [f"{mean_val:.1f} \\pm {std_val:.1f}" for mean_val, std_val in zip(m, s)]
        return pd.Series(cells, index=mean_df.columns, name=(name,))

    base_configs = set(idx[:2] for idx in mean_df.index)
    table1_rows = []
    table2_rows = []
    for config in sorted(base_configs):
        idx_dense_ffn = idx_dense_noffn = idx_univ_ffn = idx_univ_noffn = None
        for idx in mean_df.index:
            if idx[:2] == config:
                tags = idx[2]
                is_univ = 'universal' in tags
                has_ffn = 'ffn' in tags
                if is_univ and not has_ffn:
                    idx_univ_noffn = idx  # {'universal'}
                elif is_univ and has_ffn:
                    idx_univ_ffn = idx    # {'ffn', 'universal'}
                elif not is_univ and not has_ffn:
                    idx_dense_noffn = idx # {}
                elif not is_univ and has_ffn:
                    idx_dense_ffn = idx   # {'ffn'}

        config_str = f"{str(config[0]).lower()}"
        # --- TABLE 1: Impact of FFN (w. FFN vs w.o FFN) ---
        if idx_dense_ffn is not None and idx_dense_noffn is not None:
            table1_rows.append(get_base_row(idx_dense_ffn, f"{config_str} (baseline w. ffn)"))
            table1_rows.append(get_base_row(idx_dense_noffn, f"{config_str} (baseline w.o ffn)"))
            d_f, d_nf = get_seed_data(idx_dense_ffn), get_seed_data(idx_dense_noffn)
            common = set(d_f.keys()) & set(d_nf.keys())
            if common:
                diffs = [d_f[s] - d_nf[s] for s in common]
                table1_rows.append(format_diff_row(diffs, f"$\\Delta$ (w. ffn - w.o ffn)", pivot_df.columns))

        if idx_univ_ffn is not None and idx_univ_noffn is not None:
            table1_rows.append(get_base_row(idx_univ_ffn, f"{config_str} (universal w. ffn)"))
            table1_rows.append(get_base_row(idx_univ_noffn, f"{config_str} (universal w.o ffn)"))
            d_f, d_nf = get_seed_data(idx_univ_ffn), get_seed_data(idx_univ_noffn)
            common = set(d_f.keys()) & set(d_nf.keys())
            if common:
                diffs = [d_f[s] - d_nf[s] for s in common]
                table1_rows.append(format_diff_row(diffs, f"$\\Delta$ (w. FFN - w.o FFN)", pivot_df.columns))

        # --- TABLE 2: Architectural Impact (Universal vs Dense) ---
        if idx_dense_ffn is not None and idx_univ_ffn is not None:
            table2_rows.append(get_base_row(idx_dense_ffn, f"{config_str} (baseline w. ffn)"))
            table2_rows.append(get_base_row(idx_univ_ffn, f"{config_str} (universal w. ffn)"))
            d_d, d_u = get_seed_data(idx_dense_ffn), get_seed_data(idx_univ_ffn)
            common = set(d_d.keys()) & set(d_u.keys())
            if common:
                diffs = [d_u[s] - d_d[s] for s in common]
                table2_rows.append(format_diff_row(diffs, f"$\\Delta$ (universal w. ffn - baseline w. ffn)", pivot_df.columns))

        if idx_dense_noffn is not None and idx_univ_noffn is not None:
            table2_rows.append(get_base_row(idx_dense_noffn, f"{config_str} (baseline w.o ffn)"))
            table2_rows.append(get_base_row(idx_univ_noffn, f"{config_str} (universal w.o ffn)"))
            d_d, d_u = get_seed_data(idx_dense_noffn), get_seed_data(idx_univ_noffn)
            common = set(d_d.keys()) & set(d_u.keys())
            if common:
                diffs = [d_u[s] - d_d[s] for s in common]
                table2_rows.append(format_diff_row(diffs, f"$\\Delta$ (universal w.o ffn - baseline w.o ffn)", pivot_df.columns))

    df_t1 = pd.DataFrame(table1_rows) if table1_rows else pd.DataFrame(columns=pivot_df.columns)
    df_t2 = pd.DataFrame(table2_rows) if table2_rows else pd.DataFrame(columns=pivot_df.columns)
    latex1 = generate_latex_table_str(
        df_t1,
        caption="Impact of Feed-Forward Networks (FFN): w.o FFN vs w. FFN",
        label_tag="tab:ffn_ablation",
        present_domains=present_domains
    )
    latex2 = generate_latex_table_str(
        df_t2,
        caption="Architectural Impact: Universal (Looped) vs Dense Baseline",
        label_tag="tab:univ_ablation",
        present_domains=present_domains
    )
    return latex1 + "\n\n" + latex2

def generate_r2_ablation_tables(r2_impact_30, r2_impact_60):
    pivot_df, present_domains = process_and_pivot(r2_impact_30, r2_impact_60)
    pivot_df_clean = pivot_df.reset_index(level='label', drop=True)
    mean_df, std_df = get_aggregated_stats(pivot_df_clean)

    def format_diff_row(diff_list, row_label, columns):
        raw_differences = np.array(diff_list)
        k_pairs = len(raw_differences)
        mean_diff = np.mean(raw_differences, axis=0)
        std_diff = np.std(raw_differences, axis=0, ddof=1) if k_pairs > 1 else np.zeros(raw_differences.shape[1])
        formatted_cells = []
        df = k_pairs - 1
        critical_t = stats.t.ppf(1 - 0.05, df) if df > 0 else float('inf')
        for m, s, col in zip(mean_diff, std_diff, columns):
            cell_text = f"{m:.1f} \\pm {s:.1f}"
            t_stat = (m / (s / math.sqrt(k_pairs))) if s > 0 else 0
            if df > 0:
                if t_stat > critical_t:
                    cell_text = f"\\textcolor{{green!70!black}}{{{cell_text}}}"
            formatted_cells.append(cell_text)
        return pd.Series(formatted_cells, index=columns, name=(row_label,))

    def get_seed_data(target_clean_idx):
        res = {}
        for idx, row in pivot_df.iterrows():
            idx_method, idx_domain, idx_label, idx_tags = idx
            clean_tags = frozenset(t for t in idx_tags if 'seed' not in str(t).lower())
            current_clean_idx = (idx_method, idx_domain, clean_tags)
            if current_clean_idx == target_clean_idx:
                import re
                match = re.search(r'seed(\d+)', str(idx_label).lower())
                seed = f"seed{match.group(1)}" if match else "no_seed"
                res[seed] = row.values
        return res

    def get_base_row(idx, name):
        m = mean_df.loc[idx]
        s = std_df.loc[idx]
        cells = [f"{mean_val:.1f} \\pm {std_val:.1f}" for mean_val, std_val in zip(m, s)]
        return pd.Series(cells, index=mean_df.columns, name=(name,))

    latex_tables = []
    configs = {}
    for idx in mean_df.index:
        method, train_domain, tags = idx
        config_key = (method, train_domain)
        if config_key not in configs:
            configs[config_key] = {'nor2': None, 'r2half': None, 'baseline': None}

        if 'nor2' in tags:
            configs[config_key]['nor2'] = idx
        elif 'r2half' in tags:
            configs[config_key]['r2half'] = idx
        else:
            configs[config_key]['baseline'] = idx

    for (method, train_domain), models in configs.items():
        idx_nor2 = models['nor2']
        idx_r2half = models['r2half']
        idx_baseline = models['baseline']
        table_rows = [get_base_row(idx_nor2, "nor2"), get_base_row(idx_r2half, "r2half"), get_base_row(idx_baseline, "baseline")]

        d_nor2 = get_seed_data(idx_nor2)
        d_r2 = get_seed_data(idx_r2half)
        common_r2 = set(d_nor2.keys()) & set(d_r2.keys())
        diffs = [d_r2[s] - d_nor2[s] for s in common_r2]
        table_rows.append(format_diff_row(diffs, r"$\Delta$ (r2half - nor2)", pivot_df.columns))

        d_base = get_seed_data(idx_baseline)
        common_base = set(d_nor2.keys()) & set(d_base.keys())
        diffs = [d_base[s] - d_nor2[s] for s in common_base]
        table_rows.append(format_diff_row(diffs, r"$\Delta$ (baseline - nor2)", pivot_df.columns))

        df_table = pd.DataFrame(table_rows)
        caption = f"Impact of R2 Selection: {str(method).upper()} model on {train_domain}. One-tailed t-test (p < 0.05)."
        label_tag = f"tab:r2_ablation_{method}"
        latex_tables.append(generate_latex_table_str(df_table, caption, label_tag, present_domains))

    return "\n\n".join(latex_tables)


def process_grouped_scores(raw_scores, step_index=6, min_layers=0):
    grouped_data = {}
    for key, scores in raw_scores.items():
        dataset = key[0]
        method = 'corrective' if 'corrective' in key else 'direct'
        layers = 'layers=8'
        dim = 'dim=256'
        seed = 'base_seed'
        for item in key[2:]:
            if item.startswith('layers='):
                layers = item
            if item.startswith('dim='):
                dim = item
            if item.startswith('seed') and item != 'seed':
                seed = item
        layers = int(layers.split('=')[1])
        dim = int(dim.split('=')[1])
        if layers < min_layers:
            continue
        group_key = (dataset, layers, dim)
        if group_key not in grouped_data:
            grouped_data[group_key] = {'corrective': {}, 'direct': {}}
        grouped_data[group_key][method][seed] = scores[step_index]
    return grouped_data

synthesis_dataset_order = [
    "validation_lp_balanced",
    "validation_rp_balanced_1_premise",
    "validation_rp_balanced_1_2_premise",
    "validation_rp_balanced",
    "validation_rp_balanced_2_premise",
    "validation_rp_balanced_2_3_premise",
    "validation_rp_balanced_3_premise"
]

synthesis_dataset_header_mapping = {
    "validation_lp_balanced": "LP",
    "validation_rp_balanced_1_premise": "RP (exactly 1)",
    "validation_rp_balanced_1_2_premise": "RP (1 to 2)",
    "validation_rp_balanced": "RP (1 to 3)",
    "validation_rp_balanced_2_premise": "RP (exactly 2)",
    "validation_rp_balanced_2_3_premise": "RP (2 to 3)",
    "validation_rp_balanced_3_premise": "RP (exactly 3)"
}

premise_counts = {
    "validation_lp_balanced": "1.68",
    "validation_rp_balanced_1_premise": "1.00",
    "validation_rp_balanced_1_2_premise": "1.50",
    "validation_rp_balanced": "2.00",
    "validation_rp_balanced_2_premise": "2.00",
    "validation_rp_balanced_2_3_premise": "2.50",
    "validation_rp_balanced_3_premise": "3.00"
}

def print_corrective_stats(grouped_data):
    raw_data = {}
    for (dataset, layers, dim), metrics in grouped_data.items():
        if layers not in raw_data:
            raw_data[layers] = {}
        if dim not in raw_data[layers]:
            raw_data[layers][dim] = {}
        raw_data[layers][dim][dataset] = metrics.get('corrective', {})

    print(r"\toprule")
    headers = ["Model parameters"] + [synthesis_dataset_header_mapping[ds] for ds in synthesis_dataset_order]
    header_row = f"{headers[0]:<30}" + "".join([f" & {h:<16}" for h in headers[1:]]) + r" \\"
    print(header_row)
    print(r"\midrule")
    for l_val in sorted(raw_data.keys()):
        d_vals = sorted(raw_data[l_val].keys())
        baseline_d = d_vals[0]
        for d_val in d_vals:
            row_str = f"$L={l_val}$, $d_{{\\text{{model}}}}={d_val}$"
            row_str = f"{row_str:<30}"

            for dataset in synthesis_dataset_order:
                curr_dict = raw_data[l_val][d_val].get(dataset, {})
                if not curr_dict:
                    row_str += f" & {'-':<16}"
                    continue

                cor_vals = np.array(list(curr_dict.values()))
                mean_cor = np.mean(cor_vals)
                std_cor = np.std(cor_vals, ddof=1) if len(cor_vals) > 1 else 0.0
                cell_text = f"${mean_cor:.2f} \\pm {std_cor:.2f}$"
                if d_val > baseline_d:
                    baseline_dict = raw_data[l_val][baseline_d].get(dataset, {})
                    common_seeds = set(curr_dict.keys()).intersection(baseline_dict.keys())
                    if len(common_seeds) > 1:
                        diffs = np.array([curr_dict[s] - baseline_dict[s] for s in common_seeds])
                        k_pairs = len(diffs)
                        mean_diff = np.mean(diffs)
                        std_diff = np.std(diffs, ddof=1) if k_pairs > 1 else 0.0
                        df = k_pairs - 1
                        critical_t = stats.t.ppf(1 - 0.05, df) if df > 0 else float('inf')
                        t_stat = (mean_diff / (std_diff / math.sqrt(k_pairs))) if std_diff > 0 else 0
                        if df > 0 and t_stat > critical_t:
                            cell_text = f"\\textcolor{{green!70!black}}{{{cell_text}}}"
                row_str += f" & {cell_text:<16}"
            row_str += r" \\"
            print(row_str)
    print(r"\bottomrule")

def plot_corrective_stats(grouped_data, filename="shallow_grid.pdf"):
    from matplotlib.lines import Line2D

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(8.27, 2.5))

    layers_set = set()
    dims_set = set()
    for (dataset, layers, dim), metrics in grouped_data.items():
        layers_set.add(layers)
        dims_set.add(dim)
    layers_sorted = sorted(list(layers_set), reverse=True)
    dims_sorted = sorted(list(dims_set), reverse=True)

    colors = ['tab:cyan', 'tab:purple']
    linestyles = ['-', '-.', ':']
    color_map = {l: colors[i] for i, l in enumerate(layers_sorted)}
    linestyle_map = {d: linestyles[i] for i, d in enumerate(dims_sorted)}
    layer_handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=f"$L = {l}$")
                     for i, l in enumerate(layers_sorted)]
    dim_handles = [Line2D([0], [0], color='dimgrey', linestyle=linestyles[i % len(linestyles)], lw=2, label=f"$d_{{\\text{{model}}}} = {d}$")
                   for i, d in enumerate(dims_sorted)]
    model_results = {}
    for (dataset, layers, dim), metrics in grouped_data.items():
        config = (layers, dim)
        if config not in model_results:
            model_results[config] = {}
        mean_score = np.mean(list(metrics['corrective'].values())) * 100
        model_results[config][dataset] = mean_score

    x_indices = np.arange(len(synthesis_dataset_order))
    x_labels = [f"{synthesis_dataset_header_mapping[ds]}\n{premise_counts[ds]}"
                for ds in synthesis_dataset_order]

    for config, results in model_results.items():
        l_val, d_val = config
        y_values = [results.get(ds, np.nan) for ds in synthesis_dataset_order]
        c = color_map[l_val]
        ls = linestyle_map[d_val]
        label = f"L={l_val}, d={d_val}"
        plt.plot([x_indices[0] - 0.2, x_indices[0] + 0.2], [y_values[0], y_values[0]], color=c, linestyle=ls, lw=2)
        plt.plot(x_indices[1:], y_values[1:], marker='o', color=c, linestyle=ls, label=label)

    plt.xticks(x_indices, x_labels)
    ax = plt.gca()
    ax.text(x=-0.2, y=-0.03, s="dataset:\navg. premises:",
            transform=ax.get_xaxis_transform(), ha='right', va='top', fontweight='bold')
    plt.ylim(45, 105)
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(handles=layer_handles + dim_handles, bbox_to_anchor=(0.222, 0.0), loc='lower left', labelspacing=0.2)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

def generate_ablation_table(grouped_data):
    raw_data = {}
    for (dataset, layers, dim), metrics in grouped_data.items():
        if layers not in raw_data:
            raw_data[layers] = {}
        if dim not in raw_data[layers]:
            raw_data[layers][dim] = {}
        raw_data[layers][dim][dataset] = metrics

    table_rows = []
    ds_headers = [synthesis_dataset_header_mapping[ds] for ds in synthesis_dataset_order]
    header_row = "Model parameters & Metric & " + " & ".join(ds_headers) + r" \\"
    l_vals = sorted(raw_data.keys())
    for i, l_val in enumerate(l_vals):
        d_vals = sorted(raw_data[l_val].keys())
        for j, d_val in enumerate(d_vals):

            row_dir = [f"$L={l_val}$, $d_{{\\text{{model}}}}={d_val}$", "Direct"]
            row_cor = ["", "Corrective"]
            row_delta = ["", "$\\Delta$"]

            for dataset in synthesis_dataset_order:
                curr_dict = raw_data[l_val][d_val].get(dataset, {})
                d_cor = curr_dict.get('corrective', {})
                d_dir = curr_dict.get('direct', {})
                common = set(d_cor.keys()) & set(d_dir.keys())
                if not common:
                    continue

                common_seeds = sorted(list(common))
                cor_vals = np.array([d_cor[s] for s in common_seeds])
                dir_vals = np.array([d_dir[s] for s in common_seeds])
                mean_cor, std_cor = np.mean(cor_vals), np.std(cor_vals, ddof=1) if len(cor_vals) > 1 else 0.0
                mean_dir, std_dir = np.mean(dir_vals), np.std(dir_vals, ddof=1) if len(dir_vals) > 1 else 0.0

                diffs = [d_cor[s] - d_dir[s] for s in common_seeds]
                raw_differences = np.array(diffs)
                k_pairs = len(raw_differences)
                mean_diff = np.mean(raw_differences)
                std_diff = np.std(raw_differences, ddof=1) if k_pairs > 1 else 0.0
                df = k_pairs - 1
                critical_t = stats.t.ppf(1 - 0.025, df) if df > 0 else float('inf')
                t_stat = (mean_diff / (std_diff / math.sqrt(k_pairs))) if std_diff > 0 else 0
                dir_cell = f"${mean_dir:.3f} \\pm {std_dir:.3f}$"
                cor_cell = f"${mean_cor:.3f} \\pm {std_cor:.3f}$"
                if df > 0:
                    if t_stat > critical_t:
                        cor_cell = f"\\textcolor{{green!70!black}}{{{cor_cell}}}"
                    elif t_stat < -critical_t:
                        cor_cell = f"\\textcolor{{red}}{{{cor_cell}}}"
                delta_cell = f"${mean_diff:+.3f}$"
                row_dir.append(dir_cell)
                row_cor.append(cor_cell)
                row_delta.append(delta_cell)
            table_rows.append(" & ".join(row_dir) + r" \\")
            table_rows.append(" & ".join(row_cor) + r" \\")
            table_rows.append(" & ".join(row_delta) + r" \\")
            if not (i == len(l_vals) - 1 and j == len(d_vals) - 1):
                table_rows.append(r"\midrule")

    col_format = "ll" + "c" * len(synthesis_dataset_order)
    latex_str = (
            "\\begin{table}[ht!]\n"
            "\\centering\n"
            "\\caption{Performance Comparison: Corrective vs Direct (Averaged across 3 seeds)}\n"
            "\\label{tab:corrective_vs_direct}\n"
            "\\resizebox{\\textwidth}{!}{\n"
            f"\\begin{{tabular}}{{{col_format}}}\n"
            "\\toprule\n"
            f"{header_row}\n"
            "\\midrule\n"
            + "\n".join(table_rows) + "\n"
                                      "\\bottomrule\n"
                                      "\\end{tabular}\n"
                                      "}\n"
                                      "\\end{table}"
    )
    return latex_str


def plot_rl_appendix(baseline_runs, rl_runs):
    def parse_label(key):
        parts = set([str(k).lower() for k in key])
        if 'flowrl' in parts:
            method = 'FlowRL'
        elif 'grpo' in parts:
            method = 'GRPO'
        else:
            method = 'Unknown'
        reward = 'sparse' if 'sparse' in parts else 'dense'
        level = 'token' if 'token' in parts else 'sequence'
        return f"{method} {reward} {level}"

    domains = ['lp', 'lp_star', 'rp']
    domain_titles = {'lp': 'LP Evaluation', 'lp_star': 'LP* Evaluation', 'rp': 'RP Evaluation'}

    methods_styles = {
        'Baseline': ('black', '-', 2.5),
        'FlowRL dense sequence': ('tab:blue', '-', 2.0),
        'FlowRL dense token': ('tab:cyan', '--', 1.5),
        'FlowRL sparse sequence': ('tab:purple', '-.', 1.5),
        'FlowRL sparse token': ('navy', ':', 1.5),
        'GRPO dense sequence': ('tab:orange', '-', 2.0),
        'GRPO dense token': ('tab:red', '--', 1.5),
        'GRPO sparse sequence': ('gold', '-.', 1.5),
        'GRPO sparse token': ('brown', ':', 1.5)
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    axes_flat = axes.flatten()

    plot_mapping = {
        'lp': axes[0, 0],
        'lp_star': axes[0, 1],
        'rp': axes[1, 0]
    }

    axes[1, 1].axis('off')
    for i, domain in enumerate(domains):
        ax = plot_mapping[domain]
        ax.set_title(domain_titles[domain], fontsize=16, fontweight='bold')
        if ax == axes[1, 0]:
            ax.set_xlabel("Logical depth", fontsize=16)
            ax.set_ylabel("Accuracy", fontsize=16)
        elif ax == axes[0, 0]:
            ax.set_ylabel("Accuracy", fontsize=16)
        elif ax == axes[0, 1]:
            pass

        ax.set_xticks(range(13))
        ax.set_ylim(0.5, 1.02)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(6.65, 0.52, 'Training cutoff', rotation=90, verticalalignment='bottom', color='gray', fontsize=16)

        for key, scores in baseline_runs.items():
            if key[0] == domain:
                depths = sorted(scores.keys())
                accs = [scores[d] for d in depths]
                ax.plot(depths, accs, label='Baseline', color='black', linewidth=2.5, linestyle='-', zorder=10)
                break

        for key, scores in rl_runs.items():
            if key[0] == domain:
                label = parse_label(key)
                depths = sorted(scores.keys())
                accs = [scores[d] for d in depths]

                if label in methods_styles:
                    c, ls, lw = methods_styles[label]
                else:
                    c, ls, lw = 'gray', '-', 1
                ax.plot(depths, accs, label=label, color=c, linestyle=ls, linewidth=lw, alpha=0.8)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: t[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    axes[1, 1].legend(sorted_handles, sorted_labels, loc='center', fontsize=16, frameon=False)
    plt.tight_layout()
    plt.savefig("final_models_rl_appendix.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_rl_appendix_reduced(baseline_runs, rl_runs):
    def parse_label(key):
        parts = set([str(k).lower() for k in key])
        if 'flowrl' in parts:
            method = 'FlowRL'
        elif 'grpo' in parts:
            method = 'GRPO'
        else:
            method = 'Unknown'
        reward = 'sparse' if 'sparse' in parts else 'dense'
        level = 'token' if 'token' in parts else 'sequence'
        return f"{method} {reward} {level}"

    methods_styles = {
        'Baseline': ('black', '-', 2.5),
        'FlowRL dense sequence': ('tab:blue', '-', 2.0),
        'FlowRL dense token': ('tab:cyan', '--', 1.5),
        'FlowRL sparse sequence': ('tab:purple', '-.', 1.5),
        'FlowRL sparse token': ('navy', ':', 1.5),
        'GRPO dense sequence': ('tab:orange', '-', 2.0),
        'GRPO dense token': ('tab:red', '--', 1.5),
        'GRPO sparse sequence': ('gold', '-.', 1.5),
        'GRPO sparse token': ('brown', ':', 1.5)
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5, 5))

    domain = 'lp'
    ax.set_xlabel("Logical depth (δ)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)

    ax.set_xticks(range(13))
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(6.65, 0.52, 'Training cutoff', rotation=90, verticalalignment='bottom', color='gray', fontsize=16)

    for key, scores in baseline_runs.items():
        if key[0] == domain:
            depths = sorted(scores.keys())
            accs = [scores[d] for d in depths]
            ax.plot(depths, accs, label='Baseline', color='black', linewidth=2.5, linestyle='-', zorder=10)
            break

    for key, scores in rl_runs.items():
        if key[0] == domain:
            label = parse_label(key)
            depths = sorted(scores.keys())
            accs = [scores[d] for d in depths]

            if label in methods_styles:
                c, ls, lw = methods_styles[label]
            else:
                c, ls, lw = 'gray', '-', 1
            ax.plot(depths, accs, label=label, color=c, linestyle=ls, linewidth=lw, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: t[0])
    sorted_labels, sorted_handles = zip(*sorted_pairs)
    ax.legend(sorted_handles, sorted_labels, loc='lower left', fontsize=9, frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("final_models_rl_appendix_reduced.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)



def generate_comparison_table(baseline, mixed):
    # Structure: results[Group][Method][Domain][has_FFN] = (val_6, val_12)
    results = {}
    datasets = [('Baseline', baseline), ('Mixed', mixed)]

    for group, _ in datasets:
        results[group] = {}
        for method in ['Direct', 'CoT']:
            results[group][method] = {}
            for domain in ['lp', 'lp_star', 'rp']:
                results[group][method][domain] = {False: (np.nan, np.nan), True: (np.nan, np.nan)}

    for group, data_dict in datasets:
        for key, scores in data_dict.items():
            eval_domain = key[0]
            is_cot = key[1]
            method = 'CoT' if is_cot else 'Direct'

            has_ffn = False
            for item in key:
                if isinstance(item, str) and 'ffn' == item.lower():
                    has_ffn = True
                    break
            m6, m12 = get_means(data_dict, key)
            if eval_domain in results[group][method]:
                results[group][method][eval_domain][has_ffn] = (m6, m12)

    domains = ['LP', 'LP_STAR', 'RP']  # Display names
    domain_keys = ['lp', 'lp_star', 'rp']  # Keys in dict

    latex = []
    latex.append("\\begin{table}[ht!]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of Baseline vs. Mixed models with and without FFN.}")
    latex.append("\\label{tab:baseline_mixed_comparison}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "cccc" * 3 + "}")
    latex.append("\\toprule")

    row1 = "& "
    cmid1 = ""
    col_idx = 2
    for d in domains:
        clean_d = d.replace('_STAR', '*')
        row1 += f"\\multicolumn{{4}}{{c}}{{{clean_d}}} & "
        cmid1 += f"\\cmidrule(lr){{{col_idx}-{col_idx + 3}}} "
        col_idx += 4
    latex.append(row1.rstrip(" & ") + " \\\\")
    latex.append(cmid1)

    row2 = "& "
    cmid2 = ""
    col_idx = 2
    for _ in domains:
        row2 += "\\multicolumn{2}{c}{No FFN} & \\multicolumn{2}{c}{FFN} & "
        cmid2 += f"\\cmidrule(lr){{{col_idx}-{col_idx + 1}}} \\cmidrule(lr){{{col_idx + 2}-{col_idx + 3}}} "
        col_idx += 4
    latex.append(row2.rstrip(" & ") + " \\\\")
    latex.append(cmid2)

    row3 = "Model & "
    for _ in range(6):
        row3 += "$\\le 6$ & $6 < \\le 12$ & "
    latex.append(row3.rstrip(" & ") + " \\\\")
    latex.append("\\midrule")

    row_order = [
        ('Baseline', 'Direct'),
        ('Baseline', 'CoT'),
        ('Mixed', 'Direct'),
        ('Mixed', 'CoT')
    ]

    for group, method in row_order:
        row_str = f"{group} {method} & "
        for d_key in domain_keys:
            val6_nf, val12_nf = results[group][method][d_key][False]
            row_str += f"{val6_nf:.1f} & {val12_nf:.1f} & "
            val6_ff, val12_ff = results[group][method][d_key][True]
            row_str += f"{val6_ff:.1f} & {val12_ff:.1f} & "
        latex.append(row_str.rstrip(" & ") + " \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    return "\n".join(latex)


def plot_scaling_curves(data_30, data_60, max_depth=12, eval="lp", compare="pred", cls="", curve="layers"):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(4, 4))
    datasets = [(data_30, '-', 'o'), (data_60, '--', 'x')]

    def get_model_info(key):
        if not (key[0] == eval):
            return None
        name = "$L = 8$" if curve == "layers" else ("$H = 4$" if curve == "heads" else None)
        for item in key:
            if curve == "layers" and isinstance(item, str) and item.startswith("layers="):
                ls = item.split('=')[1]
                if ls in ["16", "32", "64", "128"]:
                    name = f"$L = {ls}$"
            if curve == "heads" and isinstance(item, str) and item.startswith("heads="):
                ls = item.split('=')[1]
                if ls in ["6", "8", "11", "16"]:
                    name = f"$H = {ls}$"
        return name

    all_models = set()
    for data, _, _ in datasets:
        for key in data.keys():
            label = get_model_info(key)
            if label:
                all_models.add(label)

    def sort_key(label):
        return int(label.split('=')[1].replace('$', '').replace(')', '').strip())

    sorted_models = sorted(list(all_models), key=sort_key)

    colors = ['black', 'navy', 'tab:purple', 'tab:cyan', 'tab:blue']
    model_color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted_models)}

    legend_handles_models = {}
    for data_dict, style, marker in datasets:
        for key, values in data_dict.items():
            label = get_model_info(key)
            if label:
                xy = sorted([(k, v) for k, v in values.items() if k <= max_depth])
                if not xy: continue
                x_vals, y_vals = zip(*xy)
                c = model_color_map[label]
                ax.plot(x_vals, y_vals, color=c, linestyle=style, marker=marker, markersize=2, alpha=0.8, linewidth=1)
                if label not in legend_handles_models:
                    legend_handles_models[label] = mlines.Line2D([], [], color=c, marker='s', linestyle='None', label=label)

    handles_models = [legend_handles_models[l] for l in sorted_models if l in legend_handles_models]
    if compare == "mode":
        handles_styles = [
            mlines.Line2D([], [], color='gray', linestyle='-', marker='o', label=r'direct'),
            mlines.Line2D([], [], color='gray', linestyle='--', marker='x', label=r'CoT')
        ]
    elif compare == "pred":
        handles_styles = [
            mlines.Line2D([], [], color='gray', linestyle='-', marker='o', label=r'$N_{pred} \leq 30$'),
            mlines.Line2D([], [], color='gray', linestyle='--', marker='x', label=r'$N_{pred} \leq 60$')
        ]
    elif compare == "corrective":
        handles_styles = [
            mlines.Line2D([], [], color='gray', linestyle='-', marker='o', label=r'direct only'),
            mlines.Line2D([], [], color='gray', linestyle='--', marker='x', label=r'direct w. corrective')
        ]

    ax.axvline(x=6.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(6.65, 0.52, 'Training cutoff', rotation=90, verticalalignment='bottom', color='gray', fontsize=16)

    ax.legend(handles=handles_models + handles_styles, loc='lower left', fontsize=9)
    ax.set_xlabel(r'Logical depth $\delta$', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.48, 1.02)
    plt.tight_layout()
    plt.savefig(f"final_models_scaling_curves_{cls}_{curve}_{eval}.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)



args = lambda x: ModelArgs(vocab_size=256, pad_token_id=pad, generated_type_indices=[0, 1], **x)


def base_eval(model, is_cot=False, distribution=None, distributions=["lp", "lp_star", "rp"], eval_f=eval_model, pred_count=30, kv_cache=True):
    collate_fn = lambda x: pad_collate(x, padding=pad)
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cuda")
    model.eval()
    evals = []

    if distribution:
        inference_ids = train_curriculum(full_dataset_dict[distribution])
        ds = DataLoader(prepare_ds_inference(process(inference_ids), is_cot=is_cot), collate_fn=collate_fn)
        with torch.no_grad():
            evals.append(eval_f(model, ds, vocabulary=special_tokens, answer_position=-1, is_cot=is_cot, kv_cache=kv_cache))
    else:
        for distribution in distributions:
            inference_ids = train_curriculum(full_dataset_dict[f"validation_{distribution.lower()}_balanced_deep_{pred_count}_pred"])
            ds = DataLoader(prepare_ds_inference(process(inference_ids), is_cot=is_cot), collate_fn=collate_fn)
            with torch.no_grad():
                evals.append(eval_f(model, ds, vocabulary=special_tokens, answer_position=-1, is_cot=is_cot, kv_cache=kv_cache))

    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cpu")

    return evals

def get_ablation_scores(entries, samples_per_depth=1000, include_direct=True, include_cot=True, kv_cache=True):
    analysis_results = {}
    for entry in entries:
        evals = []
        if ("direct" in entry.tags or "corrective" in entry.tags) and include_direct:
            evals.append(False)
        if ("cot" in entry.tags or "corrective" in entry.tags) and include_cot:
            evals.append(True)
        for is_cot in evals:
            print(f"Evaluating {entry.pretty_name} is_cot={is_cot}. model={entry.model.params}")
            distributions = ["validation_rp_balanced", "validation_lp_balanced",
                             "validation_rp_balanced_3_premise", "validation_rp_balanced_2_premise",
                             "validation_rp_balanced_1_premise", "validation_rp_balanced_1_2_premise",
                             "validation_rp_balanced_2_3_premise"]
            for distribution in distributions:
                accuracies = base_eval(entry.model, is_cot=is_cot, distribution=distribution, kv_cache=kv_cache)
                decimal_accuracies = {layer: count / samples_per_depth for layer, count in accuracies[0].items()}
                analysis_results[(distribution, is_cot, *entry.tags)] = decimal_accuracies
    return analysis_results

def get_raw_scores(entries, samples_per_depth=1000, pred_count=30, include_direct=True, include_cot=True, kv_cache=True):
    analysis_results = {}
    for entry in entries:
        evals = []
        if ("direct" in entry.tags or "corrective" in entry.tags) and include_direct:
            evals.append(False)
        if ("cot" in entry.tags or "corrective" in entry.tags) and include_cot:
            evals.append(True)
        for is_cot in evals:
            print(f"Evaluating {entry.pretty_name} is_cot={is_cot}. model={entry.model.params}")
            lp_accuracies, lp_star_accuracies, rp_accuracies = base_eval(entry.model, is_cot=is_cot, pred_count=pred_count, kv_cache=kv_cache)
            for eval_paradigm, accuracies in [('lp', lp_accuracies), ('lp_star', lp_star_accuracies), ('rp', rp_accuracies)]:
                decimal_accuracies = {layer: count / samples_per_depth for layer, count in accuracies.items()}
                analysis_results[(eval_paradigm, is_cot, *entry.tags)] = decimal_accuracies
    return analysis_results


path = EXPERIMENTS_DIR + f"/models/type_llama_no_ffn/"
repo = ModelRepository(base_path=path, model_class=Transformer, loader_func=load_weights_and_init, args_builder=args)

###### 2^4 experimental results ######
# print("raw_scores_deep_30=", get_raw_scores([x for x in repo.entries if x.tags <= {'rp', 'corrective', 'cot', 'direct', 'bidir', 'r2', 'ffn'}], pred_count=30))
raw_scores_deep_30= {('lp', False, 'bidir', 'corrective', 'rp'): {0: 0.993, 1: 0.982, 2: 0.956, 3: 0.818, 4: 0.727, 5: 0.65, 6: 0.547, 7: 0.543, 8: 0.554, 9: 0.567, 10: 0.552, 11: 0.567, 12: 0.54}, ('lp_star', False, 'bidir', 'corrective', 'rp'): {0: 0.999, 1: 0.998, 2: 1.0, 3: 0.985, 4: 0.971, 5: 0.905, 6: 0.836, 7: 0.795, 8: 0.794, 9: 0.747, 10: 0.737, 11: 0.704, 12: 0.68}, ('rp', False, 'bidir', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.997, 3: 0.992, 4: 0.984, 5: 0.971, 6: 0.945, 7: 0.91, 8: 0.836, 9: 0.754, 10: 0.664, 11: 0.652, 12: 0.597}, ('lp', True, 'bidir', 'corrective', 'rp'): {0: 1.0, 1: 0.998, 2: 0.99, 3: 0.972, 4: 0.923, 5: 0.885, 6: 0.829, 7: 0.832, 8: 0.766, 9: 0.765, 10: 0.731, 11: 0.731, 12: 0.688}, ('lp_star', True, 'bidir', 'corrective', 'rp'): {0: 0.993, 1: 0.992, 2: 1.0, 3: 0.999, 4: 0.996, 5: 0.991, 6: 0.984, 7: 0.973, 8: 0.979, 9: 0.963, 10: 0.938, 11: 0.946, 12: 0.938}, ('rp', True, 'bidir', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.985, 7: 0.978, 8: 0.967, 9: 0.957, 10: 0.917, 11: 0.899, 12: 0.851}, ('lp', True, 'bidir', 'cot', 'rp'): {0: 0.998, 1: 0.998, 2: 0.985, 3: 0.96, 4: 0.903, 5: 0.848, 6: 0.817, 7: 0.805, 8: 0.791, 9: 0.769, 10: 0.748, 11: 0.764, 12: 0.717}, ('lp_star', True, 'bidir', 'cot', 'rp'): {0: 0.997, 1: 0.999, 2: 0.999, 3: 0.997, 4: 0.989, 5: 0.985, 6: 0.98, 7: 0.965, 8: 0.958, 9: 0.943, 10: 0.953, 11: 0.942, 12: 0.932}, ('rp', True, 'bidir', 'cot', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 1.0, 5: 0.989, 6: 0.976, 7: 0.97, 8: 0.941, 9: 0.92, 10: 0.887, 11: 0.851, 12: 0.803}, ('lp', False, 'bidir', 'rp', 'direct'): {0: 0.905, 1: 0.576, 2: 0.535, 3: 0.512, 4: 0.481, 5: 0.49, 6: 0.475, 7: 0.477, 8: 0.476, 9: 0.493, 10: 0.499, 11: 0.526, 12: 0.486}, ('lp_star', False, 'bidir', 'rp', 'direct'): {0: 0.99, 1: 0.701, 2: 0.631, 3: 0.597, 4: 0.583, 5: 0.592, 6: 0.565, 7: 0.555, 8: 0.568, 9: 0.548, 10: 0.546, 11: 0.545, 12: 0.514}, ('rp', False, 'bidir', 'rp', 'direct'): {0: 0.997, 1: 0.913, 2: 0.897, 3: 0.881, 4: 0.878, 5: 0.854, 6: 0.826, 7: 0.825, 8: 0.814, 9: 0.799, 10: 0.764, 11: 0.747, 12: 0.737}, ('lp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.996, 2: 0.972, 3: 0.91, 4: 0.794, 5: 0.708, 6: 0.62, 7: 0.619, 8: 0.623, 9: 0.624, 10: 0.617, 11: 0.649, 12: 0.631}, ('lp_star', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.978, 5: 0.923, 6: 0.886, 7: 0.863, 8: 0.843, 9: 0.849, 10: 0.814, 11: 0.805, 12: 0.792}, ('rp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.983, 5: 0.974, 6: 0.936, 7: 0.902, 8: 0.813, 9: 0.757, 10: 0.657, 11: 0.644, 12: 0.604}, ('lp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.991, 4: 0.969, 5: 0.952, 6: 0.909, 7: 0.906, 8: 0.897, 9: 0.875, 10: 0.86, 11: 0.835, 12: 0.852}, ('lp_star', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.996, 2: 0.999, 3: 0.998, 4: 0.996, 5: 0.997, 6: 0.992, 7: 0.99, 8: 0.984, 9: 0.993, 10: 0.986, 11: 0.987, 12: 0.986}, ('rp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 1.0, 6: 0.996, 7: 0.995, 8: 0.988, 9: 0.981, 10: 0.965, 11: 0.95, 12: 0.928}, ('lp', True, 'bidir', 'r2', 'rp', 'cot'): {0: 1.0, 1: 0.994, 2: 0.993, 3: 0.968, 4: 0.91, 5: 0.845, 6: 0.811, 7: 0.8, 8: 0.762, 9: 0.765, 10: 0.755, 11: 0.758, 12: 0.735}, ('lp_star', True, 'bidir', 'r2', 'rp', 'cot'): {0: 1.0, 1: 0.998, 2: 0.996, 3: 0.994, 4: 0.99, 5: 0.988, 6: 0.989, 7: 0.977, 8: 0.972, 9: 0.964, 10: 0.958, 11: 0.954, 12: 0.96}, ('rp', True, 'bidir', 'r2', 'rp', 'cot'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.998, 4: 0.994, 5: 0.986, 6: 0.976, 7: 0.964, 8: 0.954, 9: 0.931, 10: 0.894, 11: 0.877, 12: 0.839}, ('lp', False, 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.734, 2: 0.626, 3: 0.588, 4: 0.556, 5: 0.533, 6: 0.556, 7: 0.546, 8: 0.552, 9: 0.543, 10: 0.539, 11: 0.527, 12: 0.524}, ('lp_star', False, 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.927, 2: 0.746, 3: 0.66, 4: 0.63, 5: 0.604, 6: 0.578, 7: 0.583, 8: 0.565, 9: 0.554, 10: 0.547, 11: 0.518, 12: 0.54}, ('rp', False, 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.904, 2: 0.868, 3: 0.816, 4: 0.785, 5: 0.751, 6: 0.705, 7: 0.73, 8: 0.74, 9: 0.754, 10: 0.692, 11: 0.715, 12: 0.74}, ('lp', False, 'corrective', 'rp'): {0: 0.989, 1: 0.92, 2: 0.702, 3: 0.576, 4: 0.504, 5: 0.482, 6: 0.467, 7: 0.486, 8: 0.475, 9: 0.484, 10: 0.5, 11: 0.485, 12: 0.479}, ('lp_star', False, 'corrective', 'rp'): {0: 0.97, 1: 1.0, 2: 0.878, 3: 0.784, 4: 0.753, 5: 0.712, 6: 0.682, 7: 0.648, 8: 0.649, 9: 0.611, 10: 0.61, 11: 0.579, 12: 0.545}, ('rp', False, 'corrective', 'rp'): {0: 1.0, 1: 0.994, 2: 0.959, 3: 0.923, 4: 0.907, 5: 0.852, 6: 0.849, 7: 0.824, 8: 0.818, 9: 0.803, 10: 0.725, 11: 0.719, 12: 0.684}, ('lp', True, 'corrective', 'rp'): {0: 1.0, 1: 0.99, 2: 0.957, 3: 0.921, 4: 0.865, 5: 0.82, 6: 0.777, 7: 0.747, 8: 0.761, 9: 0.753, 10: 0.723, 11: 0.7, 12: 0.717}, ('lp_star', True, 'corrective', 'rp'): {0: 0.974, 1: 0.999, 2: 0.989, 3: 0.985, 4: 0.96, 5: 0.944, 6: 0.93, 7: 0.92, 8: 0.908, 9: 0.886, 10: 0.862, 11: 0.853, 12: 0.861}, ('rp', True, 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.995, 4: 0.975, 5: 0.961, 6: 0.937, 7: 0.918, 8: 0.877, 9: 0.858, 10: 0.806, 11: 0.773, 12: 0.732}, ('lp', True, 'cot', 'rp'): {0: 0.968, 1: 0.925, 2: 0.885, 3: 0.819, 4: 0.737, 5: 0.7, 6: 0.641, 7: 0.681, 8: 0.631, 9: 0.66, 10: 0.638, 11: 0.625, 12: 0.657}, ('lp_star', True, 'cot', 'rp'): {0: 0.995, 1: 0.989, 2: 0.966, 3: 0.918, 4: 0.911, 5: 0.876, 6: 0.867, 7: 0.831, 8: 0.81, 9: 0.822, 10: 0.801, 11: 0.774, 12: 0.779}, ('rp', True, 'cot', 'rp'): {0: 0.998, 1: 0.991, 2: 0.982, 3: 0.959, 4: 0.948, 5: 0.909, 6: 0.888, 7: 0.825, 8: 0.831, 9: 0.79, 10: 0.75, 11: 0.744, 12: 0.712}, ('lp', False, 'rp', 'direct'): {0: 0.993, 1: 0.565, 2: 0.507, 3: 0.49, 4: 0.47, 5: 0.48, 6: 0.469, 7: 0.466, 8: 0.476, 9: 0.469, 10: 0.503, 11: 0.503, 12: 0.473}, ('lp_star', False, 'rp', 'direct'): {0: 0.999, 1: 0.703, 2: 0.623, 3: 0.607, 4: 0.598, 5: 0.589, 6: 0.578, 7: 0.573, 8: 0.572, 9: 0.554, 10: 0.55, 11: 0.559, 12: 0.517}, ('rp', False, 'rp', 'direct'): {0: 1.0, 1: 0.898, 2: 0.886, 3: 0.864, 4: 0.867, 5: 0.817, 6: 0.824, 7: 0.813, 8: 0.803, 9: 0.782, 10: 0.771, 11: 0.731, 12: 0.723}, ('lp', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 0.989, 2: 0.96, 3: 0.849, 4: 0.729, 5: 0.629, 6: 0.543, 7: 0.532, 8: 0.501, 9: 0.524, 10: 0.515, 11: 0.544, 12: 0.499}, ('lp_star', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.995, 1: 0.989, 2: 0.998, 3: 0.991, 4: 0.969, 5: 0.901, 6: 0.835, 7: 0.791, 8: 0.795, 9: 0.754, 10: 0.704, 11: 0.708, 12: 0.645}, ('rp', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.997, 4: 0.991, 5: 0.972, 6: 0.952, 7: 0.924, 8: 0.851, 9: 0.754, 10: 0.679, 11: 0.628, 12: 0.568}, ('lp', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 0.996, 2: 0.991, 3: 0.966, 4: 0.902, 5: 0.862, 6: 0.827, 7: 0.822, 8: 0.805, 9: 0.782, 10: 0.78, 11: 0.793, 12: 0.746}, ('lp_star', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.993, 1: 0.987, 2: 0.999, 3: 0.997, 4: 0.993, 5: 0.985, 6: 0.981, 7: 0.967, 8: 0.968, 9: 0.965, 10: 0.953, 11: 0.965, 12: 0.953}, ('rp', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.999, 5: 0.993, 6: 0.987, 7: 0.985, 8: 0.963, 9: 0.943, 10: 0.91, 11: 0.889, 12: 0.862}, ('lp', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 1.0, 1: 0.997, 2: 0.992, 3: 0.979, 4: 0.935, 5: 0.89, 6: 0.822, 7: 0.84, 8: 0.813, 9: 0.813, 10: 0.805, 11: 0.76, 12: 0.758}, ('lp_star', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 0.998, 1: 0.987, 2: 0.999, 3: 1.0, 4: 0.997, 5: 0.99, 6: 0.983, 7: 0.972, 8: 0.977, 9: 0.97, 10: 0.975, 11: 0.977, 12: 0.976}, ('rp', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.995, 6: 0.983, 7: 0.978, 8: 0.973, 9: 0.951, 10: 0.904, 11: 0.872, 12: 0.847}, ('lp', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 0.992, 1: 0.76, 2: 0.573, 3: 0.539, 4: 0.496, 5: 0.503, 6: 0.501, 7: 0.511, 8: 0.494, 9: 0.512, 10: 0.511, 11: 0.516, 12: 0.511}, ('lp_star', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 0.982, 1: 0.931, 2: 0.763, 3: 0.717, 4: 0.706, 5: 0.71, 6: 0.674, 7: 0.645, 8: 0.648, 9: 0.618, 10: 0.598, 11: 0.602, 12: 0.562}, ('rp', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 1.0, 1: 0.954, 2: 0.929, 3: 0.903, 4: 0.905, 5: 0.877, 6: 0.856, 7: 0.846, 8: 0.83, 9: 0.8, 10: 0.766, 11: 0.759, 12: 0.753}, ('lp', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 0.997, 2: 0.994, 3: 0.921, 4: 0.799, 5: 0.705, 6: 0.654, 7: 0.631, 8: 0.636, 9: 0.605, 10: 0.609, 11: 0.62, 12: 0.62}, ('lp_star', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.999, 4: 0.977, 5: 0.939, 6: 0.883, 7: 0.874, 8: 0.818, 9: 0.837, 10: 0.816, 11: 0.801, 12: 0.774}, ('rp', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.997, 4: 0.988, 5: 0.975, 6: 0.946, 7: 0.888, 8: 0.81, 9: 0.741, 10: 0.643, 11: 0.611, 12: 0.587}, ('lp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.988, 4: 0.935, 5: 0.888, 6: 0.881, 7: 0.874, 8: 0.832, 9: 0.844, 10: 0.818, 11: 0.828, 12: 0.819}, ('lp_star', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.991, 6: 0.993, 7: 0.986, 8: 0.985, 9: 0.981, 10: 0.98, 11: 0.988, 12: 0.985}, ('rp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.988, 7: 0.982, 8: 0.974, 9: 0.956, 10: 0.938, 11: 0.9, 12: 0.868}, ('lp', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.995, 4: 0.978, 5: 0.941, 6: 0.911, 7: 0.914, 8: 0.873, 9: 0.9, 10: 0.869, 11: 0.882, 12: 0.87}, ('lp_star', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 1.0, 1: 0.996, 2: 0.999, 3: 1.0, 4: 0.998, 5: 0.994, 6: 0.996, 7: 0.989, 8: 0.991, 9: 0.989, 10: 0.991, 11: 0.992, 12: 0.987}, ('rp', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.995, 7: 0.995, 8: 0.982, 9: 0.978, 10: 0.963, 11: 0.953, 12: 0.948}, ('lp', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.501, 1: 0.519, 2: 0.53, 3: 0.538, 4: 0.547, 5: 0.525, 6: 0.51, 7: 0.522, 8: 0.517, 9: 0.492, 10: 0.5, 11: 0.503, 12: 0.495}, ('lp_star', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.416, 1: 0.53, 2: 0.513, 3: 0.552, 4: 0.534, 5: 0.511, 6: 0.526, 7: 0.522, 8: 0.518, 9: 0.514, 10: 0.49, 11: 0.497, 12: 0.498}, ('rp', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.69, 1: 0.54, 2: 0.512, 3: 0.518, 4: 0.491, 5: 0.529, 6: 0.475, 7: 0.482, 8: 0.511, 9: 0.505, 10: 0.505, 11: 0.5, 12: 0.534}, ('lp', False, 'corrective', 'rp', 'ffn'): {0: 0.961, 1: 0.84, 2: 0.668, 3: 0.565, 4: 0.507, 5: 0.484, 6: 0.461, 7: 0.478, 8: 0.477, 9: 0.479, 10: 0.502, 11: 0.502, 12: 0.506}, ('lp_star', False, 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 0.985, 2: 0.83, 3: 0.759, 4: 0.72, 5: 0.679, 6: 0.668, 7: 0.655, 8: 0.648, 9: 0.645, 10: 0.646, 11: 0.629, 12: 0.604}, ('rp', False, 'corrective', 'rp', 'ffn'): {0: 0.997, 1: 0.983, 2: 0.935, 3: 0.904, 4: 0.891, 5: 0.847, 6: 0.824, 7: 0.814, 8: 0.788, 9: 0.765, 10: 0.706, 11: 0.69, 12: 0.684}, ('lp', True, 'corrective', 'rp', 'ffn'): {0: 0.955, 1: 0.927, 2: 0.9, 3: 0.841, 4: 0.776, 5: 0.748, 6: 0.694, 7: 0.697, 8: 0.702, 9: 0.662, 10: 0.679, 11: 0.675, 12: 0.668}, ('lp_star', True, 'corrective', 'rp', 'ffn'): {0: 1.0, 1: 0.972, 2: 0.951, 3: 0.905, 4: 0.886, 5: 0.857, 6: 0.834, 7: 0.847, 8: 0.795, 9: 0.805, 10: 0.806, 11: 0.79, 12: 0.769}, ('rp', True, 'corrective', 'rp', 'ffn'): {0: 0.996, 1: 0.979, 2: 0.984, 3: 0.958, 4: 0.935, 5: 0.9, 6: 0.882, 7: 0.856, 8: 0.861, 9: 0.796, 10: 0.759, 11: 0.73, 12: 0.71}, ('lp', True, 'cot', 'rp', 'ffn'): {0: 1.0, 1: 0.985, 2: 0.961, 3: 0.9, 4: 0.841, 5: 0.813, 6: 0.784, 7: 0.777, 8: 0.776, 9: 0.769, 10: 0.777, 11: 0.783, 12: 0.797}, ('lp_star', True, 'cot', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 0.986, 3: 0.972, 4: 0.957, 5: 0.945, 6: 0.931, 7: 0.903, 8: 0.892, 9: 0.9, 10: 0.886, 11: 0.879, 12: 0.882}, ('rp', True, 'cot', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.992, 4: 0.977, 5: 0.952, 6: 0.925, 7: 0.898, 8: 0.884, 9: 0.84, 10: 0.792, 11: 0.771, 12: 0.728}, ('lp', False, 'rp', 'ffn', 'direct'): {0: 0.973, 1: 0.546, 2: 0.511, 3: 0.486, 4: 0.484, 5: 0.491, 6: 0.483, 7: 0.472, 8: 0.484, 9: 0.478, 10: 0.517, 11: 0.503, 12: 0.507}, ('lp_star', False, 'rp', 'ffn', 'direct'): {0: 0.995, 1: 0.664, 2: 0.612, 3: 0.609, 4: 0.609, 5: 0.604, 6: 0.596, 7: 0.587, 8: 0.607, 9: 0.577, 10: 0.577, 11: 0.581, 12: 0.545}, ('rp', False, 'rp', 'ffn', 'direct'): {0: 1.0, 1: 0.892, 2: 0.878, 3: 0.862, 4: 0.867, 5: 0.818, 6: 0.817, 7: 0.813, 8: 0.813, 9: 0.773, 10: 0.764, 11: 0.735, 12: 0.722}, ('lp', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.96, 1: 0.882, 2: 0.775, 3: 0.668, 4: 0.598, 5: 0.564, 6: 0.537, 7: 0.567, 8: 0.572, 9: 0.58, 10: 0.558, 11: 0.555, 12: 0.558}, ('lp_star', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 1.0, 1: 0.929, 2: 0.84, 3: 0.784, 4: 0.731, 5: 0.695, 6: 0.679, 7: 0.711, 8: 0.643, 9: 0.67, 10: 0.657, 11: 0.676, 12: 0.678}, ('rp', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.993, 1: 0.945, 2: 0.927, 3: 0.89, 4: 0.861, 5: 0.835, 6: 0.77, 7: 0.76, 8: 0.752, 9: 0.712, 10: 0.677, 11: 0.682, 12: 0.676}, ('lp', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.971, 1: 0.902, 2: 0.867, 3: 0.822, 4: 0.783, 5: 0.772, 6: 0.696, 7: 0.712, 8: 0.689, 9: 0.647, 10: 0.626, 11: 0.623, 12: 0.62}, ('lp_star', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 1.0, 1: 0.93, 2: 0.902, 3: 0.887, 4: 0.868, 5: 0.847, 6: 0.813, 7: 0.828, 8: 0.756, 9: 0.775, 10: 0.782, 11: 0.77, 12: 0.78}, ('rp', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.996, 1: 0.951, 2: 0.953, 3: 0.937, 4: 0.926, 5: 0.917, 6: 0.878, 7: 0.869, 8: 0.86, 9: 0.801, 10: 0.778, 11: 0.761, 12: 0.741}, ('lp', True, 'r2', 'rp', 'ffn', 'cot'): {0: 1.0, 1: 0.997, 2: 0.981, 3: 0.968, 4: 0.925, 5: 0.872, 6: 0.86, 7: 0.838, 8: 0.841, 9: 0.833, 10: 0.847, 11: 0.823, 12: 0.842}, ('lp_star', True, 'r2', 'rp', 'ffn', 'cot'): {0: 0.999, 1: 1.0, 2: 0.995, 3: 0.991, 4: 0.991, 5: 0.978, 6: 0.974, 7: 0.954, 8: 0.953, 9: 0.955, 10: 0.961, 11: 0.94, 12: 0.932}, ('rp', True, 'r2', 'rp', 'ffn', 'cot'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.995, 4: 0.992, 5: 0.978, 6: 0.961, 7: 0.943, 8: 0.932, 9: 0.891, 10: 0.871, 11: 0.848, 12: 0.784}, ('lp', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.625, 1: 0.523, 2: 0.569, 3: 0.531, 4: 0.516, 5: 0.51, 6: 0.523, 7: 0.526, 8: 0.51, 9: 0.505, 10: 0.511, 11: 0.516, 12: 0.527}, ('lp_star', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.6, 1: 0.558, 2: 0.52, 3: 0.535, 4: 0.548, 5: 0.504, 6: 0.516, 7: 0.491, 8: 0.504, 9: 0.513, 10: 0.509, 11: 0.504, 12: 0.501}, ('rp', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.735, 1: 0.589, 2: 0.505, 3: 0.489, 4: 0.496, 5: 0.47, 6: 0.488, 7: 0.495, 8: 0.512, 9: 0.493, 10: 0.502, 11: 0.478, 12: 0.506}, ('lp', False, 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.969, 2: 0.858, 3: 0.769, 4: 0.655, 5: 0.608, 6: 0.578, 7: 0.611, 8: 0.622, 9: 0.607, 10: 0.595, 11: 0.629, 12: 0.566}, ('lp_star', False, 'corrective', 'r2', 'rp'): {0: 0.996, 1: 0.999, 2: 0.961, 3: 0.923, 4: 0.842, 5: 0.786, 6: 0.758, 7: 0.757, 8: 0.719, 9: 0.733, 10: 0.736, 11: 0.695, 12: 0.709}, ('rp', False, 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.995, 2: 0.966, 3: 0.942, 4: 0.909, 5: 0.878, 6: 0.822, 7: 0.798, 8: 0.773, 9: 0.772, 10: 0.696, 11: 0.673, 12: 0.695}, ('lp', True, 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.998, 2: 0.97, 3: 0.95, 4: 0.922, 5: 0.886, 6: 0.854, 7: 0.849, 8: 0.83, 9: 0.827, 10: 0.831, 11: 0.833, 12: 0.823}, ('lp_star', True, 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.995, 2: 0.994, 3: 0.988, 4: 0.966, 5: 0.969, 6: 0.947, 7: 0.951, 8: 0.949, 9: 0.935, 10: 0.937, 11: 0.909, 12: 0.909}, ('rp', True, 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.985, 5: 0.962, 6: 0.939, 7: 0.915, 8: 0.899, 9: 0.853, 10: 0.812, 11: 0.778, 12: 0.757}, ('lp', True, 'r2', 'rp', 'cot'): {0: 1.0, 1: 0.997, 2: 0.979, 3: 0.958, 4: 0.934, 5: 0.886, 6: 0.858, 7: 0.875, 8: 0.826, 9: 0.851, 10: 0.834, 11: 0.832, 12: 0.825}, ('lp_star', True, 'r2', 'rp', 'cot'): {0: 0.999, 1: 0.994, 2: 0.999, 3: 0.991, 4: 0.992, 5: 0.984, 6: 0.968, 7: 0.957, 8: 0.957, 9: 0.937, 10: 0.95, 11: 0.942, 12: 0.922}, ('rp', True, 'r2', 'rp', 'cot'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.993, 5: 0.978, 6: 0.964, 7: 0.943, 8: 0.92, 9: 0.898, 10: 0.865, 11: 0.849, 12: 0.796}, ('lp', False, 'r2', 'rp', 'direct'): {0: 0.753, 1: 0.586, 2: 0.598, 3: 0.57, 4: 0.521, 5: 0.518, 6: 0.526, 7: 0.517, 8: 0.503, 9: 0.51, 10: 0.499, 11: 0.509, 12: 0.498}, ('lp_star', False, 'r2', 'rp', 'direct'): {0: 0.815, 1: 0.561, 2: 0.516, 3: 0.533, 4: 0.522, 5: 0.512, 6: 0.502, 7: 0.496, 8: 0.515, 9: 0.515, 10: 0.507, 11: 0.487, 12: 0.497}, ('rp', False, 'r2', 'rp', 'direct'): {0: 0.882, 1: 0.651, 2: 0.542, 3: 0.484, 4: 0.459, 5: 0.444, 6: 0.435, 7: 0.449, 8: 0.461, 9: 0.47, 10: 0.474, 11: 0.461, 12: 0.466}}
# print("raw_scores_deep_60=", get_raw_scores([x for x in repo.entries if x.tags <= {'rp', 'corrective', 'cot', 'direct', 'bidir', 'r2', 'ffn'}], pred_count=60))
raw_scores_deep_60= {('lp', False, 'bidir', 'corrective', 'rp'): {0: 0.987, 1: 0.948, 2: 0.898, 3: 0.728, 4: 0.648, 5: 0.53, 6: 0.484, 8: 0.457, 7: 0.468, 9: 0.489, 10: 0.478, 11: 0.472, 12: 0.504}, ('lp_star', False, 'bidir', 'corrective', 'rp'): {0: 0.997, 1: 0.988, 2: 0.983, 3: 0.896, 4: 0.829, 5: 0.731, 6: 0.667, 7: 0.641, 8: 0.64, 9: 0.617, 10: 0.589, 11: 0.595, 12: 0.591}, ('rp', False, 'bidir', 'corrective', 'rp'): {0: 0.999, 1: 0.998, 2: 0.996, 3: 0.983, 4: 0.977, 5: 0.946, 6: 0.899, 7: 0.861, 8: 0.769, 9: 0.699, 10: 0.667, 11: 0.613, 12: 0.574}, ('lp', True, 'bidir', 'corrective', 'rp'): {0: 0.996, 1: 0.971, 2: 0.947, 3: 0.873, 4: 0.778, 5: 0.665, 6: 0.641, 8: 0.546, 7: 0.58, 9: 0.53, 10: 0.502, 11: 0.489, 12: 0.505}, ('lp_star', True, 'bidir', 'corrective', 'rp'): {0: 0.985, 1: 0.97, 2: 0.98, 3: 0.935, 4: 0.896, 5: 0.851, 6: 0.808, 7: 0.764, 8: 0.747, 9: 0.715, 10: 0.671, 11: 0.665, 12: 0.618}, ('rp', True, 'bidir', 'corrective', 'rp'): {0: 0.999, 1: 0.984, 2: 0.99, 3: 0.98, 4: 0.965, 5: 0.928, 6: 0.869, 7: 0.856, 8: 0.797, 9: 0.724, 10: 0.665, 11: 0.636, 12: 0.586}, ('lp', True, 'bidir', 'cot', 'rp'): {0: 0.994, 1: 0.973, 2: 0.938, 3: 0.877, 4: 0.802, 5: 0.704, 6: 0.664, 8: 0.582, 7: 0.638, 9: 0.57, 10: 0.547, 11: 0.536, 12: 0.527}, ('lp_star', True, 'bidir', 'cot', 'rp'): {0: 0.995, 1: 0.976, 2: 0.964, 3: 0.935, 4: 0.889, 5: 0.847, 6: 0.81, 7: 0.781, 8: 0.776, 9: 0.733, 10: 0.724, 11: 0.708, 12: 0.693}, ('rp', True, 'bidir', 'cot', 'rp'): {0: 0.983, 1: 0.973, 2: 0.97, 3: 0.951, 4: 0.934, 5: 0.899, 6: 0.83, 7: 0.831, 8: 0.786, 9: 0.71, 10: 0.69, 11: 0.663, 12: 0.612}, ('lp', False, 'bidir', 'rp', 'direct'): {0: 0.885, 1: 0.565, 2: 0.5, 3: 0.478, 4: 0.48, 5: 0.462, 6: 0.477, 8: 0.431, 7: 0.452, 9: 0.46, 10: 0.458, 11: 0.449, 12: 0.465}, ('lp_star', False, 'bidir', 'rp', 'direct'): {0: 0.973, 1: 0.673, 2: 0.599, 3: 0.544, 4: 0.536, 5: 0.54, 6: 0.532, 7: 0.531, 8: 0.532, 9: 0.522, 10: 0.519, 11: 0.517, 12: 0.521}, ('rp', False, 'bidir', 'rp', 'direct'): {0: 0.994, 1: 0.913, 2: 0.904, 3: 0.879, 4: 0.87, 5: 0.85, 6: 0.816, 7: 0.804, 8: 0.752, 9: 0.73, 10: 0.684, 11: 0.68, 12: 0.653}, ('lp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.991, 2: 0.929, 3: 0.83, 4: 0.73, 5: 0.631, 6: 0.552, 8: 0.553, 7: 0.553, 9: 0.55, 10: 0.551, 11: 0.569, 12: 0.546}, ('lp_star', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.999, 2: 0.997, 3: 0.966, 4: 0.9, 5: 0.803, 6: 0.766, 7: 0.747, 8: 0.745, 9: 0.737, 10: 0.721, 11: 0.73, 12: 0.73}, ('rp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 1.0, 2: 0.997, 3: 0.988, 4: 0.965, 5: 0.94, 6: 0.896, 7: 0.866, 8: 0.816, 9: 0.75, 10: 0.71, 11: 0.678, 12: 0.676}, ('lp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.983, 2: 0.965, 3: 0.919, 4: 0.843, 5: 0.768, 6: 0.696, 8: 0.62, 7: 0.69, 9: 0.606, 10: 0.592, 11: 0.576, 12: 0.567}, ('lp_star', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.996, 1: 0.987, 2: 0.984, 3: 0.962, 4: 0.933, 5: 0.901, 6: 0.869, 7: 0.857, 8: 0.836, 9: 0.809, 10: 0.773, 11: 0.765, 12: 0.742}, ('rp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.988, 2: 0.989, 3: 0.969, 4: 0.938, 5: 0.919, 6: 0.879, 7: 0.864, 8: 0.82, 9: 0.749, 10: 0.712, 11: 0.693, 12: 0.675}, ('lp', True, 'bidir', 'r2', 'rp', 'cot'): {0: 0.994, 1: 0.977, 2: 0.935, 3: 0.86, 4: 0.772, 5: 0.7, 6: 0.647, 8: 0.562, 7: 0.599, 9: 0.57, 10: 0.557, 11: 0.539, 12: 0.524}, ('lp_star', True, 'bidir', 'r2', 'rp', 'cot'): {0: 0.991, 1: 0.97, 2: 0.983, 3: 0.944, 4: 0.897, 5: 0.866, 6: 0.822, 7: 0.815, 8: 0.787, 9: 0.771, 10: 0.76, 11: 0.725, 12: 0.714}, ('rp', True, 'bidir', 'r2', 'rp', 'cot'): {0: 0.999, 1: 0.997, 2: 0.994, 3: 0.971, 4: 0.941, 5: 0.887, 6: 0.857, 7: 0.803, 8: 0.753, 9: 0.665, 10: 0.666, 11: 0.612, 12: 0.573}, ('lp', False, 'bidir', 'r2', 'rp', 'direct'): {0: 0.997, 1: 0.707, 2: 0.597, 3: 0.562, 4: 0.54, 5: 0.514, 6: 0.524, 8: 0.516, 7: 0.543, 9: 0.552, 10: 0.541, 11: 0.552, 12: 0.552}, ('lp_star', False, 'bidir', 'r2', 'rp', 'direct'): {0: 0.994, 1: 0.877, 2: 0.719, 3: 0.638, 4: 0.576, 5: 0.56, 6: 0.542, 7: 0.519, 8: 0.538, 9: 0.513, 10: 0.51, 11: 0.52, 12: 0.533}, ('rp', False, 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.915, 2: 0.861, 3: 0.803, 4: 0.812, 5: 0.717, 6: 0.719, 7: 0.733, 8: 0.658, 9: 0.635, 10: 0.635, 11: 0.609, 12: 0.66}, ('lp', False, 'corrective', 'rp'): {0: 0.985, 1: 0.868, 2: 0.603, 3: 0.489, 4: 0.463, 5: 0.436, 6: 0.452, 8: 0.458, 7: 0.439, 9: 0.458, 10: 0.465, 11: 0.477, 12: 0.46}, ('lp_star', False, 'corrective', 'rp'): {0: 0.981, 1: 0.968, 2: 0.764, 3: 0.662, 4: 0.632, 5: 0.587, 6: 0.599, 7: 0.571, 8: 0.575, 9: 0.561, 10: 0.554, 11: 0.549, 12: 0.551}, ('rp', False, 'corrective', 'rp'): {0: 1.0, 1: 0.989, 2: 0.928, 3: 0.873, 4: 0.823, 5: 0.801, 6: 0.753, 7: 0.717, 8: 0.692, 9: 0.639, 10: 0.612, 11: 0.595, 12: 0.585}, ('lp', True, 'corrective', 'rp'): {0: 0.998, 1: 0.924, 2: 0.871, 3: 0.793, 4: 0.715, 5: 0.633, 6: 0.609, 8: 0.549, 7: 0.578, 9: 0.54, 10: 0.526, 11: 0.541, 12: 0.523}, ('lp_star', True, 'corrective', 'rp'): {0: 0.985, 1: 0.952, 2: 0.937, 3: 0.853, 4: 0.825, 5: 0.781, 6: 0.724, 7: 0.697, 8: 0.692, 9: 0.665, 10: 0.625, 11: 0.643, 12: 0.595}, ('rp', True, 'corrective', 'rp'): {0: 1.0, 1: 0.951, 2: 0.921, 3: 0.89, 4: 0.863, 5: 0.83, 6: 0.786, 7: 0.73, 8: 0.699, 9: 0.651, 10: 0.62, 11: 0.605, 12: 0.58}, ('lp', True, 'cot', 'rp'): {0: 0.946, 1: 0.864, 2: 0.796, 3: 0.679, 4: 0.612, 5: 0.576, 6: 0.537, 8: 0.527, 7: 0.553, 9: 0.519, 10: 0.524, 11: 0.539, 12: 0.525}, ('lp_star', True, 'cot', 'rp'): {0: 0.991, 1: 0.953, 2: 0.87, 3: 0.804, 4: 0.748, 5: 0.725, 6: 0.677, 7: 0.662, 8: 0.651, 9: 0.624, 10: 0.627, 11: 0.593, 12: 0.6}, ('rp', True, 'cot', 'rp'): {0: 0.995, 1: 0.973, 2: 0.951, 3: 0.893, 4: 0.827, 5: 0.794, 6: 0.746, 7: 0.692, 8: 0.665, 9: 0.614, 10: 0.58, 11: 0.602, 12: 0.565}, ('lp', False, 'rp', 'direct'): {0: 0.983, 1: 0.543, 2: 0.482, 3: 0.458, 4: 0.472, 5: 0.467, 6: 0.466, 8: 0.438, 7: 0.453, 9: 0.471, 10: 0.452, 11: 0.445, 12: 0.451}, ('lp_star', False, 'rp', 'direct'): {0: 0.996, 1: 0.668, 2: 0.591, 3: 0.554, 4: 0.542, 5: 0.542, 6: 0.542, 7: 0.535, 8: 0.536, 9: 0.532, 10: 0.529, 11: 0.522, 12: 0.525}, ('rp', False, 'rp', 'direct'): {0: 1.0, 1: 0.906, 2: 0.882, 3: 0.875, 4: 0.859, 5: 0.829, 6: 0.783, 7: 0.79, 8: 0.77, 9: 0.743, 10: 0.702, 11: 0.677, 12: 0.654}, ('lp', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.993, 1: 0.976, 2: 0.928, 3: 0.767, 4: 0.644, 5: 0.548, 6: 0.488, 8: 0.445, 7: 0.441, 9: 0.48, 10: 0.478, 11: 0.484, 12: 0.476}, ('lp_star', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.989, 1: 0.992, 2: 0.992, 3: 0.914, 4: 0.847, 5: 0.724, 6: 0.685, 7: 0.643, 8: 0.635, 9: 0.602, 10: 0.609, 11: 0.593, 12: 0.6}, ('rp', False, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.996, 1: 1.0, 2: 1.0, 3: 0.982, 4: 0.977, 5: 0.929, 6: 0.877, 7: 0.809, 8: 0.765, 9: 0.667, 10: 0.621, 11: 0.582, 12: 0.559}, ('lp', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.998, 1: 0.979, 2: 0.953, 3: 0.906, 4: 0.806, 5: 0.709, 6: 0.654, 8: 0.571, 7: 0.603, 9: 0.561, 10: 0.542, 11: 0.541, 12: 0.547}, ('lp_star', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.991, 1: 0.972, 2: 0.988, 3: 0.959, 4: 0.91, 5: 0.862, 6: 0.813, 7: 0.799, 8: 0.786, 9: 0.736, 10: 0.703, 11: 0.694, 12: 0.692}, ('rp', True, 'bidir', 'corrective', 'rp', 'ffn'): {0: 0.997, 1: 0.99, 2: 0.994, 3: 0.983, 4: 0.946, 5: 0.907, 6: 0.861, 7: 0.83, 8: 0.791, 9: 0.716, 10: 0.676, 11: 0.642, 12: 0.607}, ('lp', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 0.998, 1: 0.991, 2: 0.967, 3: 0.918, 4: 0.833, 5: 0.726, 6: 0.673, 8: 0.582, 7: 0.589, 9: 0.564, 10: 0.53, 11: 0.544, 12: 0.526}, ('lp_star', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 0.997, 1: 0.981, 2: 0.983, 3: 0.959, 4: 0.926, 5: 0.872, 6: 0.821, 7: 0.797, 8: 0.779, 9: 0.74, 10: 0.734, 11: 0.722, 12: 0.698}, ('rp', True, 'bidir', 'cot', 'rp', 'ffn'): {0: 0.999, 1: 0.994, 2: 0.995, 3: 0.993, 4: 0.977, 5: 0.961, 6: 0.908, 7: 0.874, 8: 0.835, 9: 0.768, 10: 0.711, 11: 0.659, 12: 0.612}, ('lp', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 0.992, 1: 0.721, 2: 0.563, 3: 0.521, 4: 0.528, 5: 0.523, 6: 0.522, 8: 0.474, 7: 0.498, 9: 0.472, 10: 0.468, 11: 0.472, 12: 0.456}, ('lp_star', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 0.987, 1: 0.915, 2: 0.741, 3: 0.682, 4: 0.671, 5: 0.643, 6: 0.652, 7: 0.656, 8: 0.652, 9: 0.647, 10: 0.651, 11: 0.634, 12: 0.646}, ('rp', False, 'bidir', 'rp', 'ffn', 'direct'): {0: 0.999, 1: 0.965, 2: 0.916, 3: 0.912, 4: 0.905, 5: 0.872, 6: 0.861, 7: 0.853, 8: 0.819, 9: 0.79, 10: 0.77, 11: 0.759, 12: 0.731}, ('lp', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 0.998, 1: 0.998, 2: 0.974, 3: 0.868, 4: 0.722, 5: 0.623, 6: 0.54, 8: 0.519, 7: 0.526, 9: 0.543, 10: 0.574, 11: 0.55, 12: 0.556}, ('lp_star', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 0.998, 1: 1.0, 2: 0.998, 3: 0.979, 4: 0.9, 5: 0.825, 6: 0.749, 7: 0.716, 8: 0.709, 9: 0.694, 10: 0.691, 11: 0.698, 12: 0.676}, ('rp', False, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 0.992, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.981, 5: 0.955, 6: 0.891, 7: 0.832, 8: 0.753, 9: 0.656, 10: 0.634, 11: 0.577, 12: 0.579}, ('lp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 0.999, 1: 0.989, 2: 0.973, 3: 0.93, 4: 0.83, 5: 0.758, 6: 0.708, 8: 0.615, 7: 0.649, 9: 0.607, 10: 0.595, 11: 0.589, 12: 0.58}, ('lp_star', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 0.995, 2: 0.983, 3: 0.963, 4: 0.927, 5: 0.896, 6: 0.862, 7: 0.833, 8: 0.827, 9: 0.792, 10: 0.779, 11: 0.768, 12: 0.744}, ('rp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 0.997, 1: 0.987, 2: 0.993, 3: 0.99, 4: 0.978, 5: 0.95, 6: 0.901, 7: 0.889, 8: 0.821, 9: 0.769, 10: 0.703, 11: 0.683, 12: 0.642}, ('lp', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 0.999, 1: 0.986, 2: 0.964, 3: 0.94, 4: 0.868, 5: 0.788, 6: 0.766, 8: 0.644, 7: 0.661, 9: 0.618, 10: 0.605, 11: 0.604, 12: 0.58}, ('lp_star', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 0.999, 1: 0.986, 2: 0.989, 3: 0.962, 4: 0.945, 5: 0.926, 6: 0.879, 7: 0.858, 8: 0.861, 9: 0.829, 10: 0.818, 11: 0.805, 12: 0.804}, ('rp', True, 'r2', 'rp', 'ffn', 'cot', 'bidir'): {0: 1.0, 1: 0.993, 2: 0.995, 3: 0.994, 4: 0.987, 5: 0.957, 6: 0.918, 7: 0.912, 8: 0.863, 9: 0.802, 10: 0.742, 11: 0.716, 12: 0.678}, ('lp', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.513, 1: 0.456, 2: 0.485, 3: 0.521, 4: 0.534, 5: 0.549, 6: 0.555, 8: 0.525, 7: 0.566, 9: 0.531, 10: 0.551, 11: 0.519, 12: 0.525}, ('lp_star', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.43, 1: 0.507, 2: 0.498, 3: 0.523, 4: 0.536, 5: 0.539, 6: 0.581, 7: 0.564, 8: 0.566, 9: 0.562, 10: 0.546, 11: 0.56, 12: 0.562}, ('rp', False, 'r2', 'rp', 'ffn', 'direct', 'bidir'): {0: 0.737, 1: 0.566, 2: 0.515, 3: 0.479, 4: 0.451, 5: 0.464, 6: 0.434, 7: 0.42, 8: 0.418, 9: 0.43, 10: 0.41, 11: 0.471, 12: 0.434}, ('lp', False, 'corrective', 'rp', 'ffn'): {0: 0.953, 1: 0.792, 2: 0.601, 3: 0.516, 4: 0.478, 5: 0.461, 6: 0.5, 8: 0.464, 7: 0.444, 9: 0.457, 10: 0.468, 11: 0.462, 12: 0.442}, ('lp_star', False, 'corrective', 'rp', 'ffn'): {0: 0.996, 1: 0.956, 2: 0.797, 3: 0.657, 4: 0.612, 5: 0.595, 6: 0.622, 7: 0.596, 8: 0.606, 9: 0.584, 10: 0.592, 11: 0.597, 12: 0.59}, ('rp', False, 'corrective', 'rp', 'ffn'): {0: 0.996, 1: 0.993, 2: 0.929, 3: 0.892, 4: 0.855, 5: 0.814, 6: 0.777, 7: 0.737, 8: 0.719, 9: 0.67, 10: 0.638, 11: 0.63, 12: 0.61}, ('lp', True, 'corrective', 'rp', 'ffn'): {0: 0.948, 1: 0.903, 2: 0.841, 3: 0.759, 4: 0.678, 5: 0.669, 6: 0.598, 8: 0.547, 7: 0.572, 9: 0.54, 10: 0.536, 11: 0.538, 12: 0.539}, ('lp_star', True, 'corrective', 'rp', 'ffn'): {0: 0.998, 1: 0.938, 2: 0.906, 3: 0.833, 4: 0.781, 5: 0.755, 6: 0.718, 7: 0.703, 8: 0.682, 9: 0.656, 10: 0.664, 11: 0.676, 12: 0.637}, ('rp', True, 'corrective', 'rp', 'ffn'): {0: 0.989, 1: 0.981, 2: 0.959, 3: 0.923, 4: 0.878, 5: 0.835, 6: 0.781, 7: 0.756, 8: 0.723, 9: 0.672, 10: 0.649, 11: 0.654, 12: 0.624}, ('lp', True, 'cot', 'rp', 'ffn'): {0: 0.997, 1: 0.935, 2: 0.888, 3: 0.815, 4: 0.753, 5: 0.671, 6: 0.663, 8: 0.573, 7: 0.622, 9: 0.577, 10: 0.556, 11: 0.536, 12: 0.547}, ('lp_star', True, 'cot', 'rp', 'ffn'): {0: 0.999, 1: 0.979, 2: 0.966, 3: 0.887, 4: 0.851, 5: 0.801, 6: 0.772, 7: 0.768, 8: 0.753, 9: 0.721, 10: 0.706, 11: 0.702, 12: 0.681}, ('rp', True, 'cot', 'rp', 'ffn'): {0: 0.994, 1: 0.975, 2: 0.979, 3: 0.945, 4: 0.916, 5: 0.878, 6: 0.84, 7: 0.784, 8: 0.738, 9: 0.712, 10: 0.653, 11: 0.643, 12: 0.614}, ('lp', False, 'rp', 'ffn', 'direct'): {0: 0.95, 1: 0.54, 2: 0.486, 3: 0.477, 4: 0.473, 5: 0.459, 6: 0.484, 8: 0.44, 7: 0.475, 9: 0.458, 10: 0.46, 11: 0.459, 12: 0.45}, ('lp_star', False, 'rp', 'ffn', 'direct'): {0: 0.994, 1: 0.659, 2: 0.586, 3: 0.568, 4: 0.56, 5: 0.562, 6: 0.56, 7: 0.569, 8: 0.566, 9: 0.558, 10: 0.553, 11: 0.545, 12: 0.562}, ('rp', False, 'rp', 'ffn', 'direct'): {0: 0.999, 1: 0.89, 2: 0.877, 3: 0.87, 4: 0.849, 5: 0.828, 6: 0.78, 7: 0.79, 8: 0.754, 9: 0.711, 10: 0.678, 11: 0.659, 12: 0.632}, ('lp', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.955, 1: 0.872, 2: 0.755, 3: 0.657, 4: 0.578, 5: 0.563, 6: 0.544, 8: 0.533, 7: 0.504, 9: 0.527, 10: 0.519, 11: 0.534, 12: 0.529}, ('lp_star', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.995, 1: 0.929, 2: 0.836, 3: 0.74, 4: 0.684, 5: 0.615, 6: 0.641, 7: 0.629, 8: 0.615, 9: 0.593, 10: 0.613, 11: 0.601, 12: 0.612}, ('rp', False, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.987, 1: 0.956, 2: 0.924, 3: 0.883, 4: 0.866, 5: 0.789, 6: 0.777, 7: 0.763, 8: 0.739, 9: 0.719, 10: 0.67, 11: 0.699, 12: 0.675}, ('lp', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.973, 1: 0.887, 2: 0.832, 3: 0.751, 4: 0.705, 5: 0.686, 6: 0.62, 8: 0.559, 7: 0.6, 9: 0.557, 10: 0.549, 11: 0.549, 12: 0.541}, ('lp_star', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.995, 1: 0.946, 2: 0.892, 3: 0.833, 4: 0.785, 5: 0.75, 6: 0.722, 7: 0.707, 8: 0.682, 9: 0.65, 10: 0.652, 11: 0.65, 12: 0.629}, ('rp', True, 'corrective', 'r2', 'rp', 'ffn'): {0: 0.987, 1: 0.964, 2: 0.95, 3: 0.936, 4: 0.914, 5: 0.837, 6: 0.812, 7: 0.766, 8: 0.735, 9: 0.692, 10: 0.614, 11: 0.613, 12: 0.598}, ('lp', True, 'r2', 'rp', 'ffn', 'cot'): {0: 0.999, 1: 0.972, 2: 0.944, 3: 0.882, 4: 0.822, 5: 0.776, 6: 0.714, 8: 0.599, 7: 0.684, 9: 0.632, 10: 0.598, 11: 0.565, 12: 0.602}, ('lp_star', True, 'r2', 'rp', 'ffn', 'cot'): {0: 1.0, 1: 0.978, 2: 0.978, 3: 0.944, 4: 0.914, 5: 0.861, 6: 0.855, 7: 0.843, 8: 0.814, 9: 0.787, 10: 0.769, 11: 0.781, 12: 0.769}, ('rp', True, 'r2', 'rp', 'ffn', 'cot'): {0: 0.988, 1: 0.964, 2: 0.968, 3: 0.953, 4: 0.933, 5: 0.885, 6: 0.835, 7: 0.825, 8: 0.789, 9: 0.751, 10: 0.721, 11: 0.689, 12: 0.642}, ('lp', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.611, 1: 0.541, 2: 0.577, 3: 0.54, 4: 0.552, 5: 0.568, 6: 0.545, 8: 0.532, 7: 0.549, 9: 0.524, 10: 0.547, 11: 0.508, 12: 0.523}, ('lp_star', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.55, 1: 0.525, 2: 0.552, 3: 0.549, 4: 0.559, 5: 0.526, 6: 0.543, 7: 0.541, 8: 0.556, 9: 0.495, 10: 0.506, 11: 0.547, 12: 0.505}, ('rp', False, 'r2', 'rp', 'ffn', 'direct'): {0: 0.769, 1: 0.581, 2: 0.502, 3: 0.474, 4: 0.479, 5: 0.46, 6: 0.46, 7: 0.51, 8: 0.472, 9: 0.446, 10: 0.475, 11: 0.491, 12: 0.484}, ('lp', False, 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.921, 2: 0.802, 3: 0.669, 4: 0.592, 5: 0.539, 6: 0.549, 8: 0.556, 7: 0.527, 9: 0.544, 10: 0.549, 11: 0.57, 12: 0.55}, ('lp_star', False, 'corrective', 'r2', 'rp'): {0: 0.997, 1: 0.979, 2: 0.909, 3: 0.828, 4: 0.759, 5: 0.701, 6: 0.669, 7: 0.667, 8: 0.678, 9: 0.656, 10: 0.671, 11: 0.66, 12: 0.661}, ('rp', False, 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.984, 2: 0.951, 3: 0.899, 4: 0.852, 5: 0.797, 6: 0.768, 7: 0.718, 8: 0.668, 9: 0.65, 10: 0.628, 11: 0.613, 12: 0.613}, ('lp', True, 'corrective', 'r2', 'rp'): {0: 0.997, 1: 0.954, 2: 0.907, 3: 0.83, 4: 0.77, 5: 0.711, 6: 0.686, 8: 0.654, 7: 0.654, 9: 0.632, 10: 0.609, 11: 0.624, 12: 0.595}, ('lp_star', True, 'corrective', 'r2', 'rp'): {0: 0.985, 1: 0.954, 2: 0.937, 3: 0.872, 4: 0.844, 5: 0.802, 6: 0.764, 7: 0.761, 8: 0.741, 9: 0.708, 10: 0.675, 11: 0.687, 12: 0.673}, ('rp', True, 'corrective', 'r2', 'rp'): {0: 0.999, 1: 0.987, 2: 0.957, 3: 0.935, 4: 0.879, 5: 0.853, 6: 0.761, 7: 0.721, 8: 0.681, 9: 0.611, 10: 0.591, 11: 0.592, 12: 0.592}, ('lp', True, 'r2', 'rp', 'cot'): {0: 0.994, 1: 0.967, 2: 0.938, 3: 0.869, 4: 0.801, 5: 0.756, 6: 0.719, 8: 0.619, 7: 0.664, 9: 0.625, 10: 0.593, 11: 0.586, 12: 0.599}, ('lp_star', True, 'r2', 'rp', 'cot'): {0: 0.995, 1: 0.982, 2: 0.973, 3: 0.939, 4: 0.899, 5: 0.85, 6: 0.817, 7: 0.794, 8: 0.792, 9: 0.761, 10: 0.743, 11: 0.721, 12: 0.712}, ('rp', True, 'r2', 'rp', 'cot'): {0: 1.0, 1: 0.995, 2: 0.992, 3: 0.979, 4: 0.936, 5: 0.891, 6: 0.837, 7: 0.824, 8: 0.752, 9: 0.699, 10: 0.673, 11: 0.66, 12: 0.602}, ('lp', False, 'r2', 'rp', 'direct'): {0: 0.707, 1: 0.544, 2: 0.575, 3: 0.547, 4: 0.527, 5: 0.537, 6: 0.528, 8: 0.552, 7: 0.533, 9: 0.553, 10: 0.518, 11: 0.534, 12: 0.539}, ('lp_star', False, 'r2', 'rp', 'direct'): {0: 0.739, 1: 0.536, 2: 0.528, 3: 0.507, 4: 0.518, 5: 0.509, 6: 0.553, 7: 0.488, 8: 0.501, 9: 0.503, 10: 0.506, 11: 0.517, 12: 0.497}, ('rp', False, 'r2', 'rp', 'direct'): {0: 0.829, 1: 0.564, 2: 0.441, 3: 0.446, 4: 0.43, 5: 0.417, 6: 0.404, 7: 0.408, 8: 0.4, 9: 0.399, 10: 0.424, 11: 0.439, 12: 0.465}}
# print(generate_main_tables(raw_scores_deep_30, raw_scores_deep_60))
# print(generate_differential_summary(raw_scores_deep_30, raw_scores_deep_60))

###### Baselines ######
# print("raw_scores_baseline_ffn_deep_30=", get_raw_scores([repo.get_entry('rp', 'corrective', 'r2', 'bidir', 'ffn')], pred_count=30, include_direct=False))
raw_scores_baseline_ffn_deep_30= {('lp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.988, 4: 0.935, 5: 0.888, 6: 0.881, 7: 0.874, 8: 0.832, 9: 0.844, 10: 0.818, 11: 0.828, 12: 0.819}, ('lp_star', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.991, 6: 0.993, 7: 0.986, 8: 0.985, 9: 0.981, 10: 0.98, 11: 0.988, 12: 0.985}, ('rp', True, 'r2', 'rp', 'ffn', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.988, 7: 0.982, 8: 0.974, 9: 0.956, 10: 0.938, 11: 0.9, 12: 0.868}}

# print("raw_scores_baseline_deep_30=", get_raw_scores([repo.get_entry('rp', 'corrective', 'r2', 'bidir')], pred_count=30))
raw_scores_baseline_deep_30= {('lp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.996, 2: 0.972, 3: 0.91, 4: 0.794, 5: 0.708, 6: 0.62, 7: 0.619, 8: 0.623, 9: 0.624, 10: 0.617, 11: 0.649, 12: 0.631}, ('lp_star', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.978, 5: 0.923, 6: 0.886, 7: 0.863, 8: 0.843, 9: 0.849, 10: 0.814, 11: 0.805, 12: 0.792}, ('rp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.983, 5: 0.974, 6: 0.936, 7: 0.902, 8: 0.813, 9: 0.757, 10: 0.657, 11: 0.644, 12: 0.604}, ('lp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.991, 4: 0.969, 5: 0.952, 6: 0.909, 7: 0.906, 8: 0.897, 9: 0.875, 10: 0.86, 11: 0.835, 12: 0.852}, ('lp_star', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.996, 2: 0.999, 3: 0.998, 4: 0.996, 5: 0.997, 6: 0.992, 7: 0.99, 8: 0.984, 9: 0.993, 10: 0.986, 11: 0.987, 12: 0.986}, ('rp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 1.0, 6: 0.996, 7: 0.995, 8: 0.988, 9: 0.981, 10: 0.965, 11: 0.95, 12: 0.928}}
# print("raw_scores_baseline_deep_60=", get_raw_scores([repo.get_entry('rp', 'corrective', 'r2', 'bidir')], pred_count=60))
raw_scores_baseline_deep_60= {('lp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.991, 2: 0.929, 3: 0.83, 4: 0.73, 5: 0.631, 6: 0.552, 8: 0.553, 7: 0.553, 9: 0.55, 10: 0.551, 11: 0.569, 12: 0.546}, ('lp_star', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.999, 2: 0.997, 3: 0.966, 4: 0.9, 5: 0.803, 6: 0.766, 7: 0.747, 8: 0.745, 9: 0.737, 10: 0.721, 11: 0.73, 12: 0.73}, ('rp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 1.0, 2: 0.997, 3: 0.988, 4: 0.965, 5: 0.94, 6: 0.896, 7: 0.866, 8: 0.816, 9: 0.75, 10: 0.71, 11: 0.678, 12: 0.676}, ('lp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.983, 2: 0.965, 3: 0.919, 4: 0.843, 5: 0.768, 6: 0.696, 8: 0.62, 7: 0.69, 9: 0.606, 10: 0.592, 11: 0.576, 12: 0.567}, ('lp_star', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.996, 1: 0.987, 2: 0.984, 3: 0.962, 4: 0.933, 5: 0.901, 6: 0.869, 7: 0.857, 8: 0.836, 9: 0.809, 10: 0.773, 11: 0.765, 12: 0.742}, ('rp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.988, 2: 0.989, 3: 0.969, 4: 0.938, 5: 0.919, 6: 0.879, 7: 0.864, 8: 0.82, 9: 0.749, 10: 0.712, 11: 0.693, 12: 0.675}}

# print("raw_scores_baseline_seeds_deep_30=", get_raw_scores([x for x in repo.entries if {'rp', 'baseline'} <= x.tags], pred_count=30))
raw_scores_baseline_seeds_deep_30= {('lp', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.997, 2: 0.981, 3: 0.921, 4: 0.789, 5: 0.717, 6: 0.576, 7: 0.544, 8: 0.526, 9: 0.549, 10: 0.548, 11: 0.548, 12: 0.546}, ('lp_star', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 0.996, 4: 0.983, 5: 0.939, 6: 0.867, 7: 0.814, 8: 0.795, 9: 0.765, 10: 0.75, 11: 0.74, 12: 0.716}, ('rp', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.997, 4: 0.99, 5: 0.977, 6: 0.951, 7: 0.909, 8: 0.823, 9: 0.739, 10: 0.641, 11: 0.626, 12: 0.572}, ('lp', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.995, 3: 0.983, 4: 0.954, 5: 0.895, 6: 0.855, 7: 0.843, 8: 0.829, 9: 0.815, 10: 0.804, 11: 0.756, 12: 0.732}, ('lp_star', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.992, 2: 0.999, 3: 0.998, 4: 0.994, 5: 0.99, 6: 0.983, 7: 0.97, 8: 0.972, 9: 0.979, 10: 0.971, 11: 0.976, 12: 0.981}, ('rp', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.998, 6: 0.993, 7: 0.984, 8: 0.966, 9: 0.956, 10: 0.919, 11: 0.899, 12: 0.884}, ('lp', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.994, 2: 0.966, 3: 0.911, 4: 0.775, 5: 0.675, 6: 0.583, 7: 0.556, 8: 0.564, 9: 0.602, 10: 0.56, 11: 0.594, 12: 0.602}, ('lp_star', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.998, 2: 0.998, 3: 0.996, 4: 0.967, 5: 0.923, 6: 0.87, 7: 0.843, 8: 0.82, 9: 0.791, 10: 0.777, 11: 0.77, 12: 0.738}, ('rp', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.992, 4: 0.99, 5: 0.959, 6: 0.93, 7: 0.891, 8: 0.83, 9: 0.733, 10: 0.658, 11: 0.643, 12: 0.597}, ('lp', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.998, 2: 0.997, 3: 0.982, 4: 0.956, 5: 0.926, 6: 0.88, 7: 0.877, 8: 0.84, 9: 0.812, 10: 0.811, 11: 0.802, 12: 0.773}, ('lp_star', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 1.0, 2: 0.999, 3: 1.0, 4: 0.999, 5: 0.994, 6: 0.993, 7: 0.984, 8: 0.988, 9: 0.986, 10: 0.98, 11: 0.971, 12: 0.969}, ('rp', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.997, 6: 0.99, 7: 0.984, 8: 0.977, 9: 0.959, 10: 0.943, 11: 0.939, 12: 0.911}, ('lp', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.997, 2: 0.992, 3: 0.934, 4: 0.853, 5: 0.791, 6: 0.694, 7: 0.641, 8: 0.607, 9: 0.603, 10: 0.602, 11: 0.602, 12: 0.586}, ('lp_star', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.996, 2: 1.0, 3: 0.999, 4: 0.985, 5: 0.967, 6: 0.908, 7: 0.838, 8: 0.788, 9: 0.794, 10: 0.747, 11: 0.735, 12: 0.719}, ('rp', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.999, 4: 0.997, 5: 0.997, 6: 0.969, 7: 0.94, 8: 0.861, 9: 0.768, 10: 0.675, 11: 0.626, 12: 0.581}, ('lp', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.986, 4: 0.971, 5: 0.944, 6: 0.918, 7: 0.913, 8: 0.917, 9: 0.897, 10: 0.897, 11: 0.894, 12: 0.902}, ('lp_star', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.992, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.996, 7: 0.995, 8: 0.994, 9: 0.994, 10: 0.994, 11: 0.997, 12: 0.994}, ('rp', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.994, 7: 0.996, 8: 0.986, 9: 0.983, 10: 0.978, 11: 0.938, 12: 0.929}, ('lp', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.998, 2: 0.986, 3: 0.95, 4: 0.887, 5: 0.816, 6: 0.74, 7: 0.747, 8: 0.683, 9: 0.672, 10: 0.657, 11: 0.664, 12: 0.636}, ('lp_star', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.998, 2: 1.0, 3: 1.0, 4: 0.99, 5: 0.969, 6: 0.932, 7: 0.888, 8: 0.807, 9: 0.782, 10: 0.743, 11: 0.714, 12: 0.701}, ('rp', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.999, 4: 0.994, 5: 0.994, 6: 0.977, 7: 0.946, 8: 0.885, 9: 0.784, 10: 0.664, 11: 0.624, 12: 0.579}, ('lp', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.999, 2: 0.997, 3: 0.991, 4: 0.964, 5: 0.941, 6: 0.903, 7: 0.936, 8: 0.906, 9: 0.886, 10: 0.888, 11: 0.884, 12: 0.878}, ('lp_star', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.997, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.993, 6: 0.994, 7: 0.99, 8: 0.989, 9: 0.989, 10: 0.99, 11: 0.986, 12: 0.991}, ('rp', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 1.0, 5: 0.995, 6: 0.992, 7: 0.988, 8: 0.984, 9: 0.966, 10: 0.959, 11: 0.951, 12: 0.925}}
# print("raw_scores_baseline_seeds_deep_60=", get_raw_scores([x for x in repo.entries if {'rp', 'baseline'} <= x.tags], pred_count=60))
raw_scores_baseline_seeds_deep_60= {('lp', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.995, 1: 0.988, 2: 0.952, 3: 0.855, 4: 0.683, 5: 0.589, 6: 0.499, 8: 0.483, 7: 0.471, 9: 0.499, 10: 0.501, 11: 0.536, 12: 0.515}, ('lp_star', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.995, 1: 0.98, 2: 0.983, 3: 0.921, 4: 0.851, 5: 0.792, 6: 0.694, 7: 0.654, 8: 0.633, 9: 0.616, 10: 0.605, 11: 0.612, 12: 0.607}, ('rp', False, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.997, 1: 0.997, 2: 0.997, 3: 0.988, 4: 0.978, 5: 0.93, 6: 0.875, 7: 0.81, 8: 0.725, 9: 0.648, 10: 0.605, 11: 0.567, 12: 0.554}, ('lp', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.993, 1: 0.979, 2: 0.958, 3: 0.917, 4: 0.812, 5: 0.715, 6: 0.678, 8: 0.592, 7: 0.62, 9: 0.582, 10: 0.575, 11: 0.57, 12: 0.536}, ('lp_star', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.994, 1: 0.96, 2: 0.96, 3: 0.946, 4: 0.898, 5: 0.864, 6: 0.826, 7: 0.799, 8: 0.775, 9: 0.744, 10: 0.728, 11: 0.705, 12: 0.696}, ('rp', True, 'rp', 'r2', 'seed124', 'bidir', 'corrective', 'seed'): {0: 0.995, 1: 0.974, 2: 0.969, 3: 0.951, 4: 0.904, 5: 0.865, 6: 0.833, 7: 0.793, 8: 0.759, 9: 0.681, 10: 0.656, 11: 0.616, 12: 0.595}, ('lp', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.998, 1: 0.971, 2: 0.939, 3: 0.825, 4: 0.662, 5: 0.56, 6: 0.487, 8: 0.487, 7: 0.478, 9: 0.517, 10: 0.505, 11: 0.513, 12: 0.503}, ('lp_star', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.998, 1: 0.983, 2: 0.985, 3: 0.936, 4: 0.851, 5: 0.774, 6: 0.711, 7: 0.689, 8: 0.689, 9: 0.656, 10: 0.637, 11: 0.638, 12: 0.633}, ('rp', False, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.998, 2: 0.998, 3: 0.984, 4: 0.965, 5: 0.938, 6: 0.891, 7: 0.827, 8: 0.776, 9: 0.689, 10: 0.649, 11: 0.613, 12: 0.596}, ('lp', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.98, 2: 0.967, 3: 0.921, 4: 0.835, 5: 0.746, 6: 0.706, 8: 0.605, 7: 0.647, 9: 0.616, 10: 0.583, 11: 0.611, 12: 0.571}, ('lp_star', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 0.999, 1: 0.983, 2: 0.989, 3: 0.971, 4: 0.94, 5: 0.888, 6: 0.846, 7: 0.826, 8: 0.805, 9: 0.749, 10: 0.743, 11: 0.733, 12: 0.705}, ('rp', True, 'seed125', 'rp', 'r2', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.977, 2: 0.982, 3: 0.959, 4: 0.952, 5: 0.904, 6: 0.867, 7: 0.828, 8: 0.809, 9: 0.726, 10: 0.677, 11: 0.667, 12: 0.648}, ('lp', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.996, 1: 0.994, 2: 0.962, 3: 0.894, 4: 0.772, 5: 0.687, 6: 0.587, 8: 0.539, 7: 0.548, 9: 0.553, 10: 0.557, 11: 0.568, 12: 0.54}, ('lp_star', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.99, 1: 0.996, 2: 0.997, 3: 0.964, 4: 0.902, 5: 0.817, 6: 0.743, 7: 0.695, 8: 0.665, 9: 0.638, 10: 0.625, 11: 0.622, 12: 0.62}, ('rp', False, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.991, 1: 1.0, 2: 1.0, 3: 0.996, 4: 0.981, 5: 0.955, 6: 0.915, 7: 0.862, 8: 0.779, 9: 0.708, 10: 0.634, 11: 0.576, 12: 0.569}, ('lp', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.998, 1: 0.988, 2: 0.969, 3: 0.945, 4: 0.86, 5: 0.798, 6: 0.747, 8: 0.668, 7: 0.696, 9: 0.64, 10: 0.628, 11: 0.621, 12: 0.617}, ('lp_star', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.997, 1: 0.987, 2: 0.99, 3: 0.961, 4: 0.932, 5: 0.912, 6: 0.879, 7: 0.857, 8: 0.835, 9: 0.824, 10: 0.803, 11: 0.783, 12: 0.799}, ('rp', True, 'rp', 'r2', 'seed124', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.993, 1: 0.995, 2: 0.994, 3: 0.989, 4: 0.968, 5: 0.944, 6: 0.91, 7: 0.888, 8: 0.829, 9: 0.776, 10: 0.756, 11: 0.687, 12: 0.666}, ('lp', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.995, 2: 0.955, 3: 0.884, 4: 0.806, 5: 0.722, 6: 0.665, 8: 0.587, 7: 0.586, 9: 0.578, 10: 0.619, 11: 0.593, 12: 0.583}, ('lp_star', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.992, 1: 0.995, 2: 0.995, 3: 0.975, 4: 0.928, 5: 0.872, 6: 0.796, 7: 0.74, 8: 0.713, 9: 0.686, 10: 0.689, 11: 0.675, 12: 0.66}, ('rp', False, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.994, 1: 1.0, 2: 1.0, 3: 0.996, 4: 0.995, 5: 0.97, 6: 0.943, 7: 0.902, 8: 0.826, 9: 0.767, 10: 0.68, 11: 0.653, 12: 0.648}, ('lp', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.989, 2: 0.96, 3: 0.933, 4: 0.864, 5: 0.792, 6: 0.774, 8: 0.672, 7: 0.714, 9: 0.641, 10: 0.651, 11: 0.633, 12: 0.63}, ('lp_star', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 1.0, 1: 0.984, 2: 0.981, 3: 0.967, 4: 0.935, 5: 0.907, 6: 0.873, 7: 0.847, 8: 0.838, 9: 0.799, 10: 0.808, 11: 0.795, 12: 0.781}, ('rp', True, 'seed125', 'rp', 'r2', 'ffn', 'bidir', 'corrective', 'seed'): {0: 0.991, 1: 0.988, 2: 0.996, 3: 0.991, 4: 0.972, 5: 0.962, 6: 0.928, 7: 0.929, 8: 0.87, 9: 0.822, 10: 0.792, 11: 0.721, 12: 0.694}}
raw_scores_baseline_seeds_deep_30 |= {x: y for x,y in raw_scores_deep_30.items() if {'bidir', 'corrective', 'rp', 'r2'} <= set(x[2:])}
raw_scores_baseline_seeds_deep_60 |= {x: y for x,y in raw_scores_deep_60.items() if {'bidir', 'corrective', 'rp', 'r2'} <= set(x[2:])}

###### Universal appendix ######
# print("raw_scores_universal_deep_30=", get_raw_scores([x for x in repo.entries if {'universal'} <= x.tags], pred_count=30, include_cot=False, kv_cache=False))
raw_scores_universal_deep_30= {('lp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 1.0, 1: 0.995, 2: 0.984, 3: 0.941, 4: 0.866, 5: 0.788, 6: 0.672, 7: 0.602, 8: 0.592, 9: 0.583, 10: 0.56, 11: 0.603, 12: 0.606}, ('lp_star', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.993, 1: 1.0, 2: 0.999, 3: 1.0, 4: 0.99, 5: 0.963, 6: 0.87, 7: 0.77, 8: 0.699, 9: 0.697, 10: 0.68, 11: 0.647, 12: 0.658}, ('rp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.998, 1: 0.999, 2: 0.999, 3: 0.996, 4: 0.994, 5: 0.985, 6: 0.955, 7: 0.917, 8: 0.824, 9: 0.713, 10: 0.64, 11: 0.583, 12: 0.552}, ('lp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 1.0, 1: 0.993, 2: 0.98, 3: 0.928, 4: 0.808, 5: 0.712, 6: 0.634, 7: 0.604, 8: 0.617, 9: 0.632, 10: 0.622, 11: 0.622, 12: 0.647}, ('lp_star', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.994, 1: 0.993, 2: 1.0, 3: 0.999, 4: 0.987, 5: 0.949, 6: 0.909, 7: 0.873, 8: 0.854, 9: 0.865, 10: 0.847, 11: 0.852, 12: 0.823}, ('rp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 1.0, 1: 0.999, 2: 0.995, 3: 0.996, 4: 0.987, 5: 0.966, 6: 0.932, 7: 0.886, 8: 0.792, 9: 0.717, 10: 0.64, 11: 0.623, 12: 0.586}, ('lp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 0.999, 2: 0.999, 3: 0.994, 4: 0.983, 5: 0.952, 6: 0.901, 7: 0.838, 8: 0.749, 9: 0.694, 10: 0.601, 11: 0.595, 12: 0.557}, ('lp_star', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.997, 1: 0.998, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.996, 6: 0.977, 7: 0.948, 8: 0.902, 9: 0.881, 10: 0.785, 11: 0.718, 12: 0.646}, ('rp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.996, 7: 0.988, 8: 0.947, 9: 0.848, 10: 0.724, 11: 0.619, 12: 0.556}, ('lp', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 0.999, 2: 0.996, 3: 0.989, 4: 0.942, 5: 0.886, 6: 0.813, 7: 0.758, 8: 0.659, 9: 0.616, 10: 0.558, 11: 0.551, 12: 0.539}, ('lp_star', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.999, 1: 0.996, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.99, 6: 0.966, 7: 0.921, 8: 0.856, 9: 0.789, 10: 0.722, 11: 0.695, 12: 0.653}, ('rp', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.991, 7: 0.974, 8: 0.913, 9: 0.813, 10: 0.698, 11: 0.608, 12: 0.558}, ('lp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.988, 4: 0.972, 5: 0.906, 6: 0.821, 7: 0.774, 8: 0.667, 9: 0.601, 10: 0.573, 11: 0.546, 12: 0.532}, ('lp_star', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 0.993, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.992, 6: 0.972, 7: 0.924, 8: 0.866, 9: 0.808, 10: 0.773, 11: 0.716, 12: 0.682}, ('rp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 1.0, 5: 0.998, 6: 0.993, 7: 0.987, 8: 0.919, 9: 0.814, 10: 0.665, 11: 0.596, 12: 0.541}, ('lp', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 0.999, 1: 0.985, 2: 0.946, 3: 0.849, 4: 0.721, 5: 0.649, 6: 0.597, 7: 0.596, 8: 0.61, 9: 0.611, 10: 0.619, 11: 0.621, 12: 0.622}, ('lp_star', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.992, 4: 0.943, 5: 0.881, 6: 0.819, 7: 0.797, 8: 0.788, 9: 0.816, 10: 0.791, 11: 0.772, 12: 0.774}, ('rp', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 1.0, 1: 0.999, 2: 0.996, 3: 0.983, 4: 0.97, 5: 0.952, 6: 0.894, 7: 0.826, 8: 0.8, 9: 0.7, 10: 0.636, 11: 0.616, 12: 0.593}}
# print("raw_scores_universal_deep_60=", get_raw_scores([x for x in repo.entries if {'universal'} <= x.tags], pred_count=60, include_cot=False, kv_cache=False))
raw_scores_universal_deep_60= {('lp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.985, 1: 0.973, 2: 0.943, 3: 0.88, 4: 0.806, 5: 0.67, 6: 0.607, 8: 0.533, 7: 0.515, 9: 0.514, 10: 0.539, 11: 0.513, 12: 0.535}, ('lp_star', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.985, 1: 0.999, 2: 0.997, 3: 0.984, 4: 0.93, 5: 0.845, 6: 0.742, 7: 0.659, 8: 0.644, 9: 0.644, 10: 0.62, 11: 0.64, 12: 0.637}, ('rp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.984, 1: 0.998, 2: 0.992, 3: 0.983, 4: 0.961, 5: 0.937, 6: 0.892, 7: 0.843, 8: 0.782, 9: 0.716, 10: 0.695, 11: 0.675, 12: 0.644}, ('lp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.991, 1: 0.977, 2: 0.942, 3: 0.838, 4: 0.72, 5: 0.591, 6: 0.562, 8: 0.547, 7: 0.539, 9: 0.555, 10: 0.561, 11: 0.561, 12: 0.557}, ('lp_star', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.988, 1: 0.965, 2: 0.968, 3: 0.944, 4: 0.885, 5: 0.83, 6: 0.771, 7: 0.702, 8: 0.727, 9: 0.709, 10: 0.699, 11: 0.709, 12: 0.708}, ('rp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp'): {0: 0.978, 1: 0.937, 2: 0.927, 3: 0.918, 4: 0.89, 5: 0.855, 6: 0.8, 7: 0.756, 8: 0.684, 9: 0.629, 10: 0.605, 11: 0.583, 12: 0.574}, ('lp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.991, 1: 0.996, 2: 0.986, 3: 0.979, 4: 0.939, 5: 0.83, 6: 0.734, 8: 0.564, 7: 0.638, 9: 0.558, 10: 0.531, 11: 0.51, 12: 0.514}, ('lp_star', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.998, 1: 0.991, 2: 0.992, 3: 0.972, 4: 0.952, 5: 0.907, 6: 0.803, 7: 0.748, 8: 0.671, 9: 0.63, 10: 0.581, 11: 0.575, 12: 0.545}, ('rp', False, 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.979, 1: 0.985, 2: 0.985, 3: 0.979, 4: 0.975, 5: 0.961, 6: 0.943, 7: 0.913, 8: 0.835, 9: 0.711, 10: 0.647, 11: 0.584, 12: 0.553}, ('lp', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.994, 1: 0.998, 2: 0.986, 3: 0.952, 4: 0.876, 5: 0.73, 6: 0.654, 8: 0.524, 7: 0.567, 9: 0.516, 10: 0.515, 11: 0.515, 12: 0.506}, ('lp_star', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.994, 1: 0.989, 2: 0.995, 3: 0.979, 4: 0.95, 5: 0.891, 6: 0.798, 7: 0.717, 8: 0.664, 9: 0.59, 10: 0.575, 11: 0.553, 12: 0.56}, ('rp', False, 'r2', 'corrective', 'seed124', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.988, 1: 0.996, 2: 0.997, 3: 0.997, 4: 0.992, 5: 0.97, 6: 0.948, 7: 0.914, 8: 0.82, 9: 0.723, 10: 0.642, 11: 0.593, 12: 0.554}, ('lp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.995, 1: 0.997, 2: 0.98, 3: 0.964, 4: 0.928, 5: 0.801, 6: 0.697, 8: 0.531, 7: 0.586, 9: 0.529, 10: 0.52, 11: 0.506, 12: 0.505}, ('lp_star', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.998, 1: 0.989, 2: 0.987, 3: 0.974, 4: 0.955, 5: 0.909, 6: 0.787, 7: 0.715, 8: 0.659, 9: 0.617, 10: 0.571, 11: 0.577, 12: 0.575}, ('rp', False, 'seed125', 'r2', 'corrective', 'bidir', 'universal', 'rp', 'ffn'): {0: 0.992, 1: 0.987, 2: 0.989, 3: 0.986, 4: 0.986, 5: 0.965, 6: 0.952, 7: 0.921, 8: 0.839, 9: 0.743, 10: 0.664, 11: 0.614, 12: 0.561}, ('lp', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 0.999, 1: 0.962, 2: 0.904, 3: 0.807, 4: 0.65, 5: 0.589, 6: 0.551, 8: 0.555, 7: 0.525, 9: 0.541, 10: 0.562, 11: 0.546, 12: 0.567}, ('lp_star', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 1.0, 1: 0.996, 2: 0.994, 3: 0.949, 4: 0.872, 5: 0.811, 6: 0.76, 7: 0.761, 8: 0.766, 9: 0.757, 10: 0.773, 11: 0.776, 12: 0.779}, ('rp', False, 'bidir', 'seed124', 'rp', 'universal', 'corrective', 'seed', 'r2'): {0: 1.0, 1: 0.982, 2: 0.967, 3: 0.934, 4: 0.918, 5: 0.863, 6: 0.809, 7: 0.773, 8: 0.704, 9: 0.651, 10: 0.629, 11: 0.599, 12: 0.581}}
# print(generate_universal_appendix(raw_scores_baseline_seeds_deep_30, raw_scores_baseline_seeds_deep_60, raw_scores_universal_deep_30, raw_scores_universal_deep_60))

###### RL ######
# print("raw_scores_rl_deep_30=", get_raw_scores([x for x in repo.entries if {'grpo'} <= x.tags or {'flowrl'} <= x.tags], pred_count=30, include_direct=False))
raw_scores_rl_deep_30= {('lp', True, 'rp', 'flowrl'): {0: 0.999, 1: 0.993, 2: 0.951, 3: 0.877, 4: 0.802, 5: 0.781, 6: 0.773, 7: 0.749, 8: 0.769, 9: 0.766, 10: 0.73, 11: 0.762, 12: 0.741}, ('lp_star', True, 'rp', 'flowrl'): {0: 1.0, 1: 0.991, 2: 0.972, 3: 0.946, 4: 0.922, 5: 0.918, 6: 0.947, 7: 0.95, 8: 0.941, 9: 0.946, 10: 0.964, 11: 0.956, 12: 0.96}, ('rp', True, 'rp', 'flowrl'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.967, 4: 0.959, 5: 0.959, 6: 0.953, 7: 0.966, 8: 0.943, 9: 0.933, 10: 0.898, 11: 0.855, 12: 0.818}, ('lp', True, 'sparse', 'rp', 'flowrl'): {0: 1.0, 1: 0.997, 2: 0.997, 3: 0.932, 4: 0.84, 5: 0.802, 6: 0.755, 7: 0.761, 8: 0.753, 9: 0.733, 10: 0.724, 11: 0.739, 12: 0.74}, ('lp_star', True, 'sparse', 'rp', 'flowrl'): {0: 1.0, 1: 0.997, 2: 0.999, 3: 0.993, 4: 0.967, 5: 0.935, 6: 0.912, 7: 0.92, 8: 0.901, 9: 0.911, 10: 0.89, 11: 0.913, 12: 0.9}, ('rp', True, 'sparse', 'rp', 'flowrl'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.996, 4: 0.994, 5: 0.981, 6: 0.955, 7: 0.946, 8: 0.908, 9: 0.87, 10: 0.769, 11: 0.745, 12: 0.716}, ('lp', True, 'rp', 'flowrl', 'token'): {0: 0.999, 1: 0.997, 2: 0.98, 3: 0.901, 4: 0.837, 5: 0.811, 6: 0.789, 7: 0.79, 8: 0.794, 9: 0.784, 10: 0.762, 11: 0.788, 12: 0.761}, ('lp_star', True, 'rp', 'flowrl', 'token'): {0: 1.0, 1: 0.998, 2: 0.998, 3: 0.985, 4: 0.975, 5: 0.96, 6: 0.965, 7: 0.967, 8: 0.96, 9: 0.965, 10: 0.974, 11: 0.976, 12: 0.971}, ('rp', True, 'rp', 'flowrl', 'token'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.993, 4: 0.985, 5: 0.985, 6: 0.979, 7: 0.974, 8: 0.968, 9: 0.952, 10: 0.93, 11: 0.892, 12: 0.852}, ('lp', True, 'sparse', 'rp', 'flowrl', 'token'): {0: 1.0, 1: 0.993, 2: 0.996, 3: 0.938, 4: 0.862, 5: 0.83, 6: 0.818, 7: 0.819, 8: 0.803, 9: 0.802, 10: 0.754, 11: 0.807, 12: 0.782}, ('lp_star', True, 'sparse', 'rp', 'flowrl', 'token'): {0: 1.0, 1: 0.996, 2: 1.0, 3: 0.972, 4: 0.964, 5: 0.946, 6: 0.936, 7: 0.945, 8: 0.947, 9: 0.954, 10: 0.953, 11: 0.957, 12: 0.956}, ('rp', True, 'sparse', 'rp', 'flowrl', 'token'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.996, 5: 0.987, 6: 0.973, 7: 0.965, 8: 0.945, 9: 0.91, 10: 0.866, 11: 0.846, 12: 0.806}, ('lp', True, 'rp', 'grpo'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.988, 4: 0.936, 5: 0.895, 6: 0.885, 7: 0.875, 8: 0.844, 9: 0.84, 10: 0.82, 11: 0.837, 12: 0.823}, ('lp_star', True, 'rp', 'grpo'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.988, 6: 0.989, 7: 0.98, 8: 0.983, 9: 0.981, 10: 0.979, 11: 0.987, 12: 0.982}, ('rp', True, 'rp', 'grpo'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.994, 6: 0.986, 7: 0.985, 8: 0.973, 9: 0.957, 10: 0.93, 11: 0.894, 12: 0.865}, ('lp', True, 'sparse', 'rp', 'grpo'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.987, 4: 0.936, 5: 0.888, 6: 0.881, 7: 0.877, 8: 0.831, 9: 0.838, 10: 0.822, 11: 0.838, 12: 0.825}, ('lp_star', True, 'sparse', 'rp', 'grpo'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.989, 6: 0.989, 7: 0.98, 8: 0.983, 9: 0.979, 10: 0.979, 11: 0.987, 12: 0.984}, ('rp', True, 'sparse', 'rp', 'grpo'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.996, 6: 0.985, 7: 0.982, 8: 0.974, 9: 0.963, 10: 0.929, 11: 0.901, 12: 0.867}, ('lp', True, 'rp', 'grpo', 'token'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.987, 4: 0.936, 5: 0.896, 6: 0.881, 7: 0.868, 8: 0.845, 9: 0.849, 10: 0.819, 11: 0.846, 12: 0.833}, ('lp_star', True, 'rp', 'grpo', 'token'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.989, 6: 0.989, 7: 0.98, 8: 0.98, 9: 0.982, 10: 0.98, 11: 0.988, 12: 0.986}, ('rp', True, 'rp', 'grpo', 'token'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.993, 6: 0.986, 7: 0.983, 8: 0.973, 9: 0.961, 10: 0.93, 11: 0.898, 12: 0.866}, ('lp', True, 'sparse', 'rp', 'grpo', 'token'): {0: 1.0, 1: 1.0, 2: 0.997, 3: 0.989, 4: 0.937, 5: 0.889, 6: 0.881, 7: 0.877, 8: 0.841, 9: 0.845, 10: 0.819, 11: 0.833, 12: 0.814}, ('lp_star', True, 'sparse', 'rp', 'grpo', 'token'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.989, 6: 0.989, 7: 0.978, 8: 0.984, 9: 0.976, 10: 0.978, 11: 0.986, 12: 0.983}, ('rp', True, 'sparse', 'rp', 'grpo', 'token'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.986, 7: 0.985, 8: 0.974, 9: 0.964, 10: 0.926, 11: 0.892, 12: 0.862}}
# plot_rl_appendix(raw_scores_baseline_ffn_deep_30, raw_scores_rl_deep_30)
# plot_rl_appendix_reduced(raw_scores_baseline_ffn_deep_30, raw_scores_rl_deep_30)

###### Mixed comparison ######
# print("raw_scores_mixed_deep_30=", get_raw_scores([repo.get_entry('rp', 'corrective', 'r2', 'bidir'), repo.get_entry('rp', 'direct', 'r2', 'bidir'), repo.get_entry('rp', 'cot', 'r2', 'bidir'), repo.get_entry('rp', 'direct', 'cot', 'r2', 'bidir')], pred_count=30))
raw_scores_mixed_deep_30= {('lp', False, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 0.996, 2: 0.972, 3: 0.91, 4: 0.794, 5: 0.708, 6: 0.62, 7: 0.619, 8: 0.623, 9: 0.624, 10: 0.617, 11: 0.649, 12: 0.631}, ('lp_star', False, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.978, 5: 0.923, 6: 0.886, 7: 0.863, 8: 0.843, 9: 0.849, 10: 0.814, 11: 0.805, 12: 0.792}, ('rp', False, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.983, 5: 0.974, 6: 0.936, 7: 0.902, 8: 0.813, 9: 0.757, 10: 0.657, 11: 0.644, 12: 0.604}, ('lp', True, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.991, 4: 0.969, 5: 0.952, 6: 0.909, 7: 0.906, 8: 0.897, 9: 0.875, 10: 0.86, 11: 0.835, 12: 0.852}, ('lp_star', True, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 0.996, 2: 0.999, 3: 0.998, 4: 0.996, 5: 0.997, 6: 0.992, 7: 0.99, 8: 0.984, 9: 0.993, 10: 0.986, 11: 0.987, 12: 0.986}, ('rp', True, 'r2', 'corrective', 'bidir', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 1.0, 6: 0.996, 7: 0.995, 8: 0.988, 9: 0.981, 10: 0.965, 11: 0.95, 12: 0.928}, ('lp', False, 'r2', 'bidir', 'direct', 'rp'): {0: 1.0, 1: 0.734, 2: 0.626, 3: 0.588, 4: 0.556, 5: 0.533, 6: 0.556, 7: 0.546, 8: 0.552, 9: 0.543, 10: 0.539, 11: 0.527, 12: 0.524}, ('lp_star', False, 'r2', 'bidir', 'direct', 'rp'): {0: 1.0, 1: 0.927, 2: 0.746, 3: 0.66, 4: 0.63, 5: 0.604, 6: 0.578, 7: 0.583, 8: 0.565, 9: 0.554, 10: 0.547, 11: 0.518, 12: 0.54}, ('rp', False, 'r2', 'bidir', 'direct', 'rp'): {0: 1.0, 1: 0.904, 2: 0.868, 3: 0.816, 4: 0.785, 5: 0.751, 6: 0.705, 7: 0.73, 8: 0.74, 9: 0.754, 10: 0.692, 11: 0.715, 12: 0.74}, ('lp', True, 'cot', 'bidir', 'r2', 'rp'): {0: 1.0, 1: 0.994, 2: 0.993, 3: 0.968, 4: 0.91, 5: 0.845, 6: 0.811, 7: 0.8, 8: 0.762, 9: 0.765, 10: 0.755, 11: 0.758, 12: 0.735}, ('lp_star', True, 'cot', 'bidir', 'r2', 'rp'): {0: 1.0, 1: 0.998, 2: 0.996, 3: 0.994, 4: 0.99, 5: 0.988, 6: 0.989, 7: 0.977, 8: 0.972, 9: 0.964, 10: 0.958, 11: 0.954, 12: 0.96}, ('rp', True, 'cot', 'bidir', 'r2', 'rp'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.998, 4: 0.994, 5: 0.986, 6: 0.976, 7: 0.964, 8: 0.954, 9: 0.931, 10: 0.894, 11: 0.877, 12: 0.839}, ('lp', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.999, 1: 0.993, 2: 0.97, 3: 0.908, 4: 0.8, 5: 0.69, 6: 0.614, 7: 0.613, 8: 0.589, 9: 0.554, 10: 0.581, 11: 0.568, 12: 0.575}, ('lp_star', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 0.997, 4: 0.982, 5: 0.948, 6: 0.909, 7: 0.871, 8: 0.844, 9: 0.851, 10: 0.837, 11: 0.806, 12: 0.803}, ('rp', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.996, 4: 0.986, 5: 0.969, 6: 0.946, 7: 0.914, 8: 0.868, 9: 0.784, 10: 0.702, 11: 0.673, 12: 0.633}, ('lp', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.998, 2: 0.996, 3: 0.979, 4: 0.933, 5: 0.878, 6: 0.832, 7: 0.829, 8: 0.763, 9: 0.744, 10: 0.707, 11: 0.669, 12: 0.659}, ('lp_star', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.995, 1: 0.997, 2: 0.999, 3: 1.0, 4: 0.998, 5: 0.989, 6: 0.984, 7: 0.978, 8: 0.978, 9: 0.964, 10: 0.963, 11: 0.962, 12: 0.958}, ('rp', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 1.0, 4: 0.997, 5: 0.994, 6: 0.983, 7: 0.979, 8: 0.96, 9: 0.946, 10: 0.901, 11: 0.902, 12: 0.87}}
# print("raw_scores_mixed_deep_60=", get_raw_scores([repo.get_entry('rp', 'corrective', 'r2', 'bidir'), repo.get_entry('rp', 'direct', 'r2', 'bidir'), repo.get_entry('rp', 'cot', 'r2', 'bidir'), repo.get_entry('rp', 'direct', 'cot', 'r2', 'bidir')], pred_count=60))
raw_scores_mixed_deep_60= {('lp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.991, 2: 0.929, 3: 0.83, 4: 0.73, 5: 0.631, 6: 0.552, 8: 0.553, 7: 0.553, 9: 0.55, 10: 0.551, 11: 0.569, 12: 0.546}, ('lp_star', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.999, 2: 0.997, 3: 0.966, 4: 0.9, 5: 0.803, 6: 0.766, 7: 0.747, 8: 0.745, 9: 0.737, 10: 0.721, 11: 0.73, 12: 0.73}, ('rp', False, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 1.0, 2: 0.997, 3: 0.988, 4: 0.965, 5: 0.94, 6: 0.896, 7: 0.866, 8: 0.816, 9: 0.75, 10: 0.71, 11: 0.678, 12: 0.676}, ('lp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 1.0, 1: 0.983, 2: 0.965, 3: 0.919, 4: 0.843, 5: 0.768, 6: 0.696, 8: 0.62, 7: 0.69, 9: 0.606, 10: 0.592, 11: 0.576, 12: 0.567}, ('lp_star', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.996, 1: 0.987, 2: 0.984, 3: 0.962, 4: 0.933, 5: 0.901, 6: 0.869, 7: 0.857, 8: 0.836, 9: 0.809, 10: 0.773, 11: 0.765, 12: 0.742}, ('rp', True, 'bidir', 'corrective', 'r2', 'rp'): {0: 0.998, 1: 0.988, 2: 0.989, 3: 0.969, 4: 0.938, 5: 0.919, 6: 0.879, 7: 0.864, 8: 0.82, 9: 0.749, 10: 0.712, 11: 0.693, 12: 0.675}, ('lp', False, 'bidir', 'direct', 'r2', 'rp'): {0: 0.997, 1: 0.707, 2: 0.597, 3: 0.562, 4: 0.54, 5: 0.514, 6: 0.524, 8: 0.516, 7: 0.543, 9: 0.552, 10: 0.541, 11: 0.552, 12: 0.552}, ('lp_star', False, 'bidir', 'direct', 'r2', 'rp'): {0: 0.994, 1: 0.877, 2: 0.719, 3: 0.638, 4: 0.576, 5: 0.56, 6: 0.542, 7: 0.519, 8: 0.538, 9: 0.513, 10: 0.51, 11: 0.52, 12: 0.533}, ('rp', False, 'bidir', 'direct', 'r2', 'rp'): {0: 1.0, 1: 0.915, 2: 0.861, 3: 0.803, 4: 0.812, 5: 0.717, 6: 0.719, 7: 0.733, 8: 0.658, 9: 0.635, 10: 0.635, 11: 0.609, 12: 0.66}, ('lp', True, 'cot', 'bidir', 'r2', 'rp'): {0: 0.994, 1: 0.977, 2: 0.935, 3: 0.86, 4: 0.772, 5: 0.7, 6: 0.647, 8: 0.562, 7: 0.599, 9: 0.57, 10: 0.557, 11: 0.539, 12: 0.524}, ('lp_star', True, 'cot', 'bidir', 'r2', 'rp'): {0: 0.991, 1: 0.97, 2: 0.983, 3: 0.944, 4: 0.897, 5: 0.866, 6: 0.822, 7: 0.815, 8: 0.787, 9: 0.771, 10: 0.76, 11: 0.725, 12: 0.714}, ('rp', True, 'cot', 'bidir', 'r2', 'rp'): {0: 0.999, 1: 0.997, 2: 0.994, 3: 0.971, 4: 0.941, 5: 0.887, 6: 0.857, 7: 0.803, 8: 0.753, 9: 0.665, 10: 0.666, 11: 0.612, 12: 0.573}, ('lp', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.998, 1: 0.975, 2: 0.932, 3: 0.827, 4: 0.713, 5: 0.614, 6: 0.544, 8: 0.485, 7: 0.492, 9: 0.531, 10: 0.546, 11: 0.521, 12: 0.499}, ('lp_star', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.997, 1: 1.0, 2: 0.992, 3: 0.957, 4: 0.911, 5: 0.813, 6: 0.776, 7: 0.736, 8: 0.709, 9: 0.706, 10: 0.696, 11: 0.685, 12: 0.691}, ('rp', False, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 1.0, 2: 0.994, 3: 0.988, 4: 0.965, 5: 0.947, 6: 0.923, 7: 0.875, 8: 0.822, 9: 0.782, 10: 0.731, 11: 0.703, 12: 0.683}, ('lp', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.996, 1: 0.978, 2: 0.959, 3: 0.906, 4: 0.819, 5: 0.693, 6: 0.618, 8: 0.572, 7: 0.599, 9: 0.558, 10: 0.531, 11: 0.523, 12: 0.513}, ('lp_star', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 0.995, 1: 0.982, 2: 0.984, 3: 0.96, 4: 0.922, 5: 0.872, 6: 0.818, 7: 0.788, 8: 0.772, 9: 0.738, 10: 0.718, 11: 0.701, 12: 0.691}, ('rp', True, 'cot', 'bidir', 'r2', 'rp', 'direct'): {0: 1.0, 1: 0.991, 2: 0.987, 3: 0.967, 4: 0.949, 5: 0.916, 6: 0.844, 7: 0.813, 8: 0.775, 9: 0.708, 10: 0.68, 11: 0.632, 12: 0.601}}
# print(generate_main_tables(raw_scores_baseline_deep_30 | raw_scores_mixed_deep_30, raw_scores_baseline_deep_60 | raw_scores_mixed_deep_60))

###### r2 dataset size ablation ######
# print("raw_scores_r2half_seeds_deep_30=", get_raw_scores([x for x in repo.entries if {'rp', 'r2half'} <= x.tags], pred_count=30))
raw_scores_r2half_seeds_deep_30= {('lp', False, 'r2half', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.993, 2: 0.959, 3: 0.86, 4: 0.756, 5: 0.698, 6: 0.6, 7: 0.592, 8: 0.59, 9: 0.585, 10: 0.602, 11: 0.609, 12: 0.612}, ('lp_star', False, 'r2half', 'corrective', 'rp', 'bidir'): {0: 0.999, 1: 1.0, 2: 0.999, 3: 0.986, 4: 0.962, 5: 0.921, 6: 0.865, 7: 0.847, 8: 0.811, 9: 0.817, 10: 0.819, 11: 0.818, 12: 0.797}, ('rp', False, 'r2half', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 0.997, 3: 0.988, 4: 0.976, 5: 0.957, 6: 0.925, 7: 0.887, 8: 0.83, 9: 0.756, 10: 0.678, 11: 0.68, 12: 0.636}, ('lp', True, 'r2half', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.991, 3: 0.957, 4: 0.882, 5: 0.845, 6: 0.797, 7: 0.808, 8: 0.745, 9: 0.748, 10: 0.726, 11: 0.713, 12: 0.695}, ('lp_star', True, 'r2half', 'corrective', 'rp', 'bidir'): {0: 0.999, 1: 0.996, 2: 0.997, 3: 0.994, 4: 0.99, 5: 0.991, 6: 0.973, 7: 0.965, 8: 0.958, 9: 0.967, 10: 0.943, 11: 0.944, 12: 0.944}, ('rp', True, 'r2half', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.989, 6: 0.975, 7: 0.96, 8: 0.938, 9: 0.904, 10: 0.864, 11: 0.852, 12: 0.804}, ('lp', False, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.993, 2: 0.968, 3: 0.875, 4: 0.775, 5: 0.712, 6: 0.659, 7: 0.64, 8: 0.621, 9: 0.65, 10: 0.635, 11: 0.659, 12: 0.649}, ('lp_star', False, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.992, 4: 0.966, 5: 0.92, 6: 0.849, 7: 0.826, 8: 0.801, 9: 0.799, 10: 0.821, 11: 0.805, 12: 0.804}, ('rp', False, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.995, 4: 0.977, 5: 0.962, 6: 0.891, 7: 0.831, 8: 0.756, 9: 0.679, 10: 0.602, 11: 0.602, 12: 0.57}, ('lp', True, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.997, 2: 0.993, 3: 0.964, 4: 0.907, 5: 0.868, 6: 0.826, 7: 0.821, 8: 0.809, 9: 0.771, 10: 0.787, 11: 0.753, 12: 0.724}, ('lp_star', True, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 0.998, 4: 0.995, 5: 0.985, 6: 0.979, 7: 0.961, 8: 0.964, 9: 0.97, 10: 0.962, 11: 0.964, 12: 0.96}, ('rp', True, 'r2half', 'seed124', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.996, 5: 0.987, 6: 0.983, 7: 0.976, 8: 0.94, 9: 0.908, 10: 0.879, 11: 0.854, 12: 0.838}, ('lp', False, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 0.988, 2: 0.952, 3: 0.908, 4: 0.799, 5: 0.727, 6: 0.653, 7: 0.657, 8: 0.6, 9: 0.598, 10: 0.607, 11: 0.621, 12: 0.613}, ('lp_star', False, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 0.997, 1: 0.998, 2: 0.998, 3: 0.996, 4: 0.958, 5: 0.908, 6: 0.834, 7: 0.792, 8: 0.744, 9: 0.737, 10: 0.712, 11: 0.717, 12: 0.697}, ('rp', False, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 0.998, 2: 0.998, 3: 0.993, 4: 0.981, 5: 0.958, 6: 0.94, 7: 0.901, 8: 0.835, 9: 0.753, 10: 0.671, 11: 0.65, 12: 0.614}, ('lp', True, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 0.998, 2: 0.993, 3: 0.975, 4: 0.942, 5: 0.889, 6: 0.855, 7: 0.86, 8: 0.814, 9: 0.835, 10: 0.805, 11: 0.806, 12: 0.802}, ('lp_star', True, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 0.991, 1: 0.988, 2: 0.998, 3: 0.998, 4: 0.998, 5: 0.993, 6: 0.994, 7: 0.992, 8: 0.981, 9: 0.975, 10: 0.971, 11: 0.967, 12: 0.963}, ('rp', True, 'r2half', 'corrective', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.997, 5: 0.995, 6: 0.984, 7: 0.979, 8: 0.957, 9: 0.937, 10: 0.926, 11: 0.913, 12: 0.884}}
# print("raw_scores_r2half_seeds_deep_60=", get_raw_scores([x for x in repo.entries if {'rp', 'r2half'} <= x.tags], pred_count=60))
raw_scores_r2half_seeds_deep_60= {('lp', False, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.996, 1: 0.97, 2: 0.91, 3: 0.75, 4: 0.645, 5: 0.547, 6: 0.545, 8: 0.514, 7: 0.483, 9: 0.525, 10: 0.526, 11: 0.522, 12: 0.528}, ('lp_star', False, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.996, 1: 0.978, 2: 0.968, 3: 0.906, 4: 0.856, 5: 0.776, 6: 0.734, 7: 0.698, 8: 0.675, 9: 0.687, 10: 0.674, 11: 0.675, 12: 0.683}, ('rp', False, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.984, 1: 0.963, 2: 0.965, 3: 0.94, 4: 0.93, 5: 0.876, 6: 0.85, 7: 0.802, 8: 0.729, 9: 0.694, 10: 0.657, 11: 0.641, 12: 0.619}, ('lp', True, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.999, 1: 0.972, 2: 0.933, 3: 0.874, 4: 0.767, 5: 0.69, 6: 0.662, 8: 0.579, 7: 0.587, 9: 0.554, 10: 0.553, 11: 0.529, 12: 0.521}, ('lp_star', True, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.993, 1: 0.961, 2: 0.963, 3: 0.931, 4: 0.883, 5: 0.86, 6: 0.828, 7: 0.781, 8: 0.765, 9: 0.744, 10: 0.694, 11: 0.693, 12: 0.683}, ('rp', True, 'corrective', 'rp', 'r2half', 'bidir'): {0: 0.997, 1: 0.996, 2: 0.989, 3: 0.972, 4: 0.949, 5: 0.908, 6: 0.851, 7: 0.795, 8: 0.756, 9: 0.686, 10: 0.65, 11: 0.624, 12: 0.601}, ('lp', False, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 1.0, 1: 0.992, 2: 0.941, 3: 0.8, 4: 0.713, 5: 0.628, 6: 0.57, 8: 0.523, 7: 0.541, 9: 0.568, 10: 0.572, 11: 0.57, 12: 0.589}, ('lp_star', False, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 0.942, 4: 0.888, 5: 0.798, 6: 0.746, 7: 0.731, 8: 0.696, 9: 0.684, 10: 0.7, 11: 0.699, 12: 0.708}, ('rp', False, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 1.0, 1: 0.998, 2: 1.0, 3: 0.984, 4: 0.973, 5: 0.93, 6: 0.866, 7: 0.818, 8: 0.739, 9: 0.686, 10: 0.645, 11: 0.598, 12: 0.587}, ('lp', True, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 0.998, 1: 0.978, 2: 0.943, 3: 0.881, 4: 0.797, 5: 0.69, 6: 0.671, 8: 0.583, 7: 0.628, 9: 0.566, 10: 0.581, 11: 0.572, 12: 0.551}, ('lp_star', True, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 0.999, 1: 0.986, 2: 0.992, 3: 0.95, 4: 0.923, 5: 0.873, 6: 0.821, 7: 0.802, 8: 0.797, 9: 0.758, 10: 0.734, 11: 0.736, 12: 0.707}, ('rp', True, 'bidir', 'rp', 'r2half', 'seed124', 'corrective'): {0: 0.995, 1: 0.966, 2: 0.977, 3: 0.96, 4: 0.935, 5: 0.903, 6: 0.862, 7: 0.829, 8: 0.784, 9: 0.721, 10: 0.673, 11: 0.647, 12: 0.618}, ('lp', False, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.998, 1: 0.971, 2: 0.918, 3: 0.811, 4: 0.716, 5: 0.63, 6: 0.564, 8: 0.537, 7: 0.526, 9: 0.552, 10: 0.569, 11: 0.549, 12: 0.55}, ('lp_star', False, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.996, 1: 0.967, 2: 0.969, 3: 0.915, 4: 0.847, 5: 0.774, 6: 0.689, 7: 0.647, 8: 0.638, 9: 0.614, 10: 0.606, 11: 0.604, 12: 0.61}, ('rp', False, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.999, 1: 1.0, 2: 0.996, 3: 0.976, 4: 0.965, 5: 0.922, 6: 0.888, 7: 0.829, 8: 0.782, 9: 0.699, 10: 0.675, 11: 0.631, 12: 0.608}, ('lp', True, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.992, 1: 0.98, 2: 0.937, 3: 0.891, 4: 0.805, 5: 0.7, 6: 0.696, 8: 0.609, 7: 0.658, 9: 0.595, 10: 0.591, 11: 0.582, 12: 0.542}, ('lp_star', True, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.987, 1: 0.955, 2: 0.966, 3: 0.925, 4: 0.903, 5: 0.873, 6: 0.835, 7: 0.803, 8: 0.793, 9: 0.741, 10: 0.708, 11: 0.707, 12: 0.669}, ('rp', True, 'bidir', 'seed125', 'rp', 'r2half', 'corrective'): {0: 0.998, 1: 0.99, 2: 0.983, 3: 0.963, 4: 0.935, 5: 0.891, 6: 0.858, 7: 0.818, 8: 0.792, 9: 0.7, 10: 0.668, 11: 0.637, 12: 0.608}}
# print("raw_scores_nor2_seeds_deep_30=", get_raw_scores([x for x in repo.entries if {'rp', 'nor2'} <= x.tags], pred_count=30))
raw_scores_nor2_seeds_deep_30= {('lp', False, 'nor2', 'corrective', 'rp', 'bidir'): {0: 0.993, 1: 0.982, 2: 0.956, 3: 0.818, 4: 0.727, 5: 0.65, 6: 0.547, 7: 0.543, 8: 0.554, 9: 0.567, 10: 0.552, 11: 0.567, 12: 0.54}, ('lp_star', False, 'nor2', 'corrective', 'rp', 'bidir'): {0: 0.999, 1: 0.998, 2: 1.0, 3: 0.985, 4: 0.971, 5: 0.905, 6: 0.836, 7: 0.795, 8: 0.794, 9: 0.747, 10: 0.737, 11: 0.704, 12: 0.68}, ('rp', False, 'nor2', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.997, 3: 0.992, 4: 0.984, 5: 0.971, 6: 0.945, 7: 0.91, 8: 0.836, 9: 0.754, 10: 0.664, 11: 0.652, 12: 0.597}, ('lp', True, 'nor2', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.998, 2: 0.99, 3: 0.972, 4: 0.923, 5: 0.885, 6: 0.829, 7: 0.832, 8: 0.766, 9: 0.765, 10: 0.731, 11: 0.731, 12: 0.688}, ('lp_star', True, 'nor2', 'corrective', 'rp', 'bidir'): {0: 0.993, 1: 0.992, 2: 1.0, 3: 0.999, 4: 0.996, 5: 0.991, 6: 0.984, 7: 0.973, 8: 0.979, 9: 0.963, 10: 0.938, 11: 0.946, 12: 0.938}, ('rp', True, 'nor2', 'corrective', 'rp', 'bidir'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.995, 6: 0.985, 7: 0.978, 8: 0.967, 9: 0.957, 10: 0.917, 11: 0.899, 12: 0.851}, ('lp', False, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.995, 1: 0.917, 2: 0.798, 3: 0.648, 4: 0.554, 5: 0.508, 6: 0.499, 7: 0.497, 8: 0.554, 9: 0.552, 10: 0.557, 11: 0.555, 12: 0.547}, ('lp_star', False, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.998, 1: 1.0, 2: 0.981, 3: 0.913, 4: 0.859, 5: 0.821, 6: 0.808, 7: 0.797, 8: 0.814, 9: 0.783, 10: 0.786, 11: 0.767, 12: 0.748}, ('rp', False, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.999, 1: 0.999, 2: 0.982, 3: 0.963, 4: 0.942, 5: 0.913, 6: 0.905, 7: 0.873, 8: 0.832, 9: 0.767, 10: 0.716, 11: 0.692, 12: 0.635}, ('lp', True, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.999, 1: 0.983, 2: 0.935, 3: 0.875, 4: 0.793, 5: 0.734, 6: 0.7, 7: 0.685, 8: 0.675, 9: 0.648, 10: 0.65, 11: 0.634, 12: 0.64}, ('lp_star', True, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.999, 1: 0.997, 2: 0.989, 3: 0.976, 4: 0.958, 5: 0.936, 6: 0.918, 7: 0.906, 8: 0.893, 9: 0.884, 10: 0.855, 11: 0.844, 12: 0.843}, ('rp', True, 'seed124', 'corrective', 'nor2', 'rp', 'bidir'): {0: 0.999, 1: 0.999, 2: 1.0, 3: 0.988, 4: 0.984, 5: 0.968, 6: 0.936, 7: 0.936, 8: 0.91, 9: 0.844, 10: 0.792, 11: 0.763, 12: 0.737}, ('lp', False, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 0.996, 1: 0.981, 2: 0.947, 3: 0.817, 4: 0.692, 5: 0.604, 6: 0.526, 7: 0.511, 8: 0.497, 9: 0.537, 10: 0.552, 11: 0.558, 12: 0.568}, ('lp_star', False, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 0.988, 4: 0.98, 5: 0.924, 6: 0.878, 7: 0.843, 8: 0.828, 9: 0.824, 10: 0.819, 11: 0.799, 12: 0.772}, ('rp', False, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.998, 3: 0.988, 4: 0.985, 5: 0.964, 6: 0.954, 7: 0.924, 8: 0.866, 9: 0.799, 10: 0.703, 11: 0.68, 12: 0.617}, ('lp', True, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 0.998, 2: 0.991, 3: 0.971, 4: 0.913, 5: 0.872, 6: 0.823, 7: 0.825, 8: 0.787, 9: 0.758, 10: 0.738, 11: 0.717, 12: 0.662}, ('lp_star', True, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 0.992, 1: 0.998, 2: 0.998, 3: 0.994, 4: 0.995, 5: 0.995, 6: 0.979, 7: 0.979, 8: 0.974, 9: 0.967, 10: 0.966, 11: 0.954, 12: 0.949}, ('rp', True, 'corrective', 'nor2', 'seed125', 'rp', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.996, 6: 0.988, 7: 0.987, 8: 0.966, 9: 0.956, 10: 0.929, 11: 0.92, 12: 0.882}}
# print("raw_scores_nor2_seeds_deep_60=", get_raw_scores([x for x in repo.entries if {'rp', 'nor2'} <= x.tags], pred_count=60))
raw_scores_nor2_seeds_deep_60= {('lp', False, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.987, 1: 0.948, 2: 0.898, 3: 0.728, 4: 0.648, 5: 0.53, 6: 0.484, 8: 0.457, 7: 0.468, 9: 0.489, 10: 0.478, 11: 0.472, 12: 0.504}, ('lp_star', False, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.997, 1: 0.988, 2: 0.983, 3: 0.896, 4: 0.829, 5: 0.731, 6: 0.667, 7: 0.641, 8: 0.64, 9: 0.617, 10: 0.589, 11: 0.595, 12: 0.591}, ('rp', False, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.999, 1: 0.998, 2: 0.996, 3: 0.983, 4: 0.977, 5: 0.946, 6: 0.899, 7: 0.861, 8: 0.769, 9: 0.699, 10: 0.667, 11: 0.613, 12: 0.574}, ('lp', True, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.996, 1: 0.971, 2: 0.947, 3: 0.873, 4: 0.778, 5: 0.665, 6: 0.641, 8: 0.546, 7: 0.58, 9: 0.53, 10: 0.502, 11: 0.489, 12: 0.505}, ('lp_star', True, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.985, 1: 0.97, 2: 0.98, 3: 0.935, 4: 0.896, 5: 0.851, 6: 0.808, 7: 0.764, 8: 0.747, 9: 0.715, 10: 0.671, 11: 0.665, 12: 0.618}, ('rp', True, 'nor2', 'rp', 'corrective', 'bidir'): {0: 0.999, 1: 0.984, 2: 0.99, 3: 0.98, 4: 0.965, 5: 0.928, 6: 0.869, 7: 0.856, 8: 0.797, 9: 0.724, 10: 0.665, 11: 0.636, 12: 0.586}, ('lp', False, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 0.972, 1: 0.887, 2: 0.753, 3: 0.604, 4: 0.55, 5: 0.526, 6: 0.522, 8: 0.491, 7: 0.501, 9: 0.505, 10: 0.505, 11: 0.482, 12: 0.487}, ('lp_star', False, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 0.999, 1: 0.977, 2: 0.928, 3: 0.803, 4: 0.744, 5: 0.699, 6: 0.687, 7: 0.68, 8: 0.697, 9: 0.683, 10: 0.682, 11: 0.67, 12: 0.689}, ('rp', False, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 0.996, 1: 0.995, 2: 0.973, 3: 0.956, 4: 0.942, 5: 0.878, 6: 0.852, 7: 0.824, 8: 0.771, 9: 0.735, 10: 0.666, 11: 0.65, 12: 0.611}, ('lp', True, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 1.0, 1: 0.941, 2: 0.868, 3: 0.768, 4: 0.673, 5: 0.603, 6: 0.566, 8: 0.511, 7: 0.539, 9: 0.508, 10: 0.507, 11: 0.502, 12: 0.493}, ('lp_star', True, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 1.0, 1: 0.981, 2: 0.946, 3: 0.883, 4: 0.824, 5: 0.787, 6: 0.744, 7: 0.715, 8: 0.72, 9: 0.683, 10: 0.661, 11: 0.64, 12: 0.648}, ('rp', True, 'bidir', 'nor2', 'rp', 'seed124', 'corrective'): {0: 1.0, 1: 0.997, 2: 0.989, 3: 0.952, 4: 0.924, 5: 0.872, 6: 0.815, 7: 0.791, 8: 0.715, 9: 0.669, 10: 0.634, 11: 0.63, 12: 0.597}, ('lp', False, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.981, 1: 0.955, 2: 0.904, 3: 0.716, 4: 0.61, 5: 0.527, 6: 0.46, 8: 0.457, 7: 0.475, 9: 0.496, 10: 0.486, 11: 0.499, 12: 0.482}, ('lp_star', False, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.99, 1: 0.996, 2: 0.988, 3: 0.915, 4: 0.86, 5: 0.79, 6: 0.733, 7: 0.698, 8: 0.707, 9: 0.691, 10: 0.685, 11: 0.693, 12: 0.689}, ('rp', False, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.999, 1: 0.999, 2: 0.997, 3: 0.982, 4: 0.964, 5: 0.935, 6: 0.9, 7: 0.846, 8: 0.806, 9: 0.718, 10: 0.681, 11: 0.631, 12: 0.608}, ('lp', True, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.995, 1: 0.965, 2: 0.941, 3: 0.886, 4: 0.781, 5: 0.686, 6: 0.635, 8: 0.56, 7: 0.59, 9: 0.537, 10: 0.523, 11: 0.514, 12: 0.508}, ('lp_star', True, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.99, 1: 0.979, 2: 0.977, 3: 0.952, 4: 0.916, 5: 0.882, 6: 0.822, 7: 0.804, 8: 0.788, 9: 0.75, 10: 0.722, 11: 0.701, 12: 0.688}, ('rp', True, 'bidir', 'nor2', 'seed125', 'rp', 'corrective'): {0: 0.999, 1: 0.988, 2: 0.988, 3: 0.97, 4: 0.944, 5: 0.92, 6: 0.879, 7: 0.842, 8: 0.794, 9: 0.72, 10: 0.715, 11: 0.664, 12: 0.635}}
r2_impact_30 = raw_scores_baseline_seeds_deep_30 | raw_scores_r2half_seeds_deep_30 | raw_scores_nor2_seeds_deep_30
r2_impact_60 = raw_scores_baseline_seeds_deep_60 | raw_scores_r2half_seeds_deep_60 | raw_scores_nor2_seeds_deep_60
# print(generate_r2_ablation_tables(r2_impact_30, r2_impact_60))


###### Small ######
# print("raw_scores_small=", get_ablation_scores([x for x in repo.entries if {'rp', 'small'} <= x.tags], include_cot=False))
raw_scores_small= {('validation_rp_balanced', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.978, 2: 0.911, 3: 0.902, 4: 0.861, 5: 0.834, 6: 0.806}, ('validation_lp_balanced', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.998, 1: 0.833, 2: 0.719, 3: 0.683, 4: 0.605, 5: 0.583, 6: 0.569}, ('validation_rp_balanced_3_premise', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.975, 2: 0.873, 3: 0.724, 4: 0.653, 5: 0.607, 6: 0.559}, ('validation_rp_balanced_2_premise', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.966, 2: 0.917, 3: 0.888, 4: 0.841, 5: 0.784, 6: 0.757}, ('validation_rp_balanced_1_premise', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.925, 1: 0.991, 2: 0.979, 3: 0.97, 4: 0.935, 5: 0.911, 6: 0.915}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.998, 1: 0.967, 2: 0.952, 3: 0.907, 4: 0.889, 5: 0.879, 6: 0.856}, ('validation_rp_balanced', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.977, 2: 0.922, 3: 0.899, 4: 0.856, 5: 0.833, 6: 0.82}, ('validation_lp_balanced', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.992, 1: 0.827, 2: 0.7, 3: 0.638, 4: 0.587, 5: 0.545, 6: 0.557}, ('validation_rp_balanced_3_premise', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.967, 2: 0.899, 3: 0.781, 4: 0.672, 5: 0.628, 6: 0.569}, ('validation_rp_balanced_2_premise', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.976, 2: 0.929, 3: 0.894, 4: 0.841, 5: 0.8, 6: 0.76}, ('validation_rp_balanced_1_premise', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.871, 1: 0.985, 2: 0.957, 3: 0.941, 4: 0.931, 5: 0.923, 6: 0.931}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'seed', 'seed124', 'small', 'corrective', 'rp', 'layers=4', 'bidir'): {0: 0.997, 1: 0.983, 2: 0.953, 3: 0.92, 4: 0.896, 5: 0.893, 6: 0.868}, ('validation_rp_balanced', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.978, 2: 0.931, 3: 0.899, 4: 0.862, 5: 0.852, 6: 0.8}, ('validation_lp_balanced', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 0.996, 1: 0.872, 2: 0.714, 3: 0.617, 4: 0.608, 5: 0.6, 6: 0.59}, ('validation_rp_balanced_3_premise', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.971, 2: 0.898, 3: 0.829, 4: 0.784, 5: 0.717, 6: 0.682}, ('validation_rp_balanced_2_premise', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.969, 2: 0.932, 3: 0.886, 4: 0.854, 5: 0.814, 6: 0.785}, ('validation_rp_balanced_1_premise', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 0.987, 1: 0.983, 2: 0.961, 3: 0.951, 4: 0.936, 5: 0.921, 6: 0.922}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'seed', 'small', 'corrective', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 0.993, 1: 0.975, 2: 0.947, 3: 0.907, 4: 0.916, 5: 0.902, 6: 0.891}, ('validation_rp_balanced', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.939, 2: 0.863, 3: 0.824, 4: 0.813, 5: 0.81, 6: 0.777}, ('validation_lp_balanced', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.752, 2: 0.595, 3: 0.583, 4: 0.561, 5: 0.561, 6: 0.568}, ('validation_rp_balanced_3_premise', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.94, 2: 0.83, 3: 0.759, 4: 0.75, 5: 0.684, 6: 0.649}, ('validation_rp_balanced_2_premise', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.936, 2: 0.86, 3: 0.805, 4: 0.782, 5: 0.758, 6: 0.753}, ('validation_rp_balanced_1_premise', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.999, 1: 0.971, 2: 0.947, 3: 0.943, 4: 0.934, 5: 0.931, 6: 0.936}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'direct', 'small', 'rp', 'layers=4', 'bidir'): {0: 1.0, 1: 0.933, 2: 0.9, 3: 0.869, 4: 0.871, 5: 0.86, 6: 0.869}, ('validation_rp_balanced', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.758, 1: 0.587, 2: 0.495, 3: 0.536, 4: 0.509, 5: 0.512, 6: 0.501}, ('validation_lp_balanced', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.655, 1: 0.551, 2: 0.525, 3: 0.541, 4: 0.527, 5: 0.531, 6: 0.519}, ('validation_rp_balanced_3_premise', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.638, 1: 0.575, 2: 0.546, 3: 0.539, 4: 0.543, 5: 0.513, 6: 0.501}, ('validation_rp_balanced_2_premise', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.762, 1: 0.596, 2: 0.519, 3: 0.506, 4: 0.504, 5: 0.473, 6: 0.506}, ('validation_rp_balanced_1_premise', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.871, 1: 0.581, 2: 0.535, 3: 0.501, 4: 0.498, 5: 0.502, 6: 0.499}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'direct', 'seed', 'seed124', 'small', 'rp', 'layers=4', 'bidir'): {0: 0.852, 1: 0.615, 2: 0.514, 3: 0.502, 4: 0.474, 5: 0.479, 6: 0.505}, ('validation_rp_balanced', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.88, 2: 0.792, 3: 0.77, 4: 0.756, 5: 0.751, 6: 0.727}, ('validation_lp_balanced', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.623, 2: 0.561, 3: 0.539, 4: 0.53, 5: 0.532, 6: 0.53}, ('validation_rp_balanced_3_premise', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.845, 2: 0.782, 3: 0.693, 4: 0.676, 5: 0.617, 6: 0.572}, ('validation_rp_balanced_2_premise', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.867, 2: 0.797, 3: 0.769, 4: 0.731, 5: 0.735, 6: 0.745}, ('validation_rp_balanced_1_premise', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 0.999, 1: 0.924, 2: 0.878, 3: 0.878, 4: 0.873, 5: 0.866, 6: 0.866}, ('validation_rp_balanced_1_2_premise', False, 'r2', 'direct', 'seed', 'small', 'rp', 'layers=4', 'seed125', 'bidir'): {0: 1.0, 1: 0.877, 2: 0.839, 3: 0.816, 4: 0.805, 5: 0.807, 6: 0.83}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'rp', 'layers=4', 'corrective', 'small'): {0: 1.0, 1: 0.98, 2: 0.933, 3: 0.864, 4: 0.765, 5: 0.753, 6: 0.679}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'layers=4', 'seed124', 'corrective', 'small'): {0: 1.0, 1: 0.977, 2: 0.939, 3: 0.867, 4: 0.796, 5: 0.731, 6: 0.673}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'layers=4', 'corrective', 'seed125', 'small'): {0: 1.0, 1: 0.967, 2: 0.923, 3: 0.88, 4: 0.81, 5: 0.79, 6: 0.727}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'rp', 'layers=4', 'direct', 'small'): {0: 0.998, 1: 0.927, 2: 0.862, 3: 0.797, 4: 0.761, 5: 0.729, 6: 0.696}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'layers=4', 'seed124', 'direct', 'small'): {0: 0.665, 1: 0.564, 2: 0.54, 3: 0.526, 4: 0.497, 5: 0.523, 6: 0.5}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'layers=4', 'seed125', 'direct', 'small'): {0: 1.0, 1: 0.841, 2: 0.801, 3: 0.768, 4: 0.708, 5: 0.705, 6: 0.669},
                   ('validation_rp_balanced', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 1.0, 1: 0.977, 2: 0.894, 3: 0.834, 4: 0.813, 5: 0.775, 6: 0.742}, ('validation_lp_balanced', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 0.998, 1: 0.807, 2: 0.615, 3: 0.541, 4: 0.545, 5: 0.526, 6: 0.561}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 1.0, 1: 0.914, 2: 0.819, 3: 0.713, 4: 0.662, 5: 0.6, 6: 0.57}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 1.0, 1: 0.969, 2: 0.895, 3: 0.816, 4: 0.756, 5: 0.707, 6: 0.677}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 0.897, 1: 0.98, 2: 0.942, 3: 0.928, 4: 0.899, 5: 0.889, 6: 0.88}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 0.997, 1: 0.966, 2: 0.92, 3: 0.882, 4: 0.853, 5: 0.826, 6: 0.815}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=4'): {0: 1.0, 1: 0.936, 2: 0.858, 3: 0.767, 4: 0.713, 5: 0.677, 6: 0.614}, ('validation_rp_balanced', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 0.997, 1: 0.945, 2: 0.874, 3: 0.81, 4: 0.789, 5: 0.774, 6: 0.739}, ('validation_lp_balanced', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 0.997, 1: 0.767, 2: 0.622, 3: 0.577, 4: 0.565, 5: 0.567, 6: 0.561}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 0.999, 1: 0.944, 2: 0.829, 3: 0.735, 4: 0.706, 5: 0.641, 6: 0.614}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 1.0, 1: 0.948, 2: 0.87, 3: 0.814, 4: 0.771, 5: 0.751, 6: 0.716}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 0.938, 1: 0.948, 2: 0.917, 3: 0.904, 4: 0.883, 5: 0.869, 6: 0.891}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 0.996, 1: 0.937, 2: 0.906, 3: 0.861, 4: 0.85, 5: 0.833, 6: 0.803}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'seed', 'layers=4'): {0: 1.0, 1: 0.927, 2: 0.856, 3: 0.793, 4: 0.729, 5: 0.691, 6: 0.642}, ('validation_rp_balanced', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.934, 2: 0.86, 3: 0.829, 4: 0.814, 5: 0.791, 6: 0.76}, ('validation_lp_balanced', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 0.997, 1: 0.732, 2: 0.595, 3: 0.577, 4: 0.565, 5: 0.545, 6: 0.558}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 0.997, 1: 0.9, 2: 0.817, 3: 0.724, 4: 0.718, 5: 0.646, 6: 0.617}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.934, 2: 0.878, 3: 0.822, 4: 0.773, 5: 0.744, 6: 0.725}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 0.958, 1: 0.976, 2: 0.953, 3: 0.939, 4: 0.909, 5: 0.895, 6: 0.864}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.947, 2: 0.915, 3: 0.872, 4: 0.865, 5: 0.851, 6: 0.853}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'rp', 'corrective', 'r2', 'bidir', 'small', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.909, 2: 0.84, 3: 0.802, 4: 0.752, 5: 0.723, 6: 0.646}, ('validation_rp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 1.0, 1: 0.947, 2: 0.872, 3: 0.828, 4: 0.812, 5: 0.829, 6: 0.767}, ('validation_lp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 1.0, 1: 0.745, 2: 0.597, 3: 0.588, 4: 0.564, 5: 0.541, 6: 0.561}, ('validation_rp_balanced_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 0.999, 1: 0.925, 2: 0.847, 3: 0.783, 4: 0.752, 5: 0.691, 6: 0.661}, ('validation_rp_balanced_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 1.0, 1: 0.941, 2: 0.875, 3: 0.825, 4: 0.802, 5: 0.801, 6: 0.764}, ('validation_rp_balanced_1_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 0.968, 1: 0.971, 2: 0.942, 3: 0.933, 4: 0.908, 5: 0.889, 6: 0.875}, ('validation_rp_balanced_1_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 1.0, 1: 0.939, 2: 0.91, 3: 0.87, 4: 0.875, 5: 0.864, 6: 0.855}, ('validation_rp_balanced_2_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'layers=4'): {0: 1.0, 1: 0.932, 2: 0.872, 3: 0.818, 4: 0.789, 5: 0.742, 6: 0.703}, ('validation_rp_balanced', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 1.0, 1: 0.984, 2: 0.928, 3: 0.914, 4: 0.877, 5: 0.862, 6: 0.819}, ('validation_lp_balanced', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 0.991, 1: 0.879, 2: 0.724, 3: 0.664, 4: 0.605, 5: 0.591, 6: 0.578}, ('validation_rp_balanced_3_premise', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 0.998, 1: 0.973, 2: 0.918, 3: 0.856, 4: 0.781, 5: 0.719, 6: 0.674}, ('validation_rp_balanced_2_premise', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 0.998, 1: 0.977, 2: 0.941, 3: 0.897, 4: 0.886, 5: 0.829, 6: 0.79}, ('validation_rp_balanced_1_premise', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 0.907, 1: 0.989, 2: 0.976, 3: 0.966, 4: 0.947, 5: 0.947, 6: 0.944}, ('validation_rp_balanced_1_2_premise', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 0.984, 1: 0.976, 2: 0.955, 3: 0.923, 4: 0.918, 5: 0.902, 6: 0.906}, ('validation_rp_balanced_2_3_premise', False, 'rp', 'corrective', 'seed124', 'r2', 'bidir', 'small', 'dim=512', 'seed', 'layers=4'): {0: 1.0, 1: 0.979, 2: 0.934, 3: 0.904, 4: 0.846, 5: 0.817, 6: 0.746}, ('validation_rp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.979, 2: 0.935, 3: 0.918, 4: 0.875, 5: 0.857, 6: 0.799}, ('validation_lp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 0.996, 1: 0.856, 2: 0.747, 3: 0.659, 4: 0.599, 5: 0.581, 6: 0.579}, ('validation_rp_balanced_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.967, 2: 0.941, 3: 0.855, 4: 0.811, 5: 0.747, 6: 0.67}, ('validation_rp_balanced_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.972, 2: 0.945, 3: 0.896, 4: 0.815, 5: 0.782, 6: 0.722}, ('validation_rp_balanced_1_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 0.853, 1: 0.985, 2: 0.967, 3: 0.954, 4: 0.937, 5: 0.941, 6: 0.936}, ('validation_rp_balanced_1_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 0.996, 1: 0.979, 2: 0.959, 3: 0.92, 4: 0.908, 5: 0.88, 6: 0.85}, ('validation_rp_balanced_2_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=512', 'seed125', 'seed', 'layers=4'): {0: 1.0, 1: 0.964, 2: 0.939, 3: 0.895, 4: 0.8, 5: 0.778, 6: 0.709},
                   ('validation_rp_balanced', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.984, 1: 0.796, 2: 0.722, 3: 0.7, 4: 0.682, 5: 0.668, 6: 0.657}, ('validation_lp_balanced', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.903, 1: 0.618, 2: 0.557, 3: 0.543, 4: 0.531, 5: 0.537, 6: 0.535}, ('validation_rp_balanced_3_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.96, 1: 0.738, 2: 0.691, 3: 0.663, 4: 0.643, 5: 0.601, 6: 0.602}, ('validation_rp_balanced_2_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.975, 1: 0.811, 2: 0.716, 3: 0.692, 4: 0.643, 5: 0.699, 6: 0.678}, ('validation_rp_balanced_1_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.962, 1: 0.879, 2: 0.857, 3: 0.847, 4: 0.858, 5: 0.837, 6: 0.858}, ('validation_rp_balanced_1_2_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'small', 'corrective'): {0: 0.955, 1: 0.833, 2: 0.79, 3: 0.761, 4: 0.755, 5: 0.769, 6: 0.806}, ('validation_rp_balanced', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.968, 1: 0.787, 2: 0.688, 3: 0.685, 4: 0.672, 5: 0.653, 6: 0.656}, ('validation_lp_balanced', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.845, 1: 0.595, 2: 0.537, 3: 0.557, 4: 0.561, 5: 0.55, 6: 0.539}, ('validation_rp_balanced_3_premise', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.983, 1: 0.755, 2: 0.712, 3: 0.673, 4: 0.673, 5: 0.625, 6: 0.609}, ('validation_rp_balanced_2_premise', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.94, 1: 0.778, 2: 0.71, 3: 0.694, 4: 0.655, 5: 0.68, 6: 0.656}, ('validation_rp_balanced_1_premise', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.963, 1: 0.873, 2: 0.859, 3: 0.857, 4: 0.866, 5: 0.846, 6: 0.859}, ('validation_rp_balanced_1_2_premise', False, 'bidir', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small', 'corrective'): {0: 0.927, 1: 0.807, 2: 0.78, 3: 0.777, 4: 0.76, 5: 0.775, 6: 0.803}, ('validation_rp_balanced', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.957, 1: 0.808, 2: 0.743, 3: 0.707, 4: 0.701, 5: 0.687, 6: 0.668}, ('validation_lp_balanced', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.807, 1: 0.634, 2: 0.558, 3: 0.564, 4: 0.551, 5: 0.534, 6: 0.529}, ('validation_rp_balanced_3_premise', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.986, 1: 0.731, 2: 0.648, 3: 0.607, 4: 0.585, 5: 0.576, 6: 0.539}, ('validation_rp_balanced_2_premise', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.94, 1: 0.805, 2: 0.746, 3: 0.698, 4: 0.677, 5: 0.7, 6: 0.691}, ('validation_rp_balanced_1_premise', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.969, 1: 0.884, 2: 0.869, 3: 0.861, 4: 0.869, 5: 0.854, 6: 0.859}, ('validation_rp_balanced_1_2_premise', False, 'bidir', 'direct', 'layers=2', 'r2', 'rp', 'small'): {0: 0.927, 1: 0.831, 2: 0.808, 3: 0.784, 4: 0.777, 5: 0.795, 6: 0.818}, ('validation_rp_balanced', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.892, 1: 0.594, 2: 0.532, 3: 0.543, 4: 0.534, 5: 0.505, 6: 0.52}, ('validation_lp_balanced', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.821, 1: 0.535, 2: 0.508, 3: 0.537, 4: 0.514, 5: 0.501, 6: 0.497}, ('validation_rp_balanced_3_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.872, 1: 0.599, 2: 0.536, 3: 0.523, 4: 0.516, 5: 0.484, 6: 0.506}, ('validation_rp_balanced_2_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.907, 1: 0.621, 2: 0.583, 3: 0.551, 4: 0.544, 5: 0.543, 6: 0.567}, ('validation_rp_balanced_1_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.94, 1: 0.729, 2: 0.677, 3: 0.581, 4: 0.549, 5: 0.516, 6: 0.526}, ('validation_rp_balanced_1_2_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed124', 'r2', 'rp', 'small'): {0: 0.888, 1: 0.651, 2: 0.563, 3: 0.536, 4: 0.533, 5: 0.491, 6: 0.501}, ('validation_rp_balanced', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.951, 1: 0.791, 2: 0.714, 3: 0.694, 4: 0.696, 5: 0.69, 6: 0.682}, ('validation_lp_balanced', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.784, 1: 0.607, 2: 0.533, 3: 0.562, 4: 0.542, 5: 0.539, 6: 0.537}, ('validation_rp_balanced_3_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.936, 1: 0.756, 2: 0.684, 3: 0.655, 4: 0.655, 5: 0.627, 6: 0.601}, ('validation_rp_balanced_2_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.879, 1: 0.771, 2: 0.719, 3: 0.717, 4: 0.701, 5: 0.728, 6: 0.712}, ('validation_rp_balanced_1_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.971, 1: 0.892, 2: 0.87, 3: 0.864, 4: 0.867, 5: 0.856, 6: 0.858}, ('validation_rp_balanced_1_2_premise', False, 'bidir', 'direct', 'layers=2', 'seed', 'seed125', 'r2', 'rp', 'small'): {0: 0.926, 1: 0.833, 2: 0.802, 3: 0.782, 4: 0.78, 5: 0.788, 6: 0.823}, ('validation_rp_balanced', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.95, 1: 0.792, 2: 0.721, 3: 0.697, 4: 0.7, 5: 0.688, 6: 0.679}, ('validation_lp_balanced', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.772, 1: 0.617, 2: 0.559, 3: 0.542, 4: 0.541, 5: 0.521, 6: 0.523}, ('validation_rp_balanced_3_premise', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.968, 1: 0.723, 2: 0.648, 3: 0.626, 4: 0.597, 5: 0.582, 6: 0.563}, ('validation_rp_balanced_2_premise', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.893, 1: 0.78, 2: 0.728, 3: 0.72, 4: 0.692, 5: 0.733, 6: 0.707}, ('validation_rp_balanced_1_premise', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.958, 1: 0.872, 2: 0.85, 3: 0.839, 4: 0.854, 5: 0.835, 6: 0.847}, ('validation_rp_balanced_1_2_premise', False, 'small', 'seed', 'corrective', 'layers=2', 'bidir', 'r2', 'seed125', 'rp'): {0: 0.913, 1: 0.829, 2: 0.805, 3: 0.785, 4: 0.767, 5: 0.788, 6: 0.815}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'corrective', 'small'): {0: 0.978, 1: 0.767, 2: 0.72, 3: 0.708, 4: 0.661, 5: 0.658, 6: 0.634}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'seed', 'rp', 'seed124', 'corrective', 'small'): {0: 0.968, 1: 0.747, 2: 0.722, 3: 0.702, 4: 0.644, 5: 0.667, 6: 0.616}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'seed', 'rp', 'corrective', 'seed125', 'small'): {0: 0.942, 1: 0.753, 2: 0.706, 3: 0.684, 4: 0.638, 5: 0.617, 6: 0.607}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'rp', 'direct', 'small'): {0: 0.973, 1: 0.748, 2: 0.702, 3: 0.654, 4: 0.631, 5: 0.636, 6: 0.602}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'seed', 'rp', 'seed124', 'direct', 'small'): {0: 0.856, 1: 0.58, 2: 0.525, 3: 0.539, 4: 0.495, 5: 0.543, 6: 0.507}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'layers=2', 'r2', 'seed', 'rp', 'seed125', 'direct', 'small'): {0: 0.929, 1: 0.738, 2: 0.717, 3: 0.692, 4: 0.669, 5: 0.671, 6: 0.63},
                   ('validation_rp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.942, 1: 0.769, 2: 0.727, 3: 0.691, 4: 0.705, 5: 0.702, 6: 0.705}, ('validation_lp_balanced', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.762, 1: 0.607, 2: 0.556, 3: 0.547, 4: 0.554, 5: 0.54, 6: 0.518}, ('validation_rp_balanced_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.948, 1: 0.667, 2: 0.622, 3: 0.649, 4: 0.651, 5: 0.63, 6: 0.616}, ('validation_rp_balanced_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.865, 1: 0.763, 2: 0.726, 3: 0.709, 4: 0.699, 5: 0.731, 6: 0.695}, ('validation_rp_balanced_1_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.951, 1: 0.878, 2: 0.869, 3: 0.858, 4: 0.868, 5: 0.849, 6: 0.861}, ('validation_rp_balanced_1_2_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.908, 1: 0.817, 2: 0.797, 3: 0.783, 4: 0.782, 5: 0.796, 6: 0.823}, ('validation_rp_balanced_2_3_premise', False, 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2'): {0: 0.917, 1: 0.702, 2: 0.688, 3: 0.695, 4: 0.653, 5: 0.671, 6: 0.633}, ('validation_rp_balanced', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.974, 1: 0.768, 2: 0.697, 3: 0.679, 4: 0.692, 5: 0.673, 6: 0.669}, ('validation_lp_balanced', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.823, 1: 0.621, 2: 0.554, 3: 0.544, 4: 0.541, 5: 0.545, 6: 0.543}, ('validation_rp_balanced_3_premise', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.985, 1: 0.68, 2: 0.635, 3: 0.61, 4: 0.595, 5: 0.587, 6: 0.577}, ('validation_rp_balanced_2_premise', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.958, 1: 0.744, 2: 0.671, 3: 0.647, 4: 0.617, 5: 0.646, 6: 0.648}, ('validation_rp_balanced_1_premise', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.966, 1: 0.875, 2: 0.834, 3: 0.824, 4: 0.826, 5: 0.804, 6: 0.826}, ('validation_rp_balanced_1_2_premise', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.94, 1: 0.811, 2: 0.797, 3: 0.779, 4: 0.769, 5: 0.78, 6: 0.809}, ('validation_rp_balanced_2_3_premise', False, 'seed124', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.971, 1: 0.706, 2: 0.668, 3: 0.644, 4: 0.619, 5: 0.619, 6: 0.591}, ('validation_rp_balanced', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.951, 1: 0.697, 2: 0.662, 3: 0.633, 4: 0.655, 5: 0.636, 6: 0.639}, ('validation_lp_balanced', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.796, 1: 0.597, 2: 0.543, 3: 0.559, 4: 0.569, 5: 0.565, 6: 0.544}, ('validation_rp_balanced_3_premise', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.969, 1: 0.642, 2: 0.541, 3: 0.554, 4: 0.584, 5: 0.563, 6: 0.545}, ('validation_rp_balanced_2_premise', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.905, 1: 0.675, 2: 0.646, 3: 0.637, 4: 0.604, 5: 0.626, 6: 0.657}, ('validation_rp_balanced_1_premise', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.924, 1: 0.873, 2: 0.866, 3: 0.861, 4: 0.866, 5: 0.849, 6: 0.862}, ('validation_rp_balanced_1_2_premise', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.898, 1: 0.782, 2: 0.777, 3: 0.772, 4: 0.766, 5: 0.781, 6: 0.813}, ('validation_rp_balanced_2_3_premise', False, 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'dim=128', 'layers=2', 'seed'): {0: 0.953, 1: 0.649, 2: 0.627, 3: 0.61, 4: 0.594, 5: 0.589, 6: 0.569}, ('validation_rp_balanced', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.993, 1: 0.802, 2: 0.656, 3: 0.61, 4: 0.565, 5: 0.531, 6: 0.546}, ('validation_lp_balanced', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.961, 1: 0.59, 2: 0.546, 3: 0.51, 4: 0.52, 5: 0.51, 6: 0.507}, ('validation_rp_balanced_3_premise', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.983, 1: 0.709, 2: 0.631, 3: 0.571, 4: 0.514, 5: 0.499, 6: 0.52}, ('validation_rp_balanced_2_premise', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.973, 1: 0.779, 2: 0.662, 3: 0.583, 4: 0.568, 5: 0.572, 6: 0.537}, ('validation_rp_balanced_1_premise', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.947, 1: 0.752, 2: 0.608, 3: 0.549, 4: 0.536, 5: 0.519, 6: 0.504}, ('validation_rp_balanced_1_2_premise', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.97, 1: 0.761, 2: 0.65, 3: 0.593, 4: 0.566, 5: 0.548, 6: 0.522}, ('validation_rp_balanced_2_3_premise', False, 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2'): {0: 0.987, 1: 0.762, 2: 0.681, 3: 0.645, 4: 0.594, 5: 0.581, 6: 0.528}, ('validation_rp_balanced', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.988, 1: 0.795, 2: 0.733, 3: 0.689, 4: 0.683, 5: 0.666, 6: 0.658}, ('validation_lp_balanced', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.921, 1: 0.612, 2: 0.541, 3: 0.556, 4: 0.554, 5: 0.546, 6: 0.545}, ('validation_rp_balanced_3_premise', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.997, 1: 0.651, 2: 0.584, 3: 0.56, 4: 0.539, 5: 0.544, 6: 0.511}, ('validation_rp_balanced_2_premise', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.988, 1: 0.819, 2: 0.736, 3: 0.706, 4: 0.662, 5: 0.68, 6: 0.64}, ('validation_rp_balanced_1_premise', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.837, 1: 0.884, 2: 0.864, 3: 0.848, 4: 0.851, 5: 0.835, 6: 0.842}, ('validation_rp_balanced_1_2_premise', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.957, 1: 0.84, 2: 0.812, 3: 0.793, 4: 0.781, 5: 0.788, 6: 0.819}, ('validation_rp_balanced_2_3_premise', False, 'seed124', 'dim=512', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.989, 1: 0.758, 2: 0.709, 3: 0.674, 4: 0.615, 5: 0.62, 6: 0.57}, ('validation_rp_balanced', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.93, 1: 0.707, 2: 0.676, 3: 0.666, 4: 0.676, 5: 0.658, 6: 0.669}, ('validation_lp_balanced', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.762, 1: 0.61, 2: 0.526, 3: 0.54, 4: 0.537, 5: 0.537, 6: 0.519}, ('validation_rp_balanced_3_premise', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.949, 1: 0.549, 2: 0.547, 3: 0.534, 4: 0.579, 5: 0.57, 6: 0.548}, ('validation_rp_balanced_2_premise', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.88, 1: 0.623, 2: 0.68, 3: 0.694, 4: 0.652, 5: 0.705, 6: 0.708}, ('validation_rp_balanced_1_premise', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.956, 1: 0.876, 2: 0.861, 3: 0.854, 4: 0.863, 5: 0.842, 6: 0.853}, ('validation_rp_balanced_1_2_premise', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.897, 1: 0.799, 2: 0.788, 3: 0.772, 4: 0.766, 5: 0.793, 6: 0.815}, ('validation_rp_balanced_2_3_premise', False, 'dim=512', 'seed125', 'rp', 'corrective', 'r2', 'bidir', 'small', 'layers=2', 'seed'): {0: 0.92, 1: 0.564, 2: 0.611, 3: 0.597, 4: 0.59, 5: 0.608, 6: 0.575},
                   ('validation_rp_balanced', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 0.997, 3: 0.996, 4: 0.989, 5: 0.968, 6: 0.953}, ('validation_lp_balanced', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.974, 3: 0.91, 4: 0.822, 5: 0.704, 6: 0.631}, ('validation_rp_balanced_3_premise', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.999, 3: 0.992, 4: 0.958, 5: 0.904, 6: 0.824}, ('validation_rp_balanced_2_premise', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 0.991, 4: 0.979, 5: 0.964, 6: 0.937}, ('validation_rp_balanced_1_premise', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 0.986, 1: 1.0, 2: 0.997, 3: 0.998, 4: 0.99, 5: 0.977, 6: 0.947}, ('validation_rp_balanced_1_2_premise', False, 'baseline', 'r2', 'rp', 'corrective', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.992, 4: 0.983, 5: 0.98, 6: 0.956}, ('validation_rp_balanced', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.99, 5: 0.97, 6: 0.954}, ('validation_lp_balanced', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 1.0, 1: 0.996, 2: 0.977, 3: 0.93, 4: 0.824, 5: 0.695, 6: 0.615}, ('validation_rp_balanced_3_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.998, 3: 0.986, 4: 0.968, 5: 0.893, 6: 0.779}, ('validation_rp_balanced_2_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.985, 5: 0.971, 6: 0.939}, ('validation_rp_balanced_1_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 0.941, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.995, 5: 0.986, 6: 0.965}, ('validation_rp_balanced_1_2_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed124', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.99, 5: 0.99, 6: 0.966}, ('validation_rp_balanced', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.998, 3: 0.991, 4: 0.981, 5: 0.96, 6: 0.936}, ('validation_lp_balanced', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 1.0, 1: 0.994, 2: 0.965, 3: 0.911, 4: 0.786, 5: 0.668, 6: 0.603}, ('validation_rp_balanced_3_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.996, 3: 0.986, 4: 0.965, 5: 0.907, 6: 0.819}, ('validation_rp_balanced_2_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 1.0, 1: 1.0, 2: 0.997, 3: 0.994, 4: 0.969, 5: 0.967, 6: 0.923}, ('validation_rp_balanced_1_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.991, 5: 0.971, 6: 0.925}, ('validation_rp_balanced_1_2_premise', False, 'seed', 'baseline', 'r2', 'rp', 'corrective', 'seed125', 'small', 'bidir'): {0: 1.0, 1: 0.999, 2: 0.998, 3: 0.991, 4: 0.991, 5: 0.985, 6: 0.948}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'rp', 'corrective', 'baseline', 'small'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.989, 4: 0.973, 5: 0.942, 6: 0.894}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'seed124', 'corrective', 'baseline', 'small'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.997, 4: 0.98, 5: 0.956, 6: 0.905}, ('validation_rp_balanced_2_3_premise', False, 'bidir', 'r2', 'seed', 'rp', 'corrective', 'seed125', 'baseline', 'small'): {0: 1.0, 1: 1.0, 2: 0.997, 3: 0.994, 4: 0.968, 5: 0.952, 6: 0.886},
                   ('validation_rp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.992, 3: 0.968, 4: 0.93, 5: 0.918, 6: 0.893}, ('validation_lp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.961, 2: 0.864, 3: 0.717, 4: 0.624, 5: 0.598, 6: 0.556}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.992, 2: 0.985, 3: 0.931, 4: 0.899, 5: 0.838, 6: 0.794}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.997, 2: 0.992, 3: 0.965, 4: 0.941, 5: 0.905, 6: 0.855}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 0.999, 1: 0.999, 2: 0.995, 3: 0.985, 4: 0.968, 5: 0.963, 6: 0.953}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.992, 3: 0.97, 4: 0.96, 5: 0.951, 6: 0.929}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.994, 2: 0.989, 3: 0.951, 4: 0.903, 5: 0.882, 6: 0.82}, ('validation_rp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.993, 4: 0.984, 5: 0.972, 6: 0.955}, ('validation_lp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.994, 2: 0.98, 3: 0.901, 4: 0.805, 5: 0.725, 6: 0.654}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 0.985, 4: 0.97, 5: 0.917, 6: 0.87}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.983, 5: 0.969, 6: 0.941}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.999, 1: 1.0, 2: 0.996, 3: 0.998, 4: 0.992, 5: 0.985, 6: 0.973}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.999, 3: 0.994, 4: 0.98, 5: 0.98, 6: 0.967}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.996, 3: 0.994, 4: 0.968, 5: 0.956, 6: 0.907}, ('validation_rp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.995, 3: 0.992, 4: 0.969, 5: 0.954, 6: 0.916}, ('validation_lp_balanced', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.999, 1: 0.984, 2: 0.935, 3: 0.84, 4: 0.729, 5: 0.637, 6: 0.567}, ('validation_rp_balanced_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.997, 2: 0.993, 3: 0.959, 4: 0.895, 5: 0.785, 6: 0.68}, ('validation_rp_balanced_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.993, 3: 0.983, 4: 0.964, 5: 0.944, 6: 0.876}, ('validation_rp_balanced_1_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.996, 4: 0.985, 5: 0.97, 6: 0.939}, ('validation_rp_balanced_1_2_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.985, 4: 0.982, 5: 0.972, 6: 0.952}, ('validation_rp_balanced_2_3_premise', False, 'dim=128', 'small', 'bidir', 'r2', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.993, 3: 0.988, 4: 0.955, 5: 0.904, 6: 0.83}, ('validation_rp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.996, 4: 0.992, 5: 0.979, 6: 0.974}, ('validation_lp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 0.998, 2: 0.986, 3: 0.941, 4: 0.87, 5: 0.751, 6: 0.643}, ('validation_rp_balanced_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.993, 4: 0.971, 5: 0.939, 6: 0.881}, ('validation_rp_balanced_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.995, 4: 0.987, 5: 0.982, 6: 0.957}, ('validation_rp_balanced_1_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 0.994, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.994, 5: 0.996, 6: 0.987}, ('validation_rp_balanced_1_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.994, 4: 0.992, 5: 0.986, 6: 0.974}, ('validation_rp_balanced_2_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'layers=8', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.998, 4: 0.979, 5: 0.967, 6: 0.921}, ('validation_rp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.995, 4: 0.988, 5: 0.979, 6: 0.954}, ('validation_lp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.998, 2: 0.98, 3: 0.895, 4: 0.848, 5: 0.747, 6: 0.658}, ('validation_rp_balanced_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.997, 1: 0.999, 2: 0.998, 3: 0.988, 4: 0.983, 5: 0.926, 6: 0.845}, ('validation_rp_balanced_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.991, 4: 0.98, 5: 0.97, 6: 0.932}, ('validation_rp_balanced_1_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.999, 1: 0.999, 2: 0.999, 3: 1.0, 4: 0.995, 5: 0.989, 6: 0.983}, ('validation_rp_balanced_1_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.99, 4: 0.987, 5: 0.993, 6: 0.975}, ('validation_rp_balanced_2_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed124', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.993, 4: 0.978, 5: 0.955, 6: 0.9}, ('validation_rp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.988, 3: 0.969, 4: 0.954, 5: 0.93, 6: 0.903}, ('validation_lp_balanced', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.999, 1: 0.992, 2: 0.89, 3: 0.796, 4: 0.687, 5: 0.623, 6: 0.603}, ('validation_rp_balanced_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.998, 1: 0.995, 2: 0.985, 3: 0.957, 4: 0.919, 5: 0.858, 6: 0.749}, ('validation_rp_balanced_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.997, 2: 0.988, 3: 0.978, 4: 0.948, 5: 0.928, 6: 0.886}, ('validation_rp_balanced_1_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 0.993, 1: 0.996, 2: 0.994, 3: 0.971, 4: 0.961, 5: 0.94, 6: 0.905}, ('validation_rp_balanced_1_2_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.993, 3: 0.977, 4: 0.965, 5: 0.945, 6: 0.929}, ('validation_rp_balanced_2_3_premise', False, 'small', 'bidir', 'r2', 'dim=512', 'seed125', 'layers=8', 'seed', 'corrective', 'rp'): {0: 1.0, 1: 0.999, 2: 0.984, 3: 0.97, 4: 0.944, 5: 0.917, 6: 0.852}}
# print(generate_ablation_table(process_grouped_scores(raw_scores_small, step_index=6)))
# print_corrective_stats(process_grouped_scores(raw_scores_small, step_index=6))
# plot_corrective_stats(process_grouped_scores(raw_scores_small, step_index=5, min_layers=4))


###### Scaling ######
# print("raw_scores_scaling_deep_30=", get_raw_scores([x for x in repo.entries if {'rp', 'scaling'} <= x.tags], pred_count=30))
raw_scores_scaling_deep_30= {('lp', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.997, 4: 0.984, 5: 0.959, 6: 0.824, 7: 0.713, 8: 0.645, 9: 0.586, 10: 0.57, 11: 0.566, 12: 0.538}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.999, 1: 0.997, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.997, 6: 0.968, 7: 0.869, 8: 0.729, 9: 0.679, 10: 0.634, 11: 0.587, 12: 0.578}, ('rp', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.994, 7: 0.972, 8: 0.916, 9: 0.787, 10: 0.644, 11: 0.59, 12: 0.54}, ('lp', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.999, 4: 0.993, 5: 0.976, 6: 0.942, 7: 0.916, 8: 0.867, 9: 0.84, 10: 0.826, 11: 0.81, 12: 0.749}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.999, 1: 0.987, 2: 0.999, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.996, 8: 0.993, 9: 0.992, 10: 0.991, 11: 0.987, 12: 0.983}, ('rp', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.997, 7: 0.992, 8: 0.987, 9: 0.988, 10: 0.968, 11: 0.949, 12: 0.913}, ('lp', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.996, 5: 0.988, 6: 0.968, 7: 0.909, 8: 0.793, 9: 0.709, 10: 0.611, 11: 0.598, 12: 0.579}, ('lp_star', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.998, 7: 0.978, 8: 0.901, 9: 0.821, 10: 0.691, 11: 0.632, 12: 0.577}, ('rp', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.995, 8: 0.989, 9: 0.937, 10: 0.827, 11: 0.728, 12: 0.62}, ('lp', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.992, 6: 0.982, 7: 0.972, 8: 0.959, 9: 0.93, 10: 0.889, 11: 0.869, 12: 0.844}, ('lp_star', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 0.997, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.999, 8: 0.996, 9: 0.997, 10: 0.996, 11: 0.993, 12: 0.993}, ('rp', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.999, 7: 0.998, 8: 0.998, 9: 0.996, 10: 0.988, 11: 0.988, 12: 0.978}, ('lp', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 0.992, 6: 0.981, 7: 0.957, 8: 0.83, 9: 0.725, 10: 0.624, 11: 0.608, 12: 0.577}, ('lp_star', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.999, 7: 0.982, 8: 0.937, 9: 0.845, 10: 0.778, 11: 0.673, 12: 0.632}, ('rp', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.995, 9: 0.976, 10: 0.914, 11: 0.803, 12: 0.715}, ('lp', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.997, 6: 0.991, 7: 0.984, 8: 0.968, 9: 0.952, 10: 0.92, 11: 0.877, 12: 0.839}, ('lp_star', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.997, 1: 0.999, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.999, 9: 0.999, 10: 0.996, 11: 0.995, 12: 0.994}, ('rp', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.999, 7: 1.0, 8: 0.998, 9: 0.998, 10: 0.992, 11: 0.985, 12: 0.973}, ('lp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.998, 6: 0.995, 7: 0.984, 8: 0.918, 9: 0.768, 10: 0.682, 11: 0.63, 12: 0.59}, ('lp_star', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.999, 5: 1.0, 6: 0.998, 7: 0.99, 8: 0.959, 9: 0.881, 10: 0.785, 11: 0.682, 12: 0.606}, ('rp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.995, 9: 0.985, 10: 0.95, 11: 0.884, 12: 0.797}, ('lp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.998, 6: 0.995, 7: 0.995, 8: 0.982, 9: 0.964, 10: 0.943, 11: 0.9, 12: 0.852}, ('lp_star', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 0.998, 2: 1.0, 3: 0.999, 4: 1.0, 5: 0.999, 6: 1.0, 7: 0.999, 8: 0.997, 9: 0.996, 10: 0.992, 11: 0.988, 12: 0.973}, ('rp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 1.0, 1: 0.998, 2: 1.0, 3: 0.998, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.998, 8: 0.999, 9: 0.989, 10: 0.994, 11: 0.986, 12: 0.984}, ('lp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.998, 6: 0.996, 7: 0.988, 8: 0.96, 9: 0.905, 10: 0.789, 11: 0.68, 12: 0.587}, ('lp_star', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.999, 7: 0.997, 8: 0.98, 9: 0.952, 10: 0.888, 11: 0.819, 12: 0.749}, ('rp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.998, 9: 0.993, 10: 0.982, 11: 0.96, 12: 0.91}, ('lp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.999, 7: 0.996, 8: 0.998, 9: 0.994, 10: 0.982, 11: 0.978, 12: 0.951}, ('lp_star', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.999, 8: 0.999, 9: 0.996, 10: 0.999, 11: 0.988, 12: 0.988}, ('rp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.999, 8: 0.999, 9: 0.996, 10: 0.993, 11: 0.994, 12: 0.989}, ('lp', False, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 0.999, 1: 0.991, 2: 0.944, 3: 0.879, 4: 0.761, 5: 0.697, 6: 0.61, 7: 0.631, 8: 0.686, 9: 0.651, 10: 0.65, 11: 0.68, 12: 0.669}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 0.999, 1: 1.0, 2: 0.997, 3: 0.989, 4: 0.938, 5: 0.873, 6: 0.801, 7: 0.779, 8: 0.774, 9: 0.75, 10: 0.753, 11: 0.759, 12: 0.722}, ('rp', False, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 0.999, 1: 0.999, 2: 0.996, 3: 0.995, 4: 0.986, 5: 0.962, 6: 0.911, 7: 0.877, 8: 0.794, 9: 0.719, 10: 0.639, 11: 0.625, 12: 0.583}, ('lp', True, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 0.999, 1: 0.997, 2: 0.989, 3: 0.979, 4: 0.949, 5: 0.92, 6: 0.89, 7: 0.892, 8: 0.865, 9: 0.836, 10: 0.826, 11: 0.772, 12: 0.763}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 1.0, 1: 0.997, 2: 0.999, 3: 0.997, 4: 0.998, 5: 0.996, 6: 0.995, 7: 0.986, 8: 0.982, 9: 0.987, 10: 0.981, 11: 0.973, 12: 0.975}, ('rp', True, 'rp', 'bidir', 'scaling', 'r2', 'epochs30', 'corrective'): {0: 0.999, 1: 0.998, 2: 1.0, 3: 0.999, 4: 1.0, 5: 0.997, 6: 0.995, 7: 0.984, 8: 0.975, 9: 0.973, 10: 0.945, 11: 0.927, 12: 0.904}, ('lp', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 1.0, 1: 0.993, 2: 0.964, 3: 0.911, 4: 0.817, 5: 0.737, 6: 0.684, 7: 0.675, 8: 0.656, 9: 0.629, 10: 0.607, 11: 0.659, 12: 0.648}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 1.0, 1: 0.998, 2: 0.998, 3: 0.993, 4: 0.963, 5: 0.919, 6: 0.856, 7: 0.781, 8: 0.753, 9: 0.74, 10: 0.717, 11: 0.724, 12: 0.675}, ('rp', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 1.0, 1: 0.999, 2: 0.998, 3: 0.995, 4: 0.99, 5: 0.97, 6: 0.921, 7: 0.877, 8: 0.758, 9: 0.698, 10: 0.621, 11: 0.588, 12: 0.555}, ('lp', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 0.998, 1: 0.997, 2: 0.993, 3: 0.977, 4: 0.955, 5: 0.929, 6: 0.904, 7: 0.884, 8: 0.869, 9: 0.859, 10: 0.859, 11: 0.845, 12: 0.842}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 0.99, 1: 0.999, 2: 0.999, 3: 0.993, 4: 0.993, 5: 0.988, 6: 0.979, 7: 0.97, 8: 0.967, 9: 0.967, 10: 0.967, 11: 0.963, 12: 0.966}, ('rp', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'epochs60'): {0: 0.998, 1: 0.994, 2: 0.998, 3: 0.999, 4: 0.994, 5: 0.995, 6: 0.984, 7: 0.986, 8: 0.973, 9: 0.957, 10: 0.93, 11: 0.914, 12: 0.889}, ('lp', False, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.999, 7: 0.999, 8: 0.991, 9: 0.974, 10: 0.952, 11: 0.888, 12: 0.789}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.998, 8: 0.997, 9: 0.975, 10: 0.926, 11: 0.84, 12: 0.72}, ('rp', False, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 1.0, 7: 1.0, 8: 0.997, 9: 0.997, 10: 0.989, 11: 0.974, 12: 0.955}, ('lp', True, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 1.0, 4: 0.999, 5: 1.0, 6: 0.998, 7: 0.999, 8: 0.997, 9: 0.99, 10: 0.988, 11: 0.969, 12: 0.957}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 0.997, 10: 0.996, 11: 0.994, 12: 0.981}, ('rp', True, 'rp', 'bidir', 'scaling', 'r2', 'layers=128', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.998, 5: 1.0, 6: 0.999, 7: 0.999, 8: 0.999, 9: 0.996, 10: 0.992, 11: 0.994, 12: 0.984}, ('lp', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.995, 6: 0.988, 7: 0.978, 8: 0.917, 9: 0.848, 10: 0.722, 11: 0.632, 12: 0.574}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.998, 7: 0.993, 8: 0.972, 9: 0.924, 10: 0.859, 11: 0.759, 12: 0.676}, ('rp', False, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.996, 9: 0.984, 10: 0.95, 11: 0.883, 12: 0.777}, ('lp', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.995, 7: 0.994, 8: 0.982, 9: 0.984, 10: 0.96, 11: 0.942, 12: 0.9}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 0.999, 1: 0.999, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 0.997, 9: 0.997, 10: 0.997, 11: 0.994, 12: 0.975}, ('rp', True, 'rp', 'bidir', 'scaling', 'r2', 'corrective', 'layers=96'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.999, 7: 0.998, 8: 0.998, 9: 0.996, 10: 0.99, 11: 0.987, 12: 0.979}, ('lp', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 1.0, 1: 0.981, 2: 0.887, 3: 0.769, 4: 0.592, 5: 0.566, 6: 0.585, 7: 0.582, 8: 0.612, 9: 0.624, 10: 0.582, 11: 0.582, 12: 0.586}, ('lp_star', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 1.0, 1: 0.999, 2: 0.988, 3: 0.945, 4: 0.896, 5: 0.873, 6: 0.861, 7: 0.88, 8: 0.85, 9: 0.869, 10: 0.867, 11: 0.844, 12: 0.822}, ('rp', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 1.0, 1: 0.998, 2: 0.985, 3: 0.965, 4: 0.931, 5: 0.889, 6: 0.846, 7: 0.822, 8: 0.785, 9: 0.752, 10: 0.678, 11: 0.708, 12: 0.661}, ('lp', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 1.0, 1: 0.999, 2: 0.984, 3: 0.939, 4: 0.908, 5: 0.853, 6: 0.782, 7: 0.776, 8: 0.72, 9: 0.708, 10: 0.667, 11: 0.638, 12: 0.587}, ('lp_star', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.996, 1: 0.996, 2: 0.997, 3: 0.994, 4: 0.995, 5: 0.982, 6: 0.977, 7: 0.968, 8: 0.942, 9: 0.923, 10: 0.909, 11: 0.913, 12: 0.908}, ('rp', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 0.997, 4: 0.993, 5: 0.98, 6: 0.968, 7: 0.947, 8: 0.904, 9: 0.889, 10: 0.83, 11: 0.8, 12: 0.758}, ('lp', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.992, 3: 0.969, 4: 0.92, 5: 0.859, 6: 0.763, 7: 0.673, 8: 0.607, 9: 0.601, 10: 0.578, 11: 0.601, 12: 0.57}, ('lp_star', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.997, 5: 0.971, 6: 0.916, 7: 0.843, 8: 0.779, 9: 0.75, 10: 0.713, 11: 0.697, 12: 0.639}, ('rp', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 1.0, 4: 0.999, 5: 0.992, 6: 0.974, 7: 0.908, 8: 0.795, 9: 0.692, 10: 0.592, 11: 0.563, 12: 0.539}, ('lp', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.998, 4: 0.985, 5: 0.958, 6: 0.942, 7: 0.933, 8: 0.905, 9: 0.906, 10: 0.893, 11: 0.889, 12: 0.877}, ('lp_star', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.995, 7: 0.997, 8: 0.995, 9: 0.995, 10: 0.991, 11: 0.995, 12: 0.99}, ('rp', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.997, 7: 0.997, 8: 0.992, 9: 0.988, 10: 0.984, 11: 0.966, 12: 0.952}, ('lp', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.995, 3: 0.957, 4: 0.859, 5: 0.775, 6: 0.677, 7: 0.638, 8: 0.598, 9: 0.612, 10: 0.589, 11: 0.593, 12: 0.587}, ('lp_star', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 0.999, 2: 1.0, 3: 0.999, 4: 0.996, 5: 0.981, 6: 0.929, 7: 0.87, 8: 0.84, 9: 0.798, 10: 0.76, 11: 0.766, 12: 0.713}, ('rp', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.996, 5: 0.988, 6: 0.979, 7: 0.954, 8: 0.873, 9: 0.767, 10: 0.672, 11: 0.64, 12: 0.586}, ('lp', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 0.999, 2: 0.997, 3: 0.987, 4: 0.955, 5: 0.904, 6: 0.854, 7: 0.843, 8: 0.825, 9: 0.805, 10: 0.778, 11: 0.781, 12: 0.766}, ('lp_star', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 0.995, 2: 1.0, 3: 1.0, 4: 0.999, 5: 0.997, 6: 0.992, 7: 0.986, 8: 0.988, 9: 0.979, 10: 0.981, 11: 0.992, 12: 0.989}, ('rp', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.999, 5: 0.998, 6: 0.994, 7: 0.993, 8: 0.977, 9: 0.973, 10: 0.954, 11: 0.926, 12: 0.924}, ('lp', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.988, 1: 0.84, 2: 0.683, 3: 0.602, 4: 0.567, 5: 0.561, 6: 0.556, 7: 0.548, 8: 0.568, 9: 0.572, 10: 0.582, 11: 0.55, 12: 0.556}, ('lp_star', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 1.0, 1: 0.978, 2: 0.871, 3: 0.757, 4: 0.705, 5: 0.707, 6: 0.68, 7: 0.667, 8: 0.647, 9: 0.649, 10: 0.644, 11: 0.587, 12: 0.569}, ('rp', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.999, 1: 0.98, 2: 0.921, 3: 0.877, 4: 0.829, 5: 0.809, 6: 0.802, 7: 0.772, 8: 0.769, 9: 0.776, 10: 0.72, 11: 0.75, 12: 0.748}, ('lp', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.997, 1: 0.981, 2: 0.936, 3: 0.891, 4: 0.844, 5: 0.767, 6: 0.737, 7: 0.725, 8: 0.684, 9: 0.665, 10: 0.621, 11: 0.618, 12: 0.598}, ('lp_star', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.982, 1: 0.986, 2: 0.979, 3: 0.965, 4: 0.945, 5: 0.918, 6: 0.896, 7: 0.876, 8: 0.834, 9: 0.816, 10: 0.777, 11: 0.716, 12: 0.707}, ('rp', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.997, 1: 0.999, 2: 0.987, 3: 0.975, 4: 0.955, 5: 0.919, 6: 0.895, 7: 0.847, 8: 0.816, 9: 0.774, 10: 0.72, 11: 0.719, 12: 0.675}, ('lp', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 0.999, 2: 0.995, 3: 0.963, 4: 0.877, 5: 0.8, 6: 0.71, 7: 0.679, 8: 0.636, 9: 0.623, 10: 0.617, 11: 0.612, 12: 0.613}, ('lp_star', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 0.998, 2: 1.0, 3: 1.0, 4: 0.99, 5: 0.974, 6: 0.919, 7: 0.852, 8: 0.769, 9: 0.717, 10: 0.693, 11: 0.677, 12: 0.646}, ('rp', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.996, 5: 0.996, 6: 0.977, 7: 0.949, 8: 0.846, 9: 0.745, 10: 0.642, 11: 0.576, 12: 0.547}, ('lp', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 1.0, 2: 0.998, 3: 0.988, 4: 0.962, 5: 0.918, 6: 0.895, 7: 0.882, 8: 0.868, 9: 0.86, 10: 0.832, 11: 0.83, 12: 0.817}, ('lp_star', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 0.998, 2: 0.998, 3: 0.999, 4: 0.998, 5: 0.996, 6: 0.996, 7: 0.992, 8: 0.992, 9: 0.993, 10: 0.987, 11: 0.99, 12: 0.989}, ('rp', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.999, 6: 0.993, 7: 0.992, 8: 0.981, 9: 0.978, 10: 0.97, 11: 0.969, 12: 0.948}, ('lp', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 1.0, 1: 0.996, 2: 0.99, 3: 0.971, 4: 0.933, 5: 0.856, 6: 0.762, 7: 0.687, 8: 0.615, 9: 0.607, 10: 0.609, 11: 0.6, 12: 0.592}, ('lp_star', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 1.0, 1: 0.999, 2: 0.999, 3: 0.993, 4: 0.99, 5: 0.954, 6: 0.916, 7: 0.842, 8: 0.796, 9: 0.754, 10: 0.748, 11: 0.713, 12: 0.697}, ('rp', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.994, 5: 0.991, 6: 0.971, 7: 0.934, 8: 0.841, 9: 0.738, 10: 0.646, 11: 0.592, 12: 0.549}, ('lp', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 1.0, 1: 1.0, 2: 0.992, 3: 0.985, 4: 0.965, 5: 0.92, 6: 0.857, 7: 0.862, 8: 0.818, 9: 0.805, 10: 0.792, 11: 0.769, 12: 0.759}, ('lp_star', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.999, 1: 0.998, 2: 0.999, 3: 0.993, 4: 0.987, 5: 0.988, 6: 0.981, 7: 0.969, 8: 0.965, 9: 0.959, 10: 0.975, 11: 0.962, 12: 0.959}, ('rp', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.996, 5: 0.995, 6: 0.987, 7: 0.98, 8: 0.968, 9: 0.957, 10: 0.925, 11: 0.905, 12: 0.885}}
# print("raw_scores_scaling_deep_60=", get_raw_scores([x for x in repo.entries if {'rp', 'scaling'} <= x.tags], pred_count=60))
raw_scores_scaling_deep_60= {('lp', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.989, 1: 0.99, 2: 0.99, 3: 0.981, 4: 0.945, 5: 0.861, 6: 0.697, 8: 0.527, 7: 0.587, 9: 0.539, 10: 0.54, 11: 0.511, 12: 0.516}, ('lp_star', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.998, 1: 0.989, 2: 0.996, 3: 0.987, 4: 0.977, 5: 0.945, 6: 0.799, 7: 0.676, 8: 0.605, 9: 0.562, 10: 0.545, 11: 0.535, 12: 0.541}, ('rp', False, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.996, 1: 1.0, 2: 0.999, 3: 0.995, 4: 0.992, 5: 0.973, 6: 0.952, 7: 0.914, 8: 0.836, 9: 0.728, 10: 0.651, 11: 0.6, 12: 0.557}, ('lp', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.992, 1: 0.982, 2: 0.968, 3: 0.956, 4: 0.927, 5: 0.852, 6: 0.789, 8: 0.64, 7: 0.707, 9: 0.61, 10: 0.593, 11: 0.573, 12: 0.539}, ('lp_star', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.996, 1: 0.965, 2: 0.973, 3: 0.963, 4: 0.948, 5: 0.956, 6: 0.891, 7: 0.853, 8: 0.848, 9: 0.793, 10: 0.767, 11: 0.746, 12: 0.742}, ('rp', True, 'rp', 'bidir', 'scaling', 'layers=16', 'r2', 'corrective'): {0: 0.992, 1: 0.969, 2: 0.981, 3: 0.966, 4: 0.959, 5: 0.929, 6: 0.898, 7: 0.872, 8: 0.837, 9: 0.753, 10: 0.744, 11: 0.687, 12: 0.668}, ('lp', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.996, 1: 0.998, 2: 1.0, 3: 0.993, 4: 0.979, 5: 0.943, 6: 0.857, 8: 0.612, 7: 0.756, 9: 0.594, 10: 0.563, 11: 0.548, 12: 0.545}, ('lp_star', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.995, 2: 0.998, 3: 0.994, 4: 0.99, 5: 0.983, 6: 0.934, 7: 0.835, 8: 0.709, 9: 0.608, 10: 0.56, 11: 0.534, 12: 0.533}, ('rp', False, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.999, 4: 1.0, 5: 0.99, 6: 0.975, 7: 0.971, 8: 0.911, 9: 0.829, 10: 0.764, 11: 0.669, 12: 0.624}, ('lp', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.996, 1: 0.983, 2: 0.977, 3: 0.972, 4: 0.963, 5: 0.921, 6: 0.885, 8: 0.765, 7: 0.826, 9: 0.711, 10: 0.667, 11: 0.642, 12: 0.595}, ('lp_star', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 0.986, 2: 0.988, 3: 0.982, 4: 0.967, 5: 0.975, 6: 0.948, 7: 0.924, 8: 0.924, 9: 0.891, 10: 0.862, 11: 0.853, 12: 0.821}, ('rp', True, 'rp', 'bidir', 'layers=24', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 0.996, 2: 0.994, 3: 0.987, 4: 0.958, 5: 0.949, 6: 0.942, 7: 0.934, 8: 0.874, 9: 0.834, 10: 0.823, 11: 0.798, 12: 0.764}, ('lp', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 1.0, 2: 0.998, 3: 0.994, 4: 0.986, 5: 0.953, 6: 0.893, 8: 0.617, 7: 0.77, 9: 0.571, 10: 0.569, 11: 0.54, 12: 0.512}, ('lp_star', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.999, 2: 0.997, 3: 0.996, 4: 0.99, 5: 0.987, 6: 0.944, 7: 0.866, 8: 0.754, 9: 0.64, 10: 0.557, 11: 0.551, 12: 0.537}, ('rp', False, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.996, 5: 0.996, 6: 0.988, 7: 0.984, 8: 0.941, 9: 0.881, 10: 0.829, 11: 0.738, 12: 0.652}, ('lp', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.983, 2: 0.977, 3: 0.964, 4: 0.952, 5: 0.906, 6: 0.9, 8: 0.784, 7: 0.844, 9: 0.74, 10: 0.649, 11: 0.648, 12: 0.614}, ('lp_star', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.985, 2: 0.99, 3: 0.979, 4: 0.969, 5: 0.976, 6: 0.958, 7: 0.933, 8: 0.932, 9: 0.881, 10: 0.851, 11: 0.828, 12: 0.809}, ('rp', True, 'rp', 'bidir', 'layers=32', 'scaling', 'r2', 'corrective'): {0: 0.996, 1: 0.992, 2: 0.986, 3: 0.979, 4: 0.962, 5: 0.958, 6: 0.933, 7: 0.925, 8: 0.895, 9: 0.836, 10: 0.815, 11: 0.79, 12: 0.751}, ('lp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.997, 1: 0.995, 2: 0.996, 3: 0.987, 4: 0.979, 5: 0.956, 6: 0.922, 8: 0.706, 7: 0.859, 9: 0.612, 10: 0.571, 11: 0.541, 12: 0.524}, ('lp_star', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.998, 1: 0.999, 2: 0.997, 3: 0.994, 4: 0.993, 5: 0.985, 6: 0.953, 7: 0.89, 8: 0.811, 9: 0.664, 10: 0.6, 11: 0.561, 12: 0.531}, ('rp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 0.997, 4: 0.999, 5: 0.993, 6: 0.98, 7: 0.968, 8: 0.92, 9: 0.87, 10: 0.804, 11: 0.74, 12: 0.68}, ('lp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.992, 1: 0.989, 2: 0.982, 3: 0.968, 4: 0.96, 5: 0.946, 6: 0.924, 8: 0.839, 7: 0.895, 9: 0.783, 10: 0.714, 11: 0.666, 12: 0.618}, ('lp_star', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.997, 1: 0.989, 2: 0.989, 3: 0.988, 4: 0.985, 5: 0.986, 6: 0.966, 7: 0.959, 8: 0.954, 9: 0.901, 10: 0.874, 11: 0.841, 12: 0.81}, ('rp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=48', 'scaling'): {0: 0.992, 1: 0.992, 2: 0.985, 3: 0.985, 4: 0.98, 5: 0.973, 6: 0.949, 7: 0.941, 8: 0.902, 9: 0.867, 10: 0.861, 11: 0.801, 12: 0.803}, ('lp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 0.998, 1: 0.998, 2: 0.996, 3: 0.992, 4: 0.988, 5: 0.972, 6: 0.951, 8: 0.834, 7: 0.918, 9: 0.762, 10: 0.651, 11: 0.6, 12: 0.546}, ('lp_star', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.993, 5: 0.993, 6: 0.983, 7: 0.956, 8: 0.907, 9: 0.819, 10: 0.716, 11: 0.646, 12: 0.6}, ('rp', False, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 0.997, 1: 0.998, 2: 1.0, 3: 0.998, 4: 1.0, 5: 0.996, 6: 0.986, 7: 0.982, 8: 0.953, 9: 0.905, 10: 0.863, 11: 0.786, 12: 0.739}, ('lp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 0.996, 1: 0.989, 2: 0.99, 3: 0.976, 4: 0.979, 5: 0.968, 6: 0.947, 8: 0.87, 7: 0.908, 9: 0.838, 10: 0.808, 11: 0.762, 12: 0.71}, ('lp_star', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 1.0, 1: 0.998, 2: 0.996, 3: 0.998, 4: 0.989, 5: 0.991, 6: 0.985, 7: 0.975, 8: 0.97, 9: 0.941, 10: 0.932, 11: 0.919, 12: 0.873}, ('rp', True, 'rp', 'corrective', 'r2', 'bidir', 'layers=64', 'scaling'): {0: 0.996, 1: 0.986, 2: 0.99, 3: 0.994, 4: 0.984, 5: 0.984, 6: 0.954, 7: 0.952, 8: 0.916, 9: 0.878, 10: 0.845, 11: 0.794, 12: 0.765}, ('lp', False, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.979, 2: 0.903, 3: 0.784, 4: 0.668, 5: 0.586, 6: 0.562, 8: 0.529, 7: 0.528, 9: 0.516, 10: 0.555, 11: 0.535, 12: 0.536}, ('lp_star', False, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.995, 1: 0.987, 2: 0.975, 3: 0.91, 4: 0.815, 5: 0.72, 6: 0.667, 7: 0.645, 8: 0.638, 9: 0.615, 10: 0.609, 11: 0.61, 12: 0.618}, ('rp', False, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.996, 3: 0.982, 4: 0.97, 5: 0.91, 6: 0.867, 7: 0.809, 8: 0.746, 9: 0.672, 10: 0.633, 11: 0.592, 12: 0.585}, ('lp', True, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.997, 1: 0.964, 2: 0.937, 3: 0.878, 4: 0.8, 5: 0.712, 6: 0.67, 8: 0.58, 7: 0.644, 9: 0.583, 10: 0.582, 11: 0.584, 12: 0.525}, ('lp_star', True, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 0.974, 2: 0.976, 3: 0.943, 4: 0.905, 5: 0.878, 6: 0.854, 7: 0.823, 8: 0.817, 9: 0.786, 10: 0.755, 11: 0.733, 12: 0.734}, ('rp', True, 'bidir', 'epochs30', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.985, 1: 0.97, 2: 0.976, 3: 0.954, 4: 0.936, 5: 0.904, 6: 0.842, 7: 0.817, 8: 0.775, 9: 0.684, 10: 0.696, 11: 0.667, 12: 0.648}, ('lp', False, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.976, 1: 0.944, 2: 0.877, 3: 0.795, 4: 0.691, 5: 0.576, 6: 0.553, 8: 0.522, 7: 0.528, 9: 0.558, 10: 0.56, 11: 0.545, 12: 0.576}, ('lp_star', False, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.996, 1: 0.964, 2: 0.948, 3: 0.881, 4: 0.813, 5: 0.749, 6: 0.693, 7: 0.624, 8: 0.605, 9: 0.602, 10: 0.597, 11: 0.593, 12: 0.575}, ('rp', False, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.999, 1: 0.996, 2: 0.986, 3: 0.951, 4: 0.924, 5: 0.87, 6: 0.792, 7: 0.746, 8: 0.681, 9: 0.636, 10: 0.607, 11: 0.551, 12: 0.55}, ('lp', True, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.981, 1: 0.955, 2: 0.922, 3: 0.873, 4: 0.798, 5: 0.733, 6: 0.711, 8: 0.624, 7: 0.673, 9: 0.625, 10: 0.601, 11: 0.625, 12: 0.601}, ('lp_star', True, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.962, 1: 0.953, 2: 0.94, 3: 0.909, 4: 0.865, 5: 0.846, 6: 0.805, 7: 0.767, 8: 0.756, 9: 0.738, 10: 0.721, 11: 0.701, 12: 0.7}, ('rp', True, 'bidir', 'rp', 'scaling', 'r2', 'epochs60', 'corrective'): {0: 0.991, 1: 0.961, 2: 0.928, 3: 0.891, 4: 0.844, 5: 0.84, 6: 0.782, 7: 0.744, 8: 0.708, 9: 0.639, 10: 0.632, 11: 0.607, 12: 0.596}, ('lp', False, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.998, 4: 0.989, 5: 0.978, 6: 0.958, 8: 0.863, 7: 0.921, 9: 0.827, 10: 0.758, 11: 0.695, 12: 0.615}, ('lp_star', False, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 0.998, 2: 0.996, 3: 0.997, 4: 0.99, 5: 0.99, 6: 0.976, 7: 0.944, 8: 0.918, 9: 0.854, 10: 0.763, 11: 0.697, 12: 0.612}, ('rp', False, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.999, 5: 0.99, 6: 0.98, 7: 0.975, 8: 0.957, 9: 0.92, 10: 0.884, 11: 0.814, 12: 0.784}, ('lp', True, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.994, 1: 0.989, 2: 0.984, 3: 0.972, 4: 0.962, 5: 0.956, 6: 0.934, 8: 0.881, 7: 0.909, 9: 0.856, 10: 0.818, 11: 0.8, 12: 0.715}, ('lp_star', True, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.993, 1: 0.992, 2: 0.997, 3: 0.987, 4: 0.985, 5: 0.977, 6: 0.973, 7: 0.965, 8: 0.96, 9: 0.918, 10: 0.923, 11: 0.895, 12: 0.848}, ('rp', True, 'layers=128', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.999, 1: 0.982, 2: 0.982, 3: 0.983, 4: 0.969, 5: 0.942, 6: 0.933, 7: 0.91, 8: 0.864, 9: 0.809, 10: 0.794, 11: 0.75, 12: 0.731}, ('lp', False, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 0.999, 3: 0.994, 4: 0.991, 5: 0.972, 6: 0.93, 8: 0.741, 7: 0.873, 9: 0.67, 10: 0.608, 11: 0.544, 12: 0.538}, ('lp_star', False, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 1.0, 2: 1.0, 3: 0.998, 4: 0.995, 5: 0.991, 6: 0.974, 7: 0.923, 8: 0.839, 9: 0.724, 10: 0.624, 11: 0.575, 12: 0.544}, ('rp', False, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.999, 4: 0.998, 5: 0.992, 6: 0.981, 7: 0.971, 8: 0.943, 9: 0.876, 10: 0.813, 11: 0.73, 12: 0.675}, ('lp', True, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.998, 1: 0.992, 2: 0.985, 3: 0.974, 4: 0.963, 5: 0.94, 6: 0.92, 8: 0.827, 7: 0.884, 9: 0.798, 10: 0.747, 11: 0.71, 12: 0.674}, ('lp_star', True, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.993, 1: 0.992, 2: 0.995, 3: 0.986, 4: 0.986, 5: 0.989, 6: 0.972, 7: 0.959, 8: 0.964, 9: 0.91, 10: 0.892, 11: 0.867, 12: 0.834}, ('rp', True, 'layers=96', 'bidir', 'rp', 'scaling', 'r2', 'corrective'): {0: 0.997, 1: 0.981, 2: 0.986, 3: 0.982, 4: 0.971, 5: 0.968, 6: 0.947, 7: 0.929, 8: 0.914, 9: 0.843, 10: 0.814, 11: 0.767, 12: 0.737}, ('lp', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.995, 1: 0.937, 2: 0.839, 3: 0.717, 4: 0.589, 5: 0.586, 6: 0.586, 8: 0.581, 7: 0.555, 9: 0.566, 10: 0.576, 11: 0.564, 12: 0.556}, ('lp_star', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.998, 1: 0.973, 2: 0.955, 3: 0.867, 4: 0.793, 5: 0.763, 6: 0.736, 7: 0.728, 8: 0.743, 9: 0.716, 10: 0.729, 11: 0.726, 12: 0.743}, ('rp', False, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.995, 1: 0.996, 2: 0.975, 3: 0.934, 4: 0.912, 5: 0.852, 6: 0.806, 7: 0.777, 8: 0.731, 9: 0.7, 10: 0.675, 11: 0.671, 12: 0.645}, ('lp', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.991, 1: 0.96, 2: 0.925, 3: 0.834, 4: 0.736, 5: 0.648, 6: 0.621, 8: 0.561, 7: 0.581, 9: 0.53, 10: 0.52, 11: 0.522, 12: 0.502}, ('lp_star', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.989, 1: 0.959, 2: 0.949, 3: 0.903, 4: 0.854, 5: 0.809, 6: 0.756, 7: 0.733, 8: 0.703, 9: 0.666, 10: 0.655, 11: 0.618, 12: 0.604}, ('rp', True, 'r2', 'scaling', 'dim=128', 'layers=8', 'rp', 'heads=2', 'bidir', 'corrective'): {0: 0.981, 1: 0.967, 2: 0.932, 3: 0.898, 4: 0.858, 5: 0.822, 6: 0.735, 7: 0.727, 8: 0.674, 9: 0.61, 10: 0.589, 11: 0.575, 12: 0.543}, ('lp', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.996, 1: 0.992, 2: 0.979, 3: 0.926, 4: 0.845, 5: 0.739, 6: 0.629, 8: 0.518, 7: 0.532, 9: 0.513, 10: 0.515, 11: 0.538, 12: 0.53}, ('lp_star', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 0.996, 2: 0.996, 3: 0.988, 4: 0.958, 5: 0.878, 6: 0.773, 7: 0.699, 8: 0.664, 9: 0.653, 10: 0.634, 11: 0.652, 12: 0.622}, ('rp', False, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 0.999, 2: 1.0, 3: 0.999, 4: 0.993, 5: 0.978, 6: 0.942, 7: 0.883, 8: 0.796, 9: 0.688, 10: 0.645, 11: 0.596, 12: 0.572}, ('lp', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 0.984, 2: 0.972, 3: 0.951, 4: 0.869, 5: 0.797, 6: 0.762, 8: 0.649, 7: 0.688, 9: 0.647, 10: 0.628, 11: 0.627, 12: 0.619}, ('lp_star', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.995, 1: 0.988, 2: 0.989, 3: 0.972, 4: 0.966, 5: 0.926, 6: 0.896, 7: 0.89, 8: 0.881, 9: 0.839, 10: 0.817, 11: 0.803, 12: 0.794}, ('rp', True, 'heads=6', 'r2', 'scaling', 'dim=384', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.997, 1: 0.993, 2: 0.99, 3: 0.976, 4: 0.964, 5: 0.935, 6: 0.896, 7: 0.867, 8: 0.822, 9: 0.772, 10: 0.741, 11: 0.694, 12: 0.656}, ('lp', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 1.0, 1: 0.999, 2: 0.984, 3: 0.909, 4: 0.826, 5: 0.671, 6: 0.603, 8: 0.517, 7: 0.532, 9: 0.529, 10: 0.555, 11: 0.516, 12: 0.527}, ('lp_star', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 0.995, 2: 0.994, 3: 0.971, 4: 0.918, 5: 0.852, 6: 0.762, 7: 0.7, 8: 0.686, 9: 0.653, 10: 0.645, 11: 0.648, 12: 0.638}, ('rp', False, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 1.0, 2: 1.0, 3: 0.997, 4: 0.99, 5: 0.967, 6: 0.937, 7: 0.915, 8: 0.835, 9: 0.752, 10: 0.695, 11: 0.664, 12: 0.62}, ('lp', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.999, 1: 0.993, 2: 0.976, 3: 0.934, 4: 0.875, 5: 0.742, 6: 0.686, 8: 0.576, 7: 0.649, 9: 0.537, 10: 0.549, 11: 0.533, 12: 0.523}, ('lp_star', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.997, 1: 0.994, 2: 0.992, 3: 0.98, 4: 0.958, 5: 0.914, 6: 0.868, 7: 0.857, 8: 0.844, 9: 0.81, 10: 0.788, 11: 0.799, 12: 0.77}, ('rp', True, 'r2', 'scaling', 'dim=512', 'heads=8', 'layers=8', 'rp', 'bidir', 'corrective'): {0: 0.994, 1: 0.999, 2: 0.996, 3: 0.983, 4: 0.985, 5: 0.952, 6: 0.924, 7: 0.907, 8: 0.884, 9: 0.834, 10: 0.815, 11: 0.769, 12: 0.751}, ('lp', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.966, 1: 0.779, 2: 0.657, 3: 0.573, 4: 0.557, 5: 0.513, 6: 0.524, 8: 0.535, 7: 0.522, 9: 0.52, 10: 0.533, 11: 0.518, 12: 0.525}, ('lp_star', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.995, 1: 0.935, 2: 0.808, 3: 0.701, 4: 0.677, 5: 0.608, 6: 0.639, 7: 0.601, 8: 0.6, 9: 0.559, 10: 0.546, 11: 0.559, 12: 0.56}, ('rp', False, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.991, 1: 0.913, 2: 0.864, 3: 0.802, 4: 0.779, 5: 0.718, 6: 0.683, 7: 0.703, 8: 0.644, 9: 0.601, 10: 0.582, 11: 0.619, 12: 0.6}, ('lp', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.991, 1: 0.932, 2: 0.851, 3: 0.782, 4: 0.681, 5: 0.639, 6: 0.595, 8: 0.535, 7: 0.579, 9: 0.526, 10: 0.503, 11: 0.513, 12: 0.499}, ('lp_star', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.966, 1: 0.934, 2: 0.904, 3: 0.851, 4: 0.777, 5: 0.737, 6: 0.717, 7: 0.705, 8: 0.678, 9: 0.645, 10: 0.635, 11: 0.589, 12: 0.594}, ('rp', True, 'r2', 'scaling', 'heads=1', 'layers=8', 'rp', 'bidir', 'dim=64', 'corrective'): {0: 0.985, 1: 0.981, 2: 0.946, 3: 0.903, 4: 0.822, 5: 0.765, 6: 0.689, 7: 0.665, 8: 0.626, 9: 0.563, 10: 0.568, 11: 0.523, 12: 0.511}, ('lp', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.996, 1: 0.995, 2: 0.983, 3: 0.921, 4: 0.823, 5: 0.666, 6: 0.603, 8: 0.533, 7: 0.548, 9: 0.534, 10: 0.539, 11: 0.558, 12: 0.554}, ('lp_star', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.998, 1: 0.99, 2: 0.993, 3: 0.972, 4: 0.923, 5: 0.836, 6: 0.727, 7: 0.682, 8: 0.639, 9: 0.626, 10: 0.605, 11: 0.61, 12: 0.598}, ('rp', False, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.997, 1: 1.0, 2: 1.0, 3: 0.997, 4: 0.991, 5: 0.964, 6: 0.934, 7: 0.872, 8: 0.774, 9: 0.687, 10: 0.63, 11: 0.577, 12: 0.553}, ('lp', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.995, 1: 0.996, 2: 0.979, 3: 0.941, 4: 0.879, 5: 0.797, 6: 0.75, 8: 0.664, 7: 0.68, 9: 0.651, 10: 0.624, 11: 0.631, 12: 0.622}, ('lp_star', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.999, 1: 0.991, 2: 0.992, 3: 0.973, 4: 0.965, 5: 0.931, 6: 0.901, 7: 0.882, 8: 0.873, 9: 0.869, 10: 0.827, 11: 0.824, 12: 0.839}, ('rp', True, 'dim=1024', 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'heads=16', 'scaling'): {0: 0.998, 1: 0.986, 2: 0.997, 3: 0.991, 4: 0.983, 5: 0.972, 6: 0.951, 7: 0.934, 8: 0.901, 9: 0.862, 10: 0.851, 11: 0.794, 12: 0.765}, ('lp', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.997, 1: 0.991, 2: 0.98, 3: 0.924, 4: 0.881, 5: 0.752, 6: 0.637, 8: 0.52, 7: 0.587, 9: 0.56, 10: 0.553, 11: 0.544, 12: 0.535}, ('lp_star', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.994, 1: 0.987, 2: 0.992, 3: 0.972, 4: 0.93, 5: 0.86, 6: 0.759, 7: 0.705, 8: 0.674, 9: 0.639, 10: 0.644, 11: 0.645, 12: 0.634}, ('rp', False, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.994, 1: 0.999, 2: 0.999, 3: 0.997, 4: 0.985, 5: 0.956, 6: 0.919, 7: 0.882, 8: 0.814, 9: 0.717, 10: 0.661, 11: 0.619, 12: 0.574}, ('lp', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.996, 1: 0.987, 2: 0.971, 3: 0.938, 4: 0.886, 5: 0.793, 6: 0.738, 8: 0.608, 7: 0.665, 9: 0.6, 10: 0.583, 11: 0.61, 12: 0.573}, ('lp_star', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.997, 1: 0.985, 2: 0.98, 3: 0.956, 4: 0.937, 5: 0.902, 6: 0.847, 7: 0.83, 8: 0.799, 9: 0.778, 10: 0.755, 11: 0.746, 12: 0.736}, ('rp', True, 'bidir', 'r2', 'layers=8', 'corrective', 'rp', 'scaling', 'heads=11', 'dim=704'): {0: 0.994, 1: 0.982, 2: 0.99, 3: 0.977, 4: 0.976, 5: 0.941, 6: 0.912, 7: 0.873, 8: 0.827, 9: 0.8, 10: 0.728, 11: 0.712, 12: 0.688}}
# print(generate_main_tables(raw_scores_baseline_deep_30 | raw_scores_scaling_deep_30, raw_scores_baseline_deep_60 | raw_scores_scaling_deep_60))
scaling_gaps_30 = evaluate_gap_closure(filter_scaling_scores(raw_scores_baseline_deep_30 | raw_scores_scaling_deep_30))
scaling_gaps_60 = evaluate_gap_closure(filter_scaling_scores(raw_scores_baseline_deep_60 | raw_scores_scaling_deep_60))
# print(generate_gap_summary_table(scaling_gaps_30, scaling_gaps_60))

direct_30_scaling = {x:y for x, y in (raw_scores_baseline_deep_30 | raw_scores_scaling_deep_30).items() if x[1] == False}
cot_30_scaling = {x:y for x, y in (raw_scores_baseline_deep_30 | raw_scores_scaling_deep_30).items() if x[1] == True}
# plot_scaling_curves(direct_30_scaling, cot_30_scaling, eval="rp", compare="mode", cls="p30", curve="layers")
# plot_scaling_curves(direct_30_scaling, cot_30_scaling, eval="lp", compare="mode", cls="p30", curve="layers")
# plot_scaling_curves(direct_30_scaling, cot_30_scaling, eval="rp", compare="mode", cls="p30", curve="heads")
# plot_scaling_curves(direct_30_scaling, cot_30_scaling, eval="lp", compare="mode", cls="p30", curve="heads")

###### Tractability ######

# print("raw_scores_tractability_deep_30=", get_raw_scores([x for x in repo.entries if {'rp', 'tractability'} <= x.tags], pred_count=30))
raw_scores_tractability_deep_30= {('lp', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 1.0, 1: 0.734, 2: 0.626, 3: 0.588, 4: 0.556, 5: 0.533, 6: 0.556, 7: 0.546, 8: 0.552, 9: 0.543, 10: 0.539, 11: 0.527, 12: 0.524}, ('lp_star', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 1.0, 1: 0.927, 2: 0.746, 3: 0.66, 4: 0.63, 5: 0.604, 6: 0.578, 7: 0.583, 8: 0.565, 9: 0.554, 10: 0.547, 11: 0.518, 12: 0.54}, ('rp', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 1.0, 1: 0.904, 2: 0.868, 3: 0.816, 4: 0.785, 5: 0.751, 6: 0.705, 7: 0.73, 8: 0.74, 9: 0.754, 10: 0.692, 11: 0.715, 12: 0.74}, ('lp', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.569, 1: 0.561, 2: 0.547, 3: 0.542, 4: 0.508, 5: 0.523, 6: 0.524, 7: 0.498, 8: 0.514, 9: 0.505, 10: 0.504, 11: 0.507, 12: 0.497}, ('lp_star', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.376, 1: 0.578, 2: 0.51, 3: 0.548, 4: 0.522, 5: 0.515, 6: 0.505, 7: 0.483, 8: 0.497, 9: 0.512, 10: 0.511, 11: 0.499, 12: 0.508}, ('rp', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.631, 1: 0.561, 2: 0.5, 3: 0.49, 4: 0.502, 5: 0.493, 6: 0.482, 7: 0.501, 8: 0.511, 9: 0.493, 10: 0.506, 11: 0.478, 12: 0.49}, ('lp', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 1.0, 1: 0.656, 2: 0.56, 3: 0.536, 4: 0.513, 5: 0.504, 6: 0.54, 7: 0.521, 8: 0.529, 9: 0.515, 10: 0.516, 11: 0.508, 12: 0.498}, ('lp_star', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 0.991, 1: 0.753, 2: 0.657, 3: 0.605, 4: 0.573, 5: 0.561, 6: 0.55, 7: 0.544, 8: 0.517, 9: 0.554, 10: 0.551, 11: 0.504, 12: 0.51}, ('rp', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 1.0, 1: 0.882, 2: 0.844, 3: 0.819, 4: 0.779, 5: 0.738, 6: 0.728, 7: 0.738, 8: 0.74, 9: 0.75, 10: 0.728, 11: 0.758, 12: 0.749}}
# print("raw_scores_tractability_deep_60=", get_raw_scores([x for x in repo.entries if {'rp', 'tractability'} <= x.tags], pred_count=60))
raw_scores_tractability_deep_60= {('lp', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 0.997, 1: 0.707, 2: 0.597, 3: 0.562, 4: 0.54, 5: 0.514, 6: 0.524, 8: 0.516, 7: 0.543, 9: 0.552, 10: 0.541, 11: 0.552, 12: 0.552}, ('lp_star', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 0.994, 1: 0.877, 2: 0.719, 3: 0.638, 4: 0.576, 5: 0.56, 6: 0.542, 7: 0.519, 8: 0.538, 9: 0.513, 10: 0.51, 11: 0.52, 12: 0.533}, ('rp', False, 'rp', 'direct', 'r2', 'bidir', 'tractability'): {0: 1.0, 1: 0.915, 2: 0.861, 3: 0.803, 4: 0.812, 5: 0.717, 6: 0.719, 7: 0.733, 8: 0.658, 9: 0.635, 10: 0.635, 11: 0.609, 12: 0.66}, ('lp', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.558, 1: 0.522, 2: 0.531, 3: 0.493, 4: 0.523, 5: 0.551, 6: 0.538, 8: 0.532, 7: 0.535, 9: 0.559, 10: 0.561, 11: 0.568, 12: 0.526}, ('lp_star', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.371, 1: 0.527, 2: 0.521, 3: 0.532, 4: 0.518, 5: 0.503, 6: 0.554, 7: 0.504, 8: 0.505, 9: 0.484, 10: 0.487, 11: 0.527, 12: 0.503}, ('rp', False, 'rp', 'direct', 'layers=16', 'r2', 'bidir', 'tractability'): {0: 0.712, 1: 0.547, 2: 0.473, 3: 0.455, 4: 0.47, 5: 0.439, 6: 0.443, 7: 0.469, 8: 0.45, 9: 0.453, 10: 0.487, 11: 0.455, 12: 0.475}, ('lp', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 0.999, 1: 0.613, 2: 0.562, 3: 0.517, 4: 0.487, 5: 0.473, 6: 0.496, 8: 0.488, 7: 0.497, 9: 0.495, 10: 0.493, 11: 0.509, 12: 0.514}, ('lp_star', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 0.994, 1: 0.72, 2: 0.648, 3: 0.563, 4: 0.523, 5: 0.497, 6: 0.506, 7: 0.488, 8: 0.507, 9: 0.485, 10: 0.485, 11: 0.497, 12: 0.537}, ('rp', False, 'rp', 'direct', 'r2', 'layers=32', 'bidir', 'tractability'): {0: 1.0, 1: 0.887, 2: 0.833, 3: 0.802, 4: 0.786, 5: 0.743, 6: 0.742, 7: 0.724, 8: 0.668, 9: 0.689, 10: 0.652, 11: 0.647, 12: 0.688}}
print(generate_main_tables(raw_scores_tractability_deep_30, raw_scores_tractability_deep_60, methods=['direct']))

scaling_tractability_30 = raw_scores_baseline_deep_30 | filter_scaling_scores(raw_scores_scaling_deep_30, take_layers=[16, 32])
direct_30_tractability = {x:y for x, y in scaling_tractability_30.items() if x[1] == False}
# plot_scaling_curves(raw_scores_tractability_deep_30, direct_30_tractability, eval="rp", compare="corrective", cls="tractability", curve="layers")
# plot_scaling_curves(raw_scores_tractability_deep_30, direct_30_tractability, eval="lp", compare="corrective", cls="tractability", curve="layers")
