import math
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from dataset.processor import fact_e, ruleend_e, query_e, rulestart_e


def _get_detailed_token_categories(type_ids_list):
    """
    Parses token types to find indices for categories and the relationships
    between rule premises and their conclusions.
    """
    query_indices = [i for i, t in enumerate(type_ids_list) if query_e in t]
    fact_indices = [i for i, t in enumerate(type_ids_list) if fact_e in t]
    all_premise_indices, all_conclusion_indices = [], []
    conclusion_to_premises, premise_to_conclusion = {}, {}

    current_premise_indices = []
    in_rule = False
    for i, types in enumerate(type_ids_list):
        if rulestart_e in types and not in_rule:
            in_rule = True
        if in_rule:
            if rulestart_e in types:
                current_premise_indices.append(i)
                all_premise_indices.append(i)
            elif ruleend_e in types:
                conclusion_idx = i
                all_conclusion_indices.append(i)
                conclusion_to_premises[conclusion_idx] = current_premise_indices
                for p_idx in current_premise_indices:
                    premise_to_conclusion[p_idx] = conclusion_idx
                current_premise_indices = []
                in_rule = False
    return {
        "query": query_indices, "facts": fact_indices,
        "rule_premises": all_premise_indices,
        "rule_conclusions": all_conclusion_indices,
        "conclusion_to_premises": conclusion_to_premises,
        "premise_to_conclusion": premise_to_conclusion
    }


def _calculate_head_summary(head_map, token_map):
    """
    Calculates the attention flow summary for a single attention head.
    This helper is used by both plotting and aggregation functions.
    """
    summary = {}
    source_cats = ['query', 'facts', 'rule_premises', 'rule_conclusions']
    pretty_source_cats = ['Query', 'Facts', 'Rule premises', 'Rule conclusions']

    for i, src_name in enumerate(source_cats):
        src_indices = token_map.get(src_name, [])
        if not src_indices: continue

        total_src_attention = head_map[src_indices, :].sum()
        if total_src_attention < 1e-9: continue

        row = {}
        attn_to_premises_own_rule, attn_to_conclusion_own_rule = 0, 0

        # Calculate attention to components within the same rule
        if src_name == 'rule_premises':
            for p_idx in src_indices:
                own_c_idx = token_map['premise_to_conclusion'].get(p_idx)
                if own_c_idx is not None:
                    attn_to_conclusion_own_rule += head_map[p_idx, own_c_idx].sum()
                    sibling_premises = [sp for sp in token_map['conclusion_to_premises'].get(own_c_idx, []) if sp != p_idx]
                    if sibling_premises:
                        attn_to_premises_own_rule += head_map[p_idx, sibling_premises].sum()

        elif src_name == 'rule_conclusions':
            for c_idx in src_indices:
                own_premises = token_map['conclusion_to_premises'].get(c_idx, [])
                if own_premises:
                    attn_to_premises_own_rule += head_map[c_idx, own_premises].sum()
                attn_to_conclusion_own_rule += head_map[c_idx, c_idx].sum()

        # Calculate total and "other" attention by subtraction
        total_attn_to_all_premises = head_map[src_indices, :][:, token_map['rule_premises']].sum() if token_map['rule_premises'] else 0
        total_attn_to_all_conclusions = head_map[src_indices, :][:, token_map['rule_conclusions']].sum() if token_map['rule_conclusions'] else 0
        attn_to_premises_other_rules = total_attn_to_all_premises - attn_to_premises_own_rule
        attn_to_conclusion_other_rules = total_attn_to_all_conclusions - attn_to_conclusion_own_rule

        row['Query'] = (head_map[src_indices, :][:, token_map['query']].sum() / total_src_attention).item() if token_map['query'] else 0
        row['Facts'] = (head_map[src_indices, :][:, token_map['facts']].sum() / total_src_attention).item() if token_map['facts'] else 0
        row['Premises\n(own rule)'] = (attn_to_premises_own_rule / total_src_attention).item()
        row['Premises\n(other rules)'] = (attn_to_premises_other_rules / total_src_attention).item()
        row['Conclusion\n(own rule)'] = (attn_to_conclusion_own_rule / total_src_attention).item()
        row['Conclusion\n(other rules)'] = (attn_to_conclusion_other_rules / total_src_attention).item()
        summary[pretty_source_cats[i]] = row
    return pd.DataFrame(summary).T.fillna(0)


def inspect_eval_attention_maps(llm_model, ds, attention_maps, aggregated=True):
    """
    Inspects attention maps across an entire dataset.
    If aggregated=True, returns global mean and std DataFrames.
    If aggregated=False, returns lists of mean and std DataFrames (one per layer).
    """
    all_head_summaries = []
    layer_summaries = {}
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Inspecting attention maps"):
            input_ids = torch.tensor(ds[i]["input_ids"], device="cuda").unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device="cuda").unsqueeze(0)
            type_ids_list = type_ids.squeeze(0).tolist()
            attention_maps.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            token_map = _get_detailed_token_categories(type_ids_list)
            for layer_idx, attn_map_layer in enumerate(attention_maps):
                if not aggregated and layer_idx not in layer_summaries:
                    layer_summaries[layer_idx] = []

                attn_map = attn_map_layer.squeeze(0).float()  # (n_heads, seq_len, seq_len)
                for head_idx in range(attn_map.shape[0]):
                    head_map = attn_map[head_idx]
                    summary_df = _calculate_head_summary(head_map, token_map)
                    if aggregated:
                        all_head_summaries.append(summary_df)
                    else:
                        layer_summaries[layer_idx].append(summary_df)

    if aggregated:
        combined_df = pd.concat(all_head_summaries)
        mean_attention_df = combined_df.groupby(combined_df.index).mean()
        std_attention_df = combined_df.groupby(combined_df.index).std()
        return mean_attention_df, std_attention_df
    else:
        mean_dfs, std_dfs = [], []
        for layer_idx in range(len(layer_summaries)):
            combined_layer_df = pd.concat(layer_summaries[layer_idx])
            mean_dfs.append(combined_layer_df.groupby(combined_layer_df.index).mean())
            std_dfs.append(combined_layer_df.groupby(combined_layer_df.index).std())
        return mean_dfs, std_dfs


def visualize_aggregate_attention_maps(mean_df, std_df, aggregated=True, filename=None):
    col_order = ['Query', 'Facts', 'Premises\n(own rule)', 'Premises\n(other rules)', 'Conclusion\n(own rule)', 'Conclusion\n(other rules)']
    row_order = ['Query', 'Facts', 'Rule premises', 'Rule conclusions']

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    if aggregated:
        plot_df_mean = mean_df.reindex(index=row_order, columns=col_order)
        plot_df_std = std_df.reindex(index=row_order, columns=col_order)
        formatted_table = plot_df_mean.applymap('{:.2f}'.format) + "\n" + plot_df_std.applymap(lambda x: f"±{x:.2f}")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(plot_df_mean, ax=ax, annot=formatted_table, fmt="s", cmap="viridis", linewidths=.5, vmin=0, vmax=1.0, cbar_kws={"shrink": 0.8, "label": "Attention probability"})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        if filename:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    else:
        num_layers = len(mean_df)
        cols = 4
        rows = math.ceil(num_layers / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8.2, rows * 1.8), sharex=True, sharey=True)
        axes = axes.flatten()
        for i in range(num_layers):
            ax = axes[i]
            plot_df_mean = mean_df[i].reindex(index=row_order, columns=col_order)
            formatted_table = plot_df_mean.applymap('{:.2f}'.format)
            sns.heatmap(plot_df_mean, ax=ax, annot=formatted_table, fmt="s", cmap="viridis",
                        linewidths=.5, vmin=0, vmax=1.0, cbar=False, annot_kws={"size": 6})
            ax.set_title(f"Layer {i+1}", fontsize=9, pad=3)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout(w_pad=0.5, h_pad=0.8)

        if filename:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
