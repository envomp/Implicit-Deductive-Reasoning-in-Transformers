import math

import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from collections import defaultdict
from dataset.processor import TYPE_NAME_MAP


def inspect_linearity(llm_model, ds, manifold_changes, initial_embeddings, hidden_states):
    target_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    other_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    device = next(llm_model.parameters()).device
    anchors = F.normalize(llm_model.type_embeddings.detach(), p=2, dim=1)  # (anchors, dim)
    total_samples = len(ds)

    with torch.no_grad():
        for i in range(total_samples):
            input_ids = torch.tensor(ds[i]["input_ids"], device=device).unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device=device).unsqueeze(0)  # (1, seq, anchors)
            seq_len = input_ids.shape[1]
            all_indices_set = set(range(seq_len))
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            full_evolution = [initial_embeddings[0]] + hidden_states
            all_layers_tensor = torch.stack(full_evolution).squeeze(1).to(device)  # (layers, seq, dim)
            if manifold_changes is not None:
                aligned_layers = []
                for layer_idx in range(all_layers_tensor.shape[0]):
                    state = all_layers_tensor[layer_idx]
                    if layer_idx == 0:
                        aligned_layers.append(state)
                    else:
                        R = manifold_changes[layer_idx]["R"].to(device)
                        aligned_state = torch.matmul(state, R)
                        aligned_layers.append(aligned_state)
                all_layers_norm = F.normalize(torch.stack(aligned_layers), p=2, dim=2)
            else:
                all_layers_norm = F.normalize(all_layers_tensor, p=2, dim=2)

            raw_types = ds[i]["type_embeddings"]
            type_to_token_indices = defaultdict(list)
            for token_idx, t_val in enumerate(raw_types):
                if isinstance(t_val, list):
                    for t in t_val:
                        if t != 0: type_to_token_indices[t].append(token_idx)
                else:
                    if t_val != 0: type_to_token_indices[t_val].append(token_idx)

            for t_id, token_indices in type_to_token_indices.items():
                indices_tensor = torch.tensor(token_indices, device=device)
                selected_states = all_layers_norm.index_select(1, indices_tensor)  # (layers, anchors, dim)
                anchor_vec = anchors[t_id].unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
                similarities = torch.sum(selected_states * anchor_vec, dim=2)  # (layers, anchors)
                layer_sums = similarities.sum(dim=1).cpu().tolist()
                count = len(token_indices)
                for layer_idx, val in enumerate(layer_sums):
                    target_stats[t_id][layer_idx][0] += val
                    target_stats[t_id][layer_idx][1] += count

                other_indices = list(all_indices_set - set(token_indices))
                if other_indices:
                    other_tensor = torch.tensor(other_indices, device=device)
                    other_states = all_layers_norm.index_select(1, other_tensor)
                    other_similarities = torch.sum(other_states * anchor_vec, dim=2)
                    other_layer_sums = other_similarities.sum(dim=1).cpu().tolist()
                    other_count = len(other_indices)
                    for layer_idx, val in enumerate(other_layer_sums):
                        other_stats[t_id][layer_idx][0] += val
                        other_stats[t_id][layer_idx][1] += other_count

    def aggregate_stats(stats_dict):
        aggregated = {}
        for t_id, layer_data in stats_dict.items():
            layers_sorted = sorted(layer_data.keys())
            aggregated[t_id] = [layer_data[l][0] / layer_data[l][1] if layer_data[l][1] > 0 else 0.0 for l in layers_sorted]
        return aggregated

    return aggregate_stats(target_stats), aggregate_stats(other_stats)


def compute_procrustes_and_similarity(llm_model, ds, initial_embeddings, hidden_states, relative_to_previous=False, target_layer_idx=0):
    """ Calculates the global rotation matrix R, Procrustes similarity score and anisotropy of change. """
    device = next(llm_model.parameters()).device
    correlations = {}
    norm_sq_source = defaultdict(float)
    norm_sq_target = defaultdict(float)

    with torch.no_grad():
        for i in range(len(ds)):
            input_ids = torch.tensor(ds[i]["input_ids"], device=device).unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device=device).unsqueeze(0)
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            all_layers_tensor = torch.stack([initial_embeddings[0]] + hidden_states).squeeze(1).to(device)
            num_layers = all_layers_tensor.shape[0]

            for layer_idx in range(num_layers):
                source_layer = all_layers_tensor[layer_idx].to(torch.float64)
                if relative_to_previous:
                    target_layer_idx = max(0, layer_idx - 1)
                    target_layer = all_layers_tensor[target_layer_idx].to(torch.float64)
                else:
                    target_layer = all_layers_tensor[target_layer_idx].to(torch.float64)

                M_batch = torch.matmul(target_layer.transpose(0, 1), source_layer)
                if layer_idx not in correlations:
                    correlations[layer_idx] = M_batch
                else:
                    correlations[layer_idx] += M_batch
                norm_sq_source[layer_idx] += torch.sum(source_layer ** 2).item()
                norm_sq_target[layer_idx] += torch.sum(target_layer ** 2).item()

    results = {}
    for layer_idx, M in correlations.items():
        M_to_use = M
        ns_src = norm_sq_source[layer_idx]
        ns_tgt = norm_sq_target[layer_idx]
        U, S, Vh = torch.linalg.svd(M_to_use)
        R = torch.matmul(Vh.T, U.T)

        numerator = torch.sum(S).item()
        denom = (max(ns_src, 0) ** 0.5) * (max(ns_tgt, 0) ** 0.5)
        similarity = numerator / denom if denom > 0 else 0.0

        S_norm = S / torch.sum(S)
        svd_entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9)).item()  # Calculate Shannon entropy
        max_entropy = math.log(S.shape[0])  # Maximum possible entropy occurs if all singular values are perfectly equal
        anisotropy = 1.0 - (svd_entropy / max_entropy)  # 0.0 = uniform/preserved, closer to 1.0 = highly collapsed/changed

        results[layer_idx] = {
            "R": R.to(torch.float32),
            "similarity": similarity,
            "anisotropy": anisotropy
        }

    return results


def compute_procrustes_scipy(llm_model, ds, initial_embeddings, hidden_states, target_layer_idx=0):
    """ Same as above, but using built-ins """
    device = next(llm_model.parameters()).device
    X_raw = defaultdict(list)
    Y_raw = defaultdict(list)

    with torch.no_grad():
        for i in range(len(ds)):
            input_ids = torch.tensor(ds[i]["input_ids"], device=device).unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device=device).unsqueeze(0)
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            all_layers_tensor = torch.stack([initial_embeddings[0]] + hidden_states).squeeze(1).cpu().numpy()
            num_layers = all_layers_tensor.shape[0]
            target_layer = all_layers_tensor[target_layer_idx]  # (seq_len, dim)
            for layer_idx in range(num_layers):
                X_raw[layer_idx].append(all_layers_tensor[layer_idx, :, :])
                Y_raw[layer_idx].append(target_layer[:, :])

    results = {}
    for layer_idx in range(num_layers):
        X_c = np.vstack(X_raw[layer_idx])
        Y_c = np.vstack(Y_raw[layer_idx])
        R, _ = orthogonal_procrustes(X_c, Y_c)  # minimize ||X_c @ R - Y_c||_F
        if layer_idx == 0:
            print(f"Procrustes input X_c shape: {X_c.shape}")
            print(f"Procrustes input Y_c shape: {Y_c.shape}")
            print(f"Procrustes output R shape: {R.shape}")
        results[layer_idx] = {"R": torch.tensor(R, dtype=torch.float32)}
    return results


def visualize_type_similarity(type_sim, other_sim, uncurved_type_sim, uncurved_other_sim, filename=None):
    first_type_id = next(iter(uncurved_type_sim))
    layers = list(range(len(uncurved_type_sim[first_type_id])))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)

    type_ids = sorted(TYPE_NAME_MAP.keys())
    cmap = cm.get_cmap('tab10') if len(type_ids) <= 10 else cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=max(1, len(type_ids) - 1))
    plot_configs = [
        (ax1, type_sim, other_sim, "Without Procrustes alignment", True),
        (ax2, uncurved_type_sim, uncurved_other_sim, "With Procrustes alignment", False)
    ]

    for ax, data_target, data_other, title, first in plot_configs:
        for i, t_id in enumerate(type_ids):
            color = cmap(i) if len(type_ids) <= 10 else cmap(norm(i))
            if t_id in data_target:
                means = data_target[t_id]
                ax.plot(layers, means, marker='o', markersize=5, linestyle='-', color=color, linewidth=2, alpha=0.8)
            if t_id in data_other:
                means_other = data_other[t_id]
                ax.plot(layers, means_other, marker='x', markersize=4, linestyle=':', color=color, linewidth=1.5, alpha=1.0)

        ax.axhline(1.0, color='forestgreen', linewidth=1.5, linestyle='--', alpha=0.6)
        ax.axhline(0.707, color='orange', linewidth=1.5, linestyle='-.', alpha=0.6)
        ax.axhline(0.0, color='black', linewidth=1.5, linestyle=':', alpha=0.6)

        ax.set_xlabel("Transformer layer")
        if first:
            ax.set_ylabel("Cosine similarity to type vector")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    custom_handles = []
    for i, t_id in enumerate(type_ids):
        color = cmap(i) if len(type_ids) <= 10 else cmap(norm(i))
        label_name = TYPE_NAME_MAP.get(t_id, f"Type {t_id}")
        custom_handles.append(Line2D([0], [0], color=color, lw=2, label=label_name))
    custom_handles.append(Line2D([0], [0], color='gray', lw=2, linestyle='-', marker='o', label='Against target tokens'))
    custom_handles.append(Line2D([0], [0], color='gray', lw=1.5, linestyle=':', marker='x', label='Non-target tokens'))
    custom_handles.append(Line2D([0], [0], color='forestgreen', lw=1.5, linestyle='--', alpha=0.6, label="Parallel (1.0)"))
    custom_handles.append(Line2D([0], [0], color='orange', lw=1.5, linestyle='-.', alpha=0.6, label="Diagonal (0.707)"))
    custom_handles.append(Line2D([0], [0], color='black', lw=1.5, linestyle=':', alpha=0.6, label="Orthogonal (0.0)"))
    fig.legend(handles=custom_handles, loc='outside upper center', ncol=6, frameon=True)
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def inspect_token_linearity(llm_model, ds, manifold_changes, initial_embeddings, hidden_states, target_ids):
    """ Evaluates linearity against specific token embeddings, yielding sample-wise trajectories. """
    device = next(llm_model.parameters()).device
    embed_weights = llm_model.tok_embeddings.weight.detach()
    anchors = {t_id: F.normalize(embed_weights[t_id].view(1, 1, -1), p=2, dim=2) for t_id in target_ids}
    trajectories = defaultdict(list)
    with torch.no_grad():
        for i in range(len(ds)):
            input_ids_list = ds[i]["input_ids"]
            if not any(t_id in input_ids_list for t_id in target_ids):
                continue

            input_ids = torch.tensor(input_ids_list, device=device).unsqueeze(0)
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, start_pos=0)

            full_evolution = [initial_embeddings[0]] + hidden_states
            all_layers_tensor = torch.stack(full_evolution).squeeze(1).to(device)
            aligned_layers = []
            for layer_idx in range(all_layers_tensor.shape[0]):
                state = all_layers_tensor[layer_idx]
                if layer_idx == 0 or manifold_changes is None:
                    aligned_layers.append(state)
                else:
                    R = manifold_changes[layer_idx]["R"].to(device)
                    aligned_layers.append(torch.matmul(state, R))

            aligned_norm = F.normalize(torch.stack(aligned_layers), p=2, dim=2)  # (layers, seq, dim)
            for t_id in target_ids:
                token_indices = [idx for idx, val in enumerate(input_ids_list) if val == t_id]
                indices_tensor = torch.tensor(token_indices, device=device)
                selected_states = aligned_norm.index_select(1, indices_tensor)
                similarities = torch.sum(selected_states * anchors[t_id], dim=2)
                sample_trajectory = similarities.mean(dim=1).cpu().tolist()
                trajectories[t_id].append(sample_trajectory)
    return trajectories


def visualize_combined_trajectories(unaligned_tokens, aligned_tokens, target_tokens, filename=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, constrained_layout=True)
    token_cmap = cm.get_cmap('tab10')

    layer_boundaries = [0, 8, 16, 32, 64, 128]
    visual_anchors = np.arange(len(layer_boundaries))
    plot_configs = [
        (ax1, unaligned_tokens, "Without Procrustes alignment", True),
        (ax2, aligned_tokens, "With Procrustes alignment", False)
    ]
    for ax, tok_dict, title, first in plot_configs:
        for i, t_id in enumerate(target_tokens):
            color = token_cmap(i)
            label_added = False
            for traj in tok_dict.get(t_id, []):
                ax.plot(range(len(traj)), traj, linestyle='-', color=color, linewidth=1, alpha=0.01,
                        label=f"Token {t_id}" if not label_added else "", rasterized=True)
                label_added = True

        ax.axhline(1.0, color='forestgreen', linewidth=1.5, linestyle='--', alpha=0.6, label="Parallel (1.0)")
        ax.axhline(0.707, color='orange', linewidth=1.5, linestyle='-.', alpha=0.6, label="Diagonal (0.707)")
        ax.axhline(0.0, color='black', linewidth=1.5, linestyle=':', alpha=0.6, label="Orthogonal (0.0)")
        ax.set_xscale('function', functions=(
            lambda x: np.interp(x, layer_boundaries, visual_anchors),
            lambda y: np.interp(y, visual_anchors, layer_boundaries)
        ))
        ax.set_xticks(layer_boundaries)
        ax.set_xlim(0, layer_boundaries[-1])
        ax.set_xlabel("Transformer layer")
        if first:
            ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = fig.legend(by_label.values(), by_label.keys(), loc='outside upper center', ncol=min(6, len(by_label)), frameon=True)
    for line in leg.get_lines():
        line.set_alpha(1.0)
        if line.get_linestyle() == '-':
            line.set_linewidth(2.0)

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
