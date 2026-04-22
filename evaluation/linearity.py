import math

import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from dataset.processor import TYPE_NAME_MAP


def inspect_linearity(llm_model, ds, initial_embeddings, hidden_states):
    type_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    device = next(llm_model.parameters()).device
    anchors = F.normalize(llm_model.type_embeddings.detach(), p=2, dim=1)  # (anchors, dim)
    total_samples = len(ds)

    with torch.no_grad():
        for i in range(total_samples):
            input_ids = torch.tensor(ds[i]["input_ids"], device=device).unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device=device).unsqueeze(0)  # (1, seq, anchors)
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            full_evolution = [initial_embeddings[0]] + hidden_states
            all_layers_tensor = torch.stack(full_evolution).squeeze(1).to(device)  # (layers, seq, dim)
            all_layers_norm = F.normalize(all_layers_tensor, p=2, dim=2)  # (layers, seq, dim)
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
                anchor_vec = anchors[t_id].unsqueeze(0).unsqueeze(0)  # (1, 1, Dim)
                similarities = torch.sum(selected_states * anchor_vec, dim=2)  # (layers, anchors)
                layer_sums = similarities.sum(dim=1).cpu().tolist()
                count = len(token_indices)

                for layer_idx, val in enumerate(layer_sums):
                    type_stats[t_id][layer_idx][0] += val
                    type_stats[t_id][layer_idx][1] += count

    # type_id -> [avg_sim_layer_0, avg_sim_layer_1, ...]
    aggregated_results = {}
    for t_id, layer_data in type_stats.items():
        layers_sorted = sorted(layer_data.keys())
        means = []
        for l in layers_sorted:
            total_sim, count = layer_data[l]
            means.append(total_sim / count if count > 0 else 0.0)
        aggregated_results[t_id] = means
    return aggregated_results


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


def compute_procrustes_scipy(llm_model, ds, initial_embeddings, hidden_states, target_layer_idx=-1):
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
        results[layer_idx] = {"R": torch.tensor(R, dtype=torch.float32)}
    return results


def inspect_rotated_linearity(llm_model, ds, manifold_changes, initial_embeddings, hidden_states):
    """ Applies the pre-calculated global rotations to the hidden states and measures linearity against the anchors. """
    type_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    device = next(llm_model.parameters()).device
    anchors = F.normalize(llm_model.type_embeddings.detach(), p=2, dim=1)

    with torch.no_grad():
        for i in range(len(ds)):
            input_ids = torch.tensor(ds[i]["input_ids"], device=device).unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device=device).unsqueeze(0)
            initial_embeddings.clear()
            hidden_states.clear()

            _ = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)

            full_evolution = [initial_embeddings[0]] + hidden_states
            all_layers_tensor = torch.stack(full_evolution).squeeze(1).to(device)
            aligned_layers = []

            for layer_idx in range(all_layers_tensor.shape[0]):
                state = all_layers_tensor[layer_idx]  # (seq, dim)
                if layer_idx == 0:
                    aligned_layers.append(state)
                else:
                    R = manifold_changes[layer_idx]["R"].to(device)
                    aligned_state = torch.matmul(state, R)
                    aligned_layers.append(aligned_state)

            type_to_token_indices = defaultdict(list)
            for token_idx, t_val in enumerate(ds[i]["type_embeddings"]):
                if isinstance(t_val, list):
                    for t in t_val:
                        if t != 0: type_to_token_indices[t].append(token_idx)
                else:
                    if t_val != 0: type_to_token_indices[t_val].append(token_idx)

            aligned_norm = F.normalize(torch.stack(aligned_layers), p=2, dim=2)  # (layers, seq, dim)
            for t_id, token_indices in type_to_token_indices.items():
                indices_tensor = torch.tensor(token_indices, device=device)
                selected_states = aligned_norm.index_select(1, indices_tensor)
                anchor_vec = anchors[t_id].view(1, 1, -1)
                similarities = torch.sum(selected_states * anchor_vec, dim=2)
                layer_sums = similarities.sum(dim=1).cpu().tolist()
                count = len(token_indices)
                for l_idx, val in enumerate(layer_sums):
                    type_stats[t_id][l_idx][0] += val
                    type_stats[t_id][l_idx][1] += count

    results = {}
    for t_id, layer_data in type_stats.items():
        layers_sorted = sorted(layer_data.keys())
        means = []
        for l in layers_sorted:
            total_sim, count = layer_data[l]
            means.append(total_sim / count if count > 0 else 0.0)
        results[t_id] = means
    return results


def visualize_type_similarity(type_similarity, uncurved_type_similarity, filename=None):
    first_type_id = next(iter(uncurved_type_similarity))
    layers = list(range(len(uncurved_type_similarity[first_type_id])))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, constrained_layout=True)
    type_ids = sorted(TYPE_NAME_MAP.keys())
    cmap = cm.get_cmap('tab10') if len(type_ids) <= 10 else cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0, vmax=max(1, len(type_ids) - 1))

    plot_configs = [
        (ax1, type_similarity, "Without Procrustes alignment", True),
        (ax2, uncurved_type_similarity, "With Procrustes alignment", False)
    ]

    for ax, data_dict, title, first in plot_configs:
        for i, t_id in enumerate(type_ids):
            means = data_dict[t_id]
            color = cmap(i) if len(type_ids) <= 10 else cmap(norm(i))
            label_name = TYPE_NAME_MAP.get(t_id, f"Type {t_id}")
            ax.plot(layers, means, marker='o', markersize=5, linestyle='-',
                    label=label_name, color=color, linewidth=2, alpha=0.8)

        ax.axhline(1.0, color='forestgreen', linewidth=1.5, linestyle='--', alpha=0.6, label="Parallel (1.0)")
        ax.axhline(0.707, color='orange', linewidth=1.5, linestyle='-.', alpha=0.6, label="Diagonal (0.707)")
        ax.axhline(0.0, color='black', linewidth=1.5, linestyle=':', alpha=0.6, label="Orthogonal (0.0)")
        ax.set_xlabel("Transformer layer")
        if first:
            ax.set_ylabel("Cosine similarity to type vector")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=min(5, len(handles)), frameon=True)
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
