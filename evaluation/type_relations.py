import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

def get_spike_data(type_embeddings, type_name_map, threshold=3.0):
    records = []
    for idx, name in type_name_map.items():
        vec = type_embeddings[idx].detach().cpu().numpy()
        mean = np.mean(vec)
        std = np.std(vec)
        z_scores = (vec - mean) / (std + 1e-9)

        spike_indices = np.where(np.abs(z_scores) > threshold)[0]

        for s_idx in spike_indices:
            records.append({
                "Type": name,
                "Dimension": s_idx,
                "Value": vec[s_idx],
                "Z-Score": z_scores[s_idx]
            })
    return pd.DataFrame(records)

def visualize_dimension_relations(df):
    pivot_df = df.pivot(index="Type", columns="Dimension", values="Value").fillna(0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_df, annot=True, cmap="RdBu_r", center=0,
                cbar_kws={'label': 'Embedding value'})
    plt.xlabel("Model dimension index")
    plt.ylabel("Token type")
    plt.tight_layout()
    plt.show()

def calculate_similarity_matrix(weights, type_name_map):
    """
    Calculates the cosine similarity between each pair of semantic types.
    """
    # Filter only indices that exist in the weights
    valid_indices = [idx for idx in type_name_map.keys() if idx < weights.shape[0]]
    names = [type_name_map[idx] for idx in valid_indices]

    # Extract the relevant vectors
    selected_vectors = weights[valid_indices] # (num_types, d_model)

    # Compute Cosine Similarity: (A @ A.T) / (norm(A) * norm(A.T))
    # Using F.cosine_similarity is often easier for pairs, but matrix multiplication is faster here.
    norm_vectors = F.normalize(selected_vectors, p=2, dim=1)
    sim_matrix = torch.mm(norm_vectors, norm_vectors.t())

    return pd.DataFrame(sim_matrix.detach().cpu().numpy(), index=names, columns=names)

def visualize_similarity_matrix(sim_df):
    """
    Plots the similarity matrix as a heatmap.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Cosine similarity'}
    )
    plt.tight_layout()
    plt.show()
