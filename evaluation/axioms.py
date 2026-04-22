import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.processor import cot_answer, direct_answer


def axiom_tsne(facts_array, color_labels, legend_map, title="t-SNE"):
    from sklearn.manifold import TSNE
    import matplotlib.patches as mpatches
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(facts_array)

    plt.figure(figsize=(20, 16))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color_labels, alpha=0.3, s=50)
    patches = [mpatches.Patch(color=color, label=label) for label, color in legend_map.items()]
    plt.legend(handles=patches, title="Token type", fontsize=14)
    plt.title(title, fontsize=18)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True)
    plt.show()


def axiom_pca(facts_array, color_labels, legend_map, title="PCA"):
    import matplotlib.patches as mpatches
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(facts_array)

    plt.figure(figsize=(20, 16))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color_labels, alpha=0.3, s=50)
    patches = [mpatches.Patch(color=color, label=label) for label, color in legend_map.items()]
    plt.legend(handles=patches, title="Token type", fontsize=14)
    plt.title(title, fontsize=18)
    plt.xlabel('Principal component 1', fontsize=12)
    plt.ylabel('Principal component 2', fontsize=12)
    plt.grid(True)
    plt.show()


def calculate_effective_rank(embeddings):
    singular_values = np.linalg.svd(embeddings, compute_uv=False)
    stable_rank = np.sum(singular_values) ** 2 / np.sum(singular_values ** 2)
    return stable_rank


def shapiro_wilk_test(axiom_array, check_range):
    from scipy import stats

    normal_axiom_count = 0
    for k in check_range:
        axiom_embedding = axiom_array[k]
        stat, p_value = stats.shapiro(axiom_embedding)
        if p_value > 0.05:
            normal_axiom_count += 1
    return normal_axiom_count


def analyze_embedding_normality(embedding, name="Embedding"):
    from scipy import stats

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Normality Analysis for "{name}"', fontsize=16)

    ax1.hist(embedding, bins='auto', density=True, color='skyblue', ec='black')
    ax1.set_title("Histogram")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    stats.probplot(embedding, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_axiom_embeddings(model):
    all_ids = list(range(150)) + [cot_answer, direct_answer]
    color = list("gray" for i in range(150)) + ["blue", "green"] + ["red"]
    legend_mapping = {
        'axiom': 'gray',
        'cot task': 'blue',
        'direct task': 'green',
        'zero': 'red'
    }
    axioms = []
    for id in all_ids:
        emb = model.cpu().eval().tok_embeddings(torch.tensor([id]))[0].tolist()
        axioms.append(emb)
    axioms.append([0 for _ in range(256)])
    axiom_array = np.array(axioms)
    return axiom_array, color, legend_mapping


def plot_justification_for_corrective(model, control_token_ids=[0, 10, 100], filename=None):
    import torch
    import torch.nn.functional as F
    import seaborn as sns
    import matplotlib.pyplot as plt

    targets = {
        "CoT": cot_answer,
        "Direct": direct_answer,
    }

    embeddings = {}
    with torch.no_grad():
        for name, tid in targets.items():
            emb = model.cpu().eval().tok_embeddings(torch.tensor([tid]))[0]
            embeddings[name] = emb
        embeddings["Zero vector"] = torch.zeros_like(list(embeddings.values())[0])
        for i, tid in enumerate(control_token_ids):
            emb = model.cpu().eval().tok_embeddings(torch.tensor([tid]))[0]
            embeddings[f"Axiom {tid}"] = emb

    names = list(embeddings.keys())
    vecs = torch.stack(list(embeddings.values()))
    vecs_norm = F.normalize(vecs, p=2, dim=1)
    sim_matrix = torch.mm(vecs_norm, vecs_norm.t()).numpy()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(5, 4))
    sns.heatmap(sim_matrix, xticklabels=names, yticklabels=names,
                annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
