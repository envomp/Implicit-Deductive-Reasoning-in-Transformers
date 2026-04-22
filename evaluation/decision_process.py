import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from dataset.processor import special_tokens


class ZeroShotDecisionProbe:
    def __init__(self, llm_model, manifold_changes=None):
        self.device = next(llm_model.parameters()).device
        self.norm = llm_model.norm
        self.manifold_changes = manifold_changes
        self.false_id = special_tokens[0]
        self.true_id = special_tokens[1]
        self.probe_weights = torch.stack([
            llm_model.output.weight[self.false_id],
            llm_model.output.weight[self.true_id]
        ]).detach().to(self.device)

    def forward(self, hidden_states, layer_idx=None):
        """Returns raw logits of shape [..., 2] for any input tensor."""
        states = hidden_states.to(self.device)
        if self.manifold_changes is not None and layer_idx is not None:
            R = self.manifold_changes[layer_idx]["R"].to(self.device)
            states = torch.matmul(states, R)
        normed_states = self.norm(states)
        return torch.matmul(normed_states, self.probe_weights.T)

    def predict(self, hidden_states, layer_idx=None):
        logits = self.forward(hidden_states, layer_idx)
        return torch.argmax(logits, dim=-1)


def inspect_eval_decision_process(llm_model, ds, initial_embeddings, hidden_states, manifold_changes=None):
    device = next(llm_model.parameters()).device
    all_correct_probabilities = []
    all_correct_logits = []
    all_incorrect_logits = []
    num_correct = 0
    total_samples = len(ds)

    probe = ZeroShotDecisionProbe(llm_model, manifold_changes)
    with torch.no_grad():
        for i in range(total_samples):
            input_ids = torch.tensor(ds[i]["input_ids"], device="cuda").unsqueeze(0)
            type_ids = torch.tensor(ds[i]["type_embeddings"], device="cuda").unsqueeze(0)
            expected_answer_id = ds[i]["labels"][-1]
            initial_embeddings.clear()
            hidden_states.clear()

            generated_sequence = llm_model.forward(input_ids=input_ids, type_embeddings=type_ids, start_pos=0)
            predicted_id = torch.argmax(generated_sequence[0, -1, :], dim=-1, keepdim=True).item()
            if predicted_id == expected_answer_id:
                num_correct += 1

            all_layers_tensor = torch.stack([initial_embeddings[0]] + hidden_states).squeeze(1).to(device)
            query_token_pos = input_ids.shape[1] - 1
            query_token_evolution = all_layers_tensor[:, query_token_pos, :]

            layer_probabilities = []
            layer_correct_logits = []
            layer_incorrect_logits = []
            target_idx = 1 if expected_answer_id == special_tokens[1] else 0
            opposite_idx = 1 - target_idx
            for layer_idx, vec in enumerate(query_token_evolution):
                logits = probe.forward(vec, layer_idx=layer_idx).squeeze(0)
                probs = F.softmax(logits, dim=-1)
                layer_correct_logits.append(logits[target_idx].item())
                layer_incorrect_logits.append(logits[opposite_idx].item())
                layer_probabilities.append(probs[target_idx].item())
            all_correct_probabilities.append(layer_probabilities)
            all_correct_logits.append(layer_correct_logits)
            all_incorrect_logits.append(layer_incorrect_logits)

    accuracy = round((num_correct / total_samples) * 100 if total_samples > 0 else 0, 1)
    print("\n" + "=" * 50)
    print("Softmax Probability Inspection Summary")
    print(f"Correct Predictions: {num_correct} / {total_samples} ({accuracy:.2f}%)")
    print("=" * 50 + "\n")
    return all_correct_probabilities, all_correct_logits, all_incorrect_logits, accuracy

def get_layerwise_accuracies(correct_logits, incorrect_logits):
    c_logits_np = np.array(correct_logits)
    i_logits_np = np.array(incorrect_logits)
    is_correct = c_logits_np > i_logits_np
    layer_accuracies = np.mean(is_correct, axis=0) * 100
    return [round(float(acc), 1) for acc in layer_accuracies]

def print_results_table(results_dict):
    first_key = list(results_dict.keys())[0]
    num_layers = len(results_dict[first_key])
    headers = ["Dataset (Evaluation)"] + [f"Layer {i}" for i in range(num_layers)]
    col_widths = [max(len(str(k)) for k in results_dict.keys()) + 2]
    for i in range(num_layers):
        col_widths.append(max(len(headers[i+1]), 6) + 2)
    def format_row(items):
        return "|" + "|".join(f"{str(item):^{width}}" for item, width in zip(items, col_widths)) + "|"
    print("\n" + format_row(headers))
    print("|" + "|".join("-" * width for width in col_widths) + "|")
    for row_name, accuracies in results_dict.items():
        acc_strings = [f"{acc:.1f}" for acc in accuracies]
        row_items = [", ".join([str(x) for x in row_name])] + acc_strings
        print(format_row(row_items))

def visualize_decision_process(correct_probs, correct_logits, incorrect_logits, accuracy, filename=None):
    correct_probs_np = np.array(correct_probs)
    prob_median = np.median(correct_probs_np, axis=0)
    prob_p25 = np.percentile(correct_probs_np, 25, axis=0)
    prob_p75 = np.percentile(correct_probs_np, 75, axis=0)
    c_logits_np = np.array(correct_logits)
    i_logits_np = np.array(incorrect_logits)
    c_p25 = np.percentile(c_logits_np, 25, axis=0)
    c_p50 = np.median(c_logits_np, axis=0)
    c_p75 = np.percentile(c_logits_np, 75, axis=0)
    i_p25 = np.percentile(i_logits_np, 25, axis=0)
    i_p50 = np.median(i_logits_np, axis=0)
    i_p75 = np.percentile(i_logits_np, 75, axis=0)
    layer_indices = np.arange(len(prob_median))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(4, 2))
    ax2 = ax.twinx()

    bar_width = 0.35
    opacity = 0.25
    ax2.bar(layer_indices - bar_width / 2, c_p75 - c_p25, bottom=c_p25, color='#1f77b4', alpha=opacity, width=bar_width, zorder=1)
    ax2.scatter(layer_indices - bar_width / 2, c_p50, color='#1f77b4', marker='_', s=20, linewidth=1, zorder=2)
    ax2.bar(layer_indices + bar_width / 2, i_p75 - i_p25, bottom=i_p25, color='#d62728', alpha=opacity, width=bar_width, zorder=1)
    ax2.scatter(layer_indices + bar_width / 2, i_p50, color='#d62728', marker='_', s=20, linewidth=1, zorder=2)
    ax2.set_ylabel("Logit magnitude")
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.grid(False)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot(layer_indices, prob_median, color='#2ca02c', marker='o', markersize=2, linewidth=1, linestyle='-', zorder=10)
    ax.fill_between(layer_indices, prob_p25, prob_p75, color='#2ca02c', alpha=0.15, zorder=9)
    ax.plot(layer_indices, prob_p25, color='#2ca02c', linestyle=':', linewidth=0.5, alpha=0.5, zorder=10)
    ax.plot(layer_indices, prob_p75, color='#2ca02c', linestyle=':', linewidth=0.5, alpha=0.5, zorder=10)

    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=0.8, zorder=5)
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=1.5, label='Probability'),
        Patch(facecolor='#1f77b4', alpha=opacity, label='Correct logit'),
        Patch(facecolor='#d62728', alpha=opacity, label='Incorrect logit'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.25),
              ncol=3, fontsize=8, frameon=False, columnspacing=1)

    ax.text(0.02, 0.06, f'Final accuracy: {accuracy}%',
            transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgrey'))

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Probability")
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='y', labelcolor='#2ca02c')
    ax.set_xticks(layer_indices)
    ax.set_ylim(0, 1.05)
    plt.tight_layout(pad=0)

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
