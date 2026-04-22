import os
import ast
import matplotlib.pyplot as plt

MODELS = ["gemma3_27b", "granite4_32B", "mistral32_24B", "qwen3_30B"]
MODEL_COLORS = {
    "gemma3_27b": "#0077BB",   # Strong Blue
    "granite4_32B": "#EE7733", # Orange
    "mistral32_24B": "#009988",# Teal
    "qwen3_30B": "#CC3311"     # Red
}
MODEL_LABELS = {
    "gemma3_27b": "Gemma 3 (27B)",
    "granite4_32B": "Granite 4 (32B)",
    "mistral32_24B": "Mistral Small (24B)",
    "qwen3_30B": "Qwen 3 (30B)"
}
STYLES = {
    "direct": {"linestyle": "--", "marker": "x", "alpha": 0.7, "linewidth": 2, "markersize": 6},
    "CoT": {"linestyle": "-",  "marker": "o", "alpha": 0.9, "linewidth": 2.5, "markersize": 6}
}


# ---------------------------------------------------------
# Data Parsing Logic
# ---------------------------------------------------------

def parse_summary_from_file(filepath, max_depth=6):
    """
    Reads the file, finds the summary line, and calculates accuracy per depth.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = [line for line in f if "summary=" in line][0]
    dict_str = content.split("summary=", 1)[1].strip()
    data_dict = ast.literal_eval(dict_str)
    correct_counts = data_dict.get('correct', {})
    incorrect_counts = data_dict.get('incorrect', {})

    accuracies = {}
    all_depths = set(i for i in range(0, max_depth + 1))
    for d in all_depths:
        c = correct_counts.get(d, 0)
        i = incorrect_counts.get(d, 0)
        total = c + i
        if total > 0:
            accuracies[d] = c / total
        else:
            accuracies[d] = 0.0
    return dict(sorted(accuracies.items()))


def collect_data():
    """
    Traverses the directory structure and collects data for all models.
    Structure: results[dataset_type][model_name][mode] = {depth: acc}
    """
    results = {"LP": {}, "RP": {}}
    for model in MODELS:
        results["LP"][model] = {}
        results["RP"][model] = {}
        files_map = [
            ("LP", "direct", "results_validation_lp_balanced_is_cot=False.txt"),
            ("LP", "CoT", "results_validation_lp_balanced_is_cot=True.txt"),
            ("RP", "direct", "results_validation_rp_balanced_is_cot=False.txt"),
            ("RP", "CoT", "results_validation_rp_balanced_is_cot=True.txt"),
        ]
        for dataset, mode, filename in files_map:
            filepath = os.path.join(model, filename)
            data = parse_summary_from_file(filepath)
            results[dataset][model][mode] = data
    return results


# ---------------------------------------------------------
# Plotting Logic
# ---------------------------------------------------------
def plot_results(results):
    datasets = ["LP", "RP"]
    titles = {
        "LP": "LP Evaluation (Hierarchical)",
        "RP": "RP Evaluation (Entangled)"
    }

    for dataset in datasets:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.figure(figsize=(5, 5))
        # plt.title(titles[dataset], fontsize=14, fontweight='bold')
        plt.xlabel("Logical depth", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(-0.05, 1.05)

        for model in MODELS:
            modes = results[dataset].get(model, {})

            if "direct" in modes and modes["direct"]:
                x = list(modes["direct"].keys())
                y = list(modes["direct"].values())
                plt.plot(x, y,
                         color=MODEL_COLORS[model],
                         linestyle=STYLES["direct"]["linestyle"],
                         marker=STYLES["direct"]["marker"],
                         alpha=STYLES["direct"]["alpha"],
                         label=f"{MODEL_LABELS[model]} (direct)")

            if "CoT" in modes and modes["CoT"]:
                x = list(modes["CoT"].keys())
                y = list(modes["CoT"].values())
                plt.plot(x, y,
                         color=MODEL_COLORS[model],
                         linestyle=STYLES["CoT"]["linestyle"],
                         marker=STYLES["CoT"]["marker"],
                         linewidth=2,
                         alpha=STYLES["CoT"]["alpha"],
                         label=f"{MODEL_LABELS[model]} (CoT)")

        if dataset == "RP":
            plt.legend(loc='lower left', fontsize=9, framealpha=0.9)
        plt.tight_layout()
        filename = f"benchmark_results_{dataset}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Success! Saved {filename}")
        plt.close()


plot_results(collect_data())
