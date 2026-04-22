import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.processor import special_tokens


def inspect_eval_missed_and_hallucinations(llm_model, ds):
    missed = {x: 0 for x in range(150)}
    hallucinated = {x: 0 for x in range(150)}
    correct_cot = 0
    incorrect_cot = 0

    with torch.no_grad():
        for i in range(len(ds)):
            input_ids = torch.tensor(ds[i]["input_ids"], device="cuda").unsqueeze(0)
            type_embeddings = torch.tensor(ds[i]["type_embeddings"], device="cuda").unsqueeze(0)
            expected_sequence = ds[i]["labels"]
            expected_full_sequence = ds[i]["full_labels"]
            generated_sequence = llm_model.generate(
                input_ids=input_ids, type_embeddings=type_embeddings,
                max_new_tokens=32, stop_tokens=list(special_tokens.values()))
            num_input_tokens = input_ids.shape[1]
            generated_ids = generated_sequence[0, num_input_tokens:].tolist()
            expected_generated_ids = expected_sequence[num_input_tokens - 1:]

            if expected_generated_ids[-1] == generated_ids[-1]:
                correct_cot += 1
            else:
                incorrect_cot += 1

            for elem in set(generated_ids[:-1]) - set(expected_full_sequence[:-1]):
                if elem >= 150:
                    print("invalid seq:", generated_ids)
                    continue
                hallucinated[elem] += 1

            for elem in set(expected_generated_ids[:-1]) - set(generated_ids[:-1]):
                if elem >= 150:
                    print("invalid seq:", generated_ids)
                    continue
                missed[elem] += 1

    print(f"+: {correct_cot} -: {incorrect_cot}")
    print("hallucinated:", hallucinated)
    print("missed:", missed)
    return hallucinated, missed


def visualize_hallucinations(hallucinated, missed, y_lim=50, x_lim=60, filename=None):
    from matplotlib.ticker import MaxNLocator

    hallucinated_counts = [x for x in hallucinated.values() if x > 0]
    missed_counts = [x for x in missed.values() if x > 0]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5, 3))

    all_counts = hallucinated_counts + missed_counts
    min_val = min(all_counts)
    max_val = max(all_counts)
    bins = np.arange(min_val, max_val + 2) - 0.5

    ax.hist([hallucinated_counts, missed_counts], bins=bins, color=['skyblue', 'salmon'], label=['Hallucinated deductions', 'Missed deductions'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Count per fact ID')
    ax.set_ylabel('Frequency (number of fact IDs)')
    ax.set_ylim(0, y_lim)
    ax.set_xlim(0, x_lim)
    ax.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
