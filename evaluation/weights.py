import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_weight_distribution(model, pretty_name, identifier):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5, 3))

    all_params = []
    for name, param in model.named_parameters():
        if identifier in name:
            param_data = param.detach().cpu().numpy().flatten()
            all_params.extend(param_data)

    all_params = np.array(all_params, dtype=np.float32)
    ax.hist(all_params, bins=30, density=True, color='#2ca02c', alpha=0.6, label='Weight distribution')

    if np.all(all_params == all_params[0]):
        ax.axvline(all_params[0], color='#d62728', linestyle='-', linewidth=2, label='Constant value')
    else:
        kde = stats.gaussian_kde(all_params)
        x_vals = np.linspace(all_params.min(), all_params.max(), 100)
        ax.plot(x_vals, kde(x_vals), color='#d62728', linestyle='-', linewidth=2, label='KDE')

    ax.set_xlabel("Weight value")
    ax.set_ylabel("Density")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.set_ylim(0, 5)
    plt.tight_layout()

    if pretty_name:
        plt.savefig(f"weight_distribution_{pretty_name}_{identifier.replace('.', '_')}.pdf", format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
