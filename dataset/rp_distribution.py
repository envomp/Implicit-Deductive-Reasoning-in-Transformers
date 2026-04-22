import collections
from tqdm import tqdm
import numpy as np
from sample import sample_one_rule, sample_rule_priority
import random
import matplotlib.pyplot as plt
from scipy import stats


def get_proof_depth(rules, facts, query):
    if query in facts:
        return 0
    provable_facts = set(facts)
    depth = 0

    while True:
        depth += 1
        newly_derived_facts = set()
        for premise, conclusion in rules:
            if conclusion not in provable_facts and set(premise).issubset(provable_facts):
                newly_derived_facts.add(conclusion)
        if not newly_derived_facts:
            return None
        if query in newly_derived_facts:
            return depth
        provable_facts.update(newly_derived_facts)

def sample_rule_priority_deterministic(preds):
    pred_num = len(preds)
    rule_num = pred_num
    fact_num = 3

    cache = set()
    rules = []
    for _ in range(0, rule_num):
        rule = None
        while True:
            rule = sample_one_rule(preds)
            rule_hash = ' '.join(sorted(rule[0])) + ' ' + rule[1]
            if rule_hash not in cache:
                cache.add(rule_hash)
                break
        rules.append(rule)

    facts = random.sample(preds, fact_num)

    query = random.sample(preds, 1)[0]

    return rules, facts, query

if __name__ == "__main__":
    NUM_RUNS = 10
    NUM_SAMPLES_PER_RUN = 100_000
    all_run_distributions = []
    DETERMINISTIC = False

    print(f"Running {NUM_RUNS} simulations of {NUM_SAMPLES_PER_RUN} samples each...")
    for i in range(NUM_RUNS):
        print(f"\n--- Running Simulation {i + 1}/{NUM_RUNS} ---")
        run_distribution = collections.defaultdict(int)
        for _ in tqdm(range(NUM_SAMPLES_PER_RUN)):
            if DETERMINISTIC:
                preds = [str(i) for i in range(30)]
                rules, facts, query = sample_rule_priority_deterministic(preds)
            else:
                preds = [str(i) for i in range(random.randint(5, 30))]
                rules, facts, query = sample_rule_priority(preds)

            depth = get_proof_depth(rules, set(facts), query)
            run_distribution[depth] += 1
        all_run_distributions.append(run_distribution)

    print("\n" + "=" * 80)
    print("      Averaged Proof Depth Distribution for RP Samples")
    print(f"      (Total Runs: {NUM_RUNS}, Samples per Run: {NUM_SAMPLES_PER_RUN})")
    print("=" * 80)
    print(f"{'Proof Depth':<15} | {'Count':<15}")
    print("-" * 80)

    total_samples = NUM_RUNS * NUM_SAMPLES_PER_RUN
    all_keys = set().union(*[d.keys() for d in all_run_distributions])
    sorted_keys = sorted([k for k in all_keys if k is not None])

    plot_depths = []
    plot_counts = []

    for depth in sorted_keys:
        total_count = sum(run.get(depth, 0) for run in all_run_distributions)
        print(f"{depth:<15} | {f'{total_count}':<15}")
        if depth >= 1 and total_count > 0:
            plot_depths.append(depth)
            plot_counts.append(total_count)

    unprovable_count = sum(run.get(None, 0) for run in all_run_distributions)
    print(f"{'Unprovable':<15} | {f'{unprovable_count}':<15}")
    print("-" * 80)

    log_percentages = np.log(plot_counts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(plot_depths, log_percentages)
    r_squared = r_value ** 2
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(plot_depths, log_percentages, label='Empirical data', color='blue', zorder=5)
    fit_line = [slope * d + intercept for d in plot_depths]
    ax.plot(plot_depths, fit_line, color='red', linestyle='--', label=f'Linear fit ($R^2={r_squared:.3f}$)')
    ax.set_xlabel('Proof depth (δ)', fontsize=12)
    ax.set_ylabel('Log of counts', fontsize=12)
    # ax.set_title('Proof depth distribution', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"proof_depth_distribution_rp_det={DETERMINISTIC}.pdf", format='pdf', bbox_inches='tight')
