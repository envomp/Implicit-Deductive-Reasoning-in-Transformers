import matplotlib.pyplot as plt
from result_file_parsing import get_last_epoch_accuracy, parse_epoch, parse_depth_data

from conf import *

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'violet', 'azure']


def plot_combined_loss_curves(runs, path, x_lim, x_axis="epoch", figsize=(4, 3), y_scale="linear"):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    epsilon = 0.0001

    for i, xs in enumerate(runs):
        if len(xs) >= 3:
            run_id, file_name, color, *_ = xs
        else:
            raise RuntimeError(xs)
        with open(path + file_name, 'r', encoding='utf-8') as file:
            contents = file.read()

        tloss_curve = {}
        vloss_curve = {}
        contents = contents.split("\n\n")
        color = color if color else colors[i]
        linestyle = '-'
        for c in contents:
            epoch_info = parse_epoch(c)
            if epoch_info:
                epoch = epoch_info["epoch"] + 1
                batch = epoch_info["len_train"]
                t_loss = epoch_info["train_loss"] + epsilon
                v_loss = epoch_info["val_loss"] + epsilon

                if x_axis == "epoch":
                    key = epoch
                elif x_axis == "batch":
                    key = batch * epoch
                elif x_axis == "items":
                    key = batch * epoch * 500
                else:
                    raise RuntimeError(f"x_axis {x_axis} not supported")

                tloss_curve[key] = t_loss
                vloss_curve[key] = v_loss

        t, tloss_values = zip(*sorted(tloss_curve.items()))
        ax1.plot(t, tloss_values, linestyle=linestyle, color=color, label=run_id)
        ax1.annotate(f' {run_id}', (t[-1], tloss_values[-1]), ha='left')

        v, vloss_values = zip(*sorted(vloss_curve.items()))
        ax2.plot(v, vloss_values, linestyle=linestyle, color=color, label=run_id)
        ax2.annotate(f' {run_id}', (v[-1], vloss_values[-1]), ha='left')

    ax1.set_ylabel('tLoss')
    ax1.set_title('Loss Curves')
    ax1.grid(True)
    ax1.legend()
    ax1.set_yscale(y_scale)
    ax1.set_xlim(0, x_lim)
    ax1.set_ylim(0, None)

    ax2.set_xlabel(x_axis)
    ax2.set_ylabel('vLoss')
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale(y_scale)
    ax2.set_xlim(0, x_lim)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    plt.show()


def plot_combined_accuracy_curves(runs, path, x_lim, x_axis="epoch", figsize=(4, 3),
                                  default_filter_f=lambda x: True,
                                  default_ranges=[("-", [0, 1, 2], "d2"), ("--", [0, 1, 2, 3, 4, 5, 6], "d6")]):
    plt.figure(figsize=figsize)

    for i, xs in enumerate(runs):
        if len(xs) == 3:
            run_id, file_name, color = xs
            filter_f = default_filter_f
            ranges = default_ranges
        elif len(xs) == 4:
            run_id, file_name, color, filter_f = xs
            ranges = default_ranges
        elif len(xs) == 5:
            run_id, file_name, color, filter_f, ranges = xs
        else:
            raise RuntimeError(xs)

        with open(path + file_name, 'r', encoding='utf-8') as file:
            contents = file.read()

        combined_accuracy = {(marker, ann): {} for marker, _, ann in ranges}
        color = color if color else colors[i]
        contents = contents.strip().split("\n\n")
        epoch = 0
        batch = 0

        for c in range(len(contents)):
            epoch_info = parse_epoch(contents[c])
            if epoch_info:
                epoch = epoch_info["epoch"]
                batch += epoch_info["len_train"]
            if not epoch_info and filter_f(contents[c]):
                results_data = parse_depth_data(contents[c])
                if results_data and len(results_data) >= 3:
                    for marker, depths, ann in ranges:
                        res = 0
                        for j in depths:
                            for i in range(3):
                                res += (results_data[i][str(j)])
                        res /= 3000
                        res /= len(depths)

                        if x_axis == "epoch":
                            key = epoch
                        elif x_axis == "batch":
                            key = batch
                        elif x_axis == "items":
                            key = batch * 500
                        else:
                            raise RuntimeError(f"x_axis {x_axis} not supported")
                        combined_accuracy[(marker, ann)][key] = res

        for (linestyle, annotation), items in combined_accuracy.items():
            x_combined, combined_values = zip(*sorted(items.items()))
            plt.plot(x_combined, combined_values, linestyle=linestyle, color=color, label=f"{annotation} {run_id}")
            plt.annotate(f" {annotation} {run_id}", (x_combined[-1], combined_values[-1]), ha='left')

    plt.xlabel(x_axis)
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.xlim(0, x_lim)
    plt.ylim(0.45, 1.05)
    plt.legend()
    plt.show()


def generate_accuracy_table(runs, path):
    table_rows = []

    for i, xs in enumerate(runs):
        if 2 == len(xs):
            run_id, file_name = xs
            filter_f = lambda x: True
        elif 3 <= len(xs):
            run_id, file_name, filter_f, *_ = xs
        else:
            raise RuntimeError(f"Invalid run configuration: {xs}")

        lp_accuracies, lp_star_accuracies, rp_accuracies = get_last_epoch_accuracy(file_name, path, filter_f, average=True)
        table_rows.append(f"| {run_id} | {lp_accuracies} | {lp_star_accuracies} | {rp_accuracies} |")

    markdown_header = [
        "| Run ID | LP (%) | LP* (%) | RP (%) |",
        "| :--- | ---: | ---: | ---: |"
    ]

    return "\n".join(markdown_header + table_rows)


predicate_logs_path = "../../predicate_runs/"

direct_runs = [  # ec >= rc > no-curriculum
    ("direct ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("direct rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("direct", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_direct_runs = [  # no-curriculum >= rc > ec
    ("r2 direct ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample,r2_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 direct rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample,r2_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 direct", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=direct_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

cot_random_runs = [  # no-curriculum >= ec > rc
    ("r-cot ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r-cot rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r-cot", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

cot_eager_runs = [  # no-curriculum >= ec > rc
    ("e-cot ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("e-cot rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("e-cot", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_cot_random_runs = [  # benefited from curriculum
    ("r2 r-cot ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample,r2_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 r-cot rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample,r2_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 r-cot", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,random_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_cot_eager_runs = [  # no benefit from curriculum
    ("r2 e-cot ec", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,eager_sample,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 e-cot rc", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp,d7,curriculum,random_sample,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 e-cot", "part_1/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_direct_variations = [  # corrective >> mixed >= direct > shuffle > masked
    ("r2 direct corrective", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt", lambda x: "correction>" not in x),
    ("r2 direct mixed", "part_2/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=mixed_cot,eager_path_seed=123_dtype=float.txt", lambda x: "cot>" not in x),
    ("r2 direct shuffle", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2,custom_shuffle_solver=direct,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 direct masked", "part_2/lp_adamw_model=d256_h4_l8_do_mask=True_dataset=train_lp_downsampled,d7,r2_solver=direct,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 direct", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=direct,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_cot_variations = [  # everything > mixed
    ("r2 cot grpo", "part_2/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_grpo=True_15e0.2f-linear_seed=123_dtype=float.txt"),
    ("r2 cot corrective", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt", lambda x: "correction>" in x),
    ("r2 cot mixed", "part_2/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=mixed_cot,eager_path_seed=123_dtype=float.txt", lambda x: "cot>" in x),
    ("r2 cot shuffle", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2,custom_shuffle_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 cot masked", "part_2/lp_adamw_model=d256_h4_l8_do_mask=True_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("r2 cot", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt")
]

r2_cot_ablation = [
    ("r2 cot -wd_on_lnorm", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-06b20.99wd0.1-lnorm_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("r2 cot bfloat16", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-06b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_epochs=15_seed=123_dtype=bfloat16.txt"),
]

r2_cot_wd_scan = [  # wd=0.1
    ("wd=0.4", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.4_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("wd=0.2", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.2_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("wd=0.1", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("wd=0.05", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.05_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
]

r2_cot_beta2_scan = [  # beta2=0.99
    ("beta2=0.999", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.999wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("beta2=0.995", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.995wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("beta2=0.99", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("beta2=0.98", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.98wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
]

r2_cot_lr_scan = [  # to 1e-6 and 5e-6
    ("lr=0.01..5e-6", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-5e-06b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("lr=0.01..1e-6", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-06b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("lr=0.01..5e-5", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-5e-05b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("lr=0.01..1e-5", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-05b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("lr=0.01..5e-4", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0005b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_seed=123_dtype=float.txt"),
    ("lr=0.01..1e-4", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
]

r2_corrective_cot_wd_scan = [  # wd=0.1
    ("wd=0.4", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.4_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("wd=0.2", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.2_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("wd=0.1", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("wd=0.05", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.05_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
]

r2_corrective_cot_beta2_scan = [  # beta2=0.99
    ("beta2=0.999", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.999wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("beta2=0.995", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.995wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("beta2=0.99", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("beta2=0.98", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.98wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
]

r2_corrective_cot_lr_scan = [  # 0.0001
    ("lr=0.01..1e-6", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-06b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("lr=0.01..1e-5", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-1e-05b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("lr=0.01..5e-5", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-5e-05b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("lr=0.01..1e-4", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("lr=0.01..5e-4", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0005b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("lr=0.01..0.001", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.001b20.99wd0.1_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
]

r2_corrective_cot_bs_scan = [  # >= 500
    ("bs=1000", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.1_dataset=train_lp_downsampled,d7,bs1000,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
    ("bs=500", "part_2/lp_adamw_model=d256_h4_l8_dataset=train_lp_downsampled,d7,r2_solver=corrective_cot,eager_path_wd=0.1_dropout=0.0_seed=123_dtype=float.txt"),
    ("bs=250", "part_3/lp_adamw_model=d256_h4_l8_optimizer=lr0.01-0.0001b20.99wd0.1_dataset=train_lp_downsampled,d7,bs250,r2_solver=corrective_cot,eager_path_epochs=15_seed=123_dtype=float.txt"),
]

standalone_runs = [direct_runs, r2_direct_runs, r2_direct_variations, cot_random_runs, cot_eager_runs,
                   r2_cot_random_runs, r2_cot_eager_runs, r2_cot_variations, r2_cot_ablation]
for runs in standalone_runs:
    print(generate_accuracy_table(runs, predicate_logs_path), "\n\n\n")

corrective_runs = [r2_corrective_cot_wd_scan, r2_corrective_cot_beta2_scan, r2_corrective_cot_lr_scan, r2_corrective_cot_bs_scan]
for runs in corrective_runs:
    new_runs = []
    for filter_f, evaluation in [(lambda x: ">" not in x, "direct"), (lambda x: ">" in x, "cot")]:
        for n, p in runs:
            new_runs.append((evaluation + " " + n, p, filter_f))

    # plot_combined_loss_curves(runs, predicate_logs_path, 60, figsize=(5, 4), x_axis="epoch")
    # plot_combined_accuracy_curves(runs, predicate_logs_path, 60, figsize=(5, 4), x_axis="epoch")
    print(generate_accuracy_table(new_runs, predicate_logs_path), "\n\n\n")
