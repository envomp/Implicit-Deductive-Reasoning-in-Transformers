import pprint


def calculate_sample_distribution(epoch_samples: int, current_depth: int, max_depth: int) -> dict:
    """ Calculates a sample distribution that sums to a fixed total, balanced across all active depths. """
    samples_by_depth = {}

    num_active_depths = current_depth + 1
    samples_per_depth = epoch_samples // num_active_depths

    for depth in range(num_active_depths):
        samples_by_depth[depth] = samples_per_depth

    for depth in range(current_depth + 1, max_depth + 1):
        samples_by_depth[depth] = 0

    remainder = epoch_samples % num_active_depths
    for i in range(remainder):
        samples_by_depth[i] += 1

    return samples_by_depth


if __name__ == '__main__':
    current_depth = 1
    mock_accuracies = {1: 0.95, 2: 0.82, 3: 0.85, 4: 0.91}

    for epoch in range(10):
        distribution = calculate_sample_distribution(700_000, current_depth, 6)

        print(f"--- Epoch: {epoch + 1} | Current Depth: {current_depth} ---")
        print("Sample Distribution:")
        pprint.pprint(distribution)
        print(f"Total Samples: {sum(distribution.values()):,}")
        val_accuracy = mock_accuracies.get(epoch, 0.90)
        print(f"Validation Accuracy for Depth {current_depth}: {val_accuracy:.2f}\n")

        if val_accuracy >= 0.9:
            if current_depth < 6:
                current_depth += 1

    from torch.optim.lr_scheduler import CosineAnnealingLR
    from training.adamw import AdamW
    from torch.nn.parameter import Parameter

    scheduler = CosineAnnealingLR(AdamW([Parameter()], lr=0.01), T_max=9, eta_min=1e-4)
    for i in range(10):
        print(f"epoch {i} lr {scheduler.get_last_lr()}")
        scheduler.step()
