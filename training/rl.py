import math
import torch
import torch.nn as nn

from torch.functional import F

from dataset.processor import special_tokens, pad


def get_log_probs(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_of_tokens = torch.gather(log_probs, 2, tokens.unsqueeze(2)).squeeze(2)
    return log_probs_of_tokens


def verify_and_reward(generated_sequences, batch, config):
    total_rewards = []
    n_samples = config['n_samples']
    sparse_reward = config['sparse_reward']

    for i, seq_tensor in enumerate(generated_sequences):
        ground_truth = batch['extra_data'][i // n_samples]

        seq_list = [token.item() for token in seq_tensor if token.item() != pad]
        if not seq_list:
            total_rewards.append(0)
            continue

        generated_label_1 = seq_list[-1]
        generated_path = seq_list[:-1]
        terminal_reward_1 = 1.0 if generated_label_1 == ground_truth['label'] else 0

        eval_reward = terminal_reward_1
        if not sparse_reward:
            seen_correct_steps = []
            for step in generated_path:
                is_correct = step in ground_truth['path']
                is_new = step not in seen_correct_steps
                if is_correct and is_new:
                    eval_reward += 0.1
                    seen_correct_steps.append(step)
                else:
                    eval_reward -= 0.1
        total_rewards.append(eval_reward)

    return torch.tensor(total_rewards, device=generated_sequences.device, dtype=torch.float)


def get_grpo_loss(llm, llm_ref, batch, config, backward=True):
    """ E[ min(ratio * A, clip * A) - beta * KL ] """
    device = next(llm.parameters()).device
    input_ids = batch["input_ids"].to(device)
    type_indices = batch["type_embeddings"].to(device)
    batch_size = input_ids.size(0)
    prompt_len = input_ids.size(1)

    # Step 1: Generate samples and get their log_probs directly from the generation process.
    with torch.no_grad():
        expanded_input_ids = input_ids.repeat_interleave(config['n_samples'], dim=0)
        expanded_type_indices = type_indices.repeat_interleave(config['n_samples'], dim=0)
        generated_sequences, old_log_probs = llm.generate(
            input_ids=expanded_input_ids,
            stop_tokens=special_tokens.values(),
            type_embeddings=expanded_type_indices,
            max_new_tokens=config['max_gen_len'],
            top_p=0.95,
            return_log_probs=True
        )
        old_log_probs = old_log_probs.sum(dim=-1)  # (batch_size * n_samples, gen_len) -> (batch_size * n_samples,)

    # Step 2: Calculate rewards for the generated samples
    newly_generated_tokens = generated_sequences[:, prompt_len:]
    rewards = verify_and_reward(newly_generated_tokens, batch, config)
    rewards = rewards.view(batch_size, config['n_samples'])  # (batch_size * n_samples,) -> (batch_size, n_samples)

    # Step 3: Calculate advantages
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    advantages = advantages.view(-1)  # (batch_size, n_samples) -> (batch_size * n_samples,)

    # --- Steps 4 & 5: Process in micro-batches to save memory ---
    total_loss = 0.0
    micro_batch_size = config["batch_size"]
    generated_type_ids = llm.generated_type_indices.to(input_ids.device).expand(expanded_type_indices.size(0), generated_sequences.size(1) - expanded_type_indices.size(1), -1)
    full_type_indices = torch.cat([expanded_type_indices, generated_type_ids], dim=1)
    total_samples = generated_sequences.size(0)
    num_minibatches = math.ceil(total_samples / micro_batch_size)

    # Retrieve clipping thresholds (clip-higher strategy)
    eps_low = config['epsilon']
    eps_high = config.get('epsilon_high', config['epsilon'])

    for i in range(0, total_samples, micro_batch_size):
        start_idx, end_idx = i, min(i + micro_batch_size, total_samples)

        micro_input_sequences = generated_sequences[start_idx:end_idx].clone()
        micro_type_indices = full_type_indices[start_idx:end_idx]
        micro_advantages = advantages[start_idx:end_idx]
        micro_old_log_probs = old_log_probs[start_idx:end_idx].detach()

        # Step 4: Re-evaluate sequences for the micro-batch (with gradients)
        policy_logits = llm.forward_train(input_ids=micro_input_sequences, type_embeddings=micro_type_indices)
        # Get reference logits from the frozen reference model (no gradients needed)
        with torch.no_grad():
            ref_logits = llm_ref.forward_train(input_ids=micro_input_sequences, type_embeddings=micro_type_indices)

        # Isolate the generated part of the sequences for loss calculation
        generated_logits = policy_logits[:, prompt_len - 1:-1, :]
        ref_generated_logits = ref_logits[:, prompt_len - 1:-1, :]
        generated_tokens = micro_input_sequences[:, prompt_len:]

        # Get log probabilities for both policy and reference models
        policy_log_probs = get_log_probs(generated_logits, generated_tokens)
        ref_log_probs = get_log_probs(ref_generated_logits, generated_tokens)

        # Step 5: Calculate GRPO loss for the micro-batch
        gen_mask = (generated_tokens != pad).float()
        ratio = torch.exp((policy_log_probs * gen_mask).sum(dim=-1) - micro_old_log_probs)
        log_ratio_ref = (policy_log_probs - ref_log_probs) * gen_mask  # This is equivalent to E[log(π_policy / π_ref)]
        approx_kl = (torch.exp(log_ratio_ref) - 1) - log_ratio_ref  # Unbiased estimator (k3) to ensure positivity
        if config['token_level']:
            gen_lengths = gen_mask.sum(dim=1).clamp(min=1)
            surr1 = ratio * micro_advantages / gen_lengths
            surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * micro_advantages / gen_lengths
            kl_divergence_loss = (approx_kl.sum(dim=-1) / gen_lengths).mean()
        else:
            surr1 = ratio * micro_advantages
            surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * micro_advantages
            kl_divergence_loss = approx_kl.sum(dim=-1).mean()

        # This is the surrogate objective loss from the GRPO/PPO policy update
        policy_surrogate_loss = -torch.min(surr1, surr2).mean()

        # Combine the policy loss and the KL divergence penalty
        micro_loss = policy_surrogate_loss + config['kl_coeff'] * kl_divergence_loss

        # This ensures the accumulated gradient is the mean, not the sum
        micro_loss /= num_minibatches

        if backward:
            micro_loss.backward()
        total_loss += micro_loss.item()

    # Return only the scalar value for logging
    return torch.tensor(total_loss, device='cpu')


class PartitionZ(nn.Module):
    """A simple 3-layer MLP to act as the learnable partition function Z_phi."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_flowrl_loss(llm, llm_ref, partition_Z, batch, config, backward=True):
    """log Z_φ(x) + log π_θ(y|x)  ≈  β * r(x,y) + log π_ref(y|x)"""
    device = next(llm.parameters()).device
    input_ids = batch["input_ids"].to(device)
    type_indices = batch["type_embeddings"].to(device)
    batch_size = input_ids.size(0)
    prompt_len = input_ids.size(1)

    # Step 1: Generate samples from the old policy (llm's state at generation time)
    with torch.no_grad():
        expanded_input_ids = input_ids.repeat_interleave(config['n_samples'], dim=0)
        expanded_type_indices = type_indices.repeat_interleave(config['n_samples'], dim=0)
        generated_sequences, old_log_probs = llm.generate(
            input_ids=expanded_input_ids,
            stop_tokens=special_tokens.values(),
            type_embeddings=expanded_type_indices,
            max_new_tokens=config['max_gen_len'],
            top_p=0.95,
            return_log_probs=True
        )
        old_log_probs = old_log_probs.sum(dim=-1)  # (batch_size * n_samples, gen_len) -> (batch_size * n_samples,)

    # Step 2: Calculate rewards for the generated samples
    newly_generated_tokens = generated_sequences[:, prompt_len:]
    rewards = verify_and_reward(newly_generated_tokens, batch, config)
    rewards = rewards.view(batch_size, config['n_samples'])  # (batch_size * n_samples,) -> (batch_size, n_samples)

    # Step 3: Calculate group-normalized rewards
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    norm_rewards = (rewards - mean_rewards) / (std_rewards + 1e-8)
    norm_rewards = norm_rewards.view(-1)  # (batch_size, n_samples) -> (batch_size * n_samples,)

    # --- Process in micro-batches to save memory ---
    total_loss = 0.0
    micro_batch_size = config["batch_size"]
    generated_type_ids = llm.generated_type_indices.to(input_ids.device).expand(expanded_type_indices.size(0), generated_sequences.size(1) - expanded_type_indices.size(1), -1)
    full_type_indices = torch.cat([expanded_type_indices, generated_type_ids], dim=1)
    total_samples = generated_sequences.size(0)
    num_minibatches = math.ceil(total_samples / micro_batch_size)

    # Retrieve clipping thresholds (clip-higher strategy)
    eps_low = config['epsilon']
    eps_high = config.get('epsilon_high', config['epsilon'])

    for i in range(0, total_samples, micro_batch_size):
        start_idx, end_idx = i, min(i + micro_batch_size, total_samples)

        # --- Prepare micro-batch data ---
        micro_input_sequences = generated_sequences[start_idx:end_idx].clone()
        micro_type_indices = full_type_indices[start_idx:end_idx]
        micro_norm_rewards = norm_rewards[start_idx:end_idx]
        micro_old_log_probs = old_log_probs[start_idx:end_idx].detach()
        micro_expanded_input_ids = expanded_input_ids[start_idx:end_idx]
        micro_expanded_type_indices = expanded_type_indices[start_idx:end_idx]

        # --- Calculate FlowRL loss components for the micro-batch ---

        # Get log Z_phi(x) from the partition function
        prompt_hidden_states = llm.forward_train(input_ids=micro_expanded_input_ids, type_embeddings=micro_expanded_type_indices, output_hidden_states=True)
        prompt_mask = (micro_expanded_input_ids != pad).float().unsqueeze(-1)
        summed_states = (prompt_hidden_states[:, -1] * prompt_mask).sum(dim=1)  # (micro_batch_size, layers, prompt_len, model_dim) -> (micro_batch_size, model_dim)
        valid_token_counts = prompt_mask.sum(dim=1).clamp(min=1)
        prompt_embedding = summed_states / valid_token_counts
        log_z = partition_Z(prompt_embedding.detach()).squeeze(-1)  # (micro_batch_size, 1) -> (micro_batch_size,)

        # Get log probabilities of generated tokens for policy and reference models
        policy_logits = llm.forward_train(input_ids=micro_input_sequences, type_embeddings=micro_type_indices)
        with torch.no_grad():
            ref_logits = llm_ref.forward_train(input_ids=micro_input_sequences, type_embeddings=micro_type_indices)

        generated_logits = policy_logits[:, prompt_len - 1:-1, :]
        ref_generated_logits = ref_logits[:, prompt_len - 1:-1, :]
        generated_tokens = micro_input_sequences[:, prompt_len:]

        # (micro_batch_size, gen_len, vocab_size) -> (micro_batch_size, gen_len)
        policy_log_probs = get_log_probs(generated_logits, generated_tokens)
        ref_log_probs = get_log_probs(ref_generated_logits, generated_tokens)

        gen_mask = (generated_tokens != pad).float()
        if config['token_level']:
            # Apply length normalization, (micro_batch_size,)
            gen_lengths = gen_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero

            # Maximize token-level reward over sequence-level reward -> prefer shorter proofs
            norm_log_z = log_z / gen_lengths
            norm_policy_log_probs = (policy_log_probs * gen_mask).sum(dim=-1) / gen_lengths
            norm_ref_log_probs = (ref_log_probs * gen_mask).sum(dim=-1) / gen_lengths
            norm_micro_norm_rewards = config['beta'] * micro_norm_rewards / gen_lengths

            # Calculate trajectory balance, (micro_batch_size,)
            trajectory_balance_term = norm_log_z + norm_policy_log_probs - norm_micro_norm_rewards - norm_ref_log_probs
        else:
            # Calculate sequence-level trajectory balance, (micro_batch_size,)
            trajectory_balance_term = log_z + (policy_log_probs * gen_mask).sum(dim=-1) - config['beta'] * micro_norm_rewards - (ref_log_probs * gen_mask).sum(dim=-1)

        # Final loss is the importance-weighted squared error
        ratio = torch.exp((policy_log_probs * gen_mask).sum(dim=-1) - micro_old_log_probs)
        w = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high).detach()
        micro_loss = (w * trajectory_balance_term.pow(2)).mean()  # (micro_batch_size,) -> scalar

        # This ensures the accumulated gradient is the mean, not the sum
        micro_loss /= num_minibatches

        if backward:
            micro_loss.backward()
        total_loss += micro_loss.item()

    return torch.tensor(total_loss, device='cpu')
