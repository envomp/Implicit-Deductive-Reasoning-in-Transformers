import copy
import gzip
import pickle
import random
import itertools
from conf import *
from datasets import load_dataset
from model.type_llama_no_ffn import Transformer, ModelArgs
from training.train_loop import *
from training.adamw import AdamW
from training.rl import get_grpo_loss, get_flowrl_loss, PartitionZ
from training.curriculum import calculate_sample_distribution
from dataset.processor import process, train_curriculum, pad, prepare_ds_train, prepare_ds_inference, direct_answer, cot_answer, special_tokens
from dataset.data_preprocessing import pad_collate
from dataset.eval import eval_model
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_str = "float32"
seed = 123
processing_cores = 12
gradient_checkpointing = False
debug = False

dropout = 0.0
wd = 0.1
wd_on_lnorm = True
wd_on_2d = True
lr_start = 1e-2
scheduler_lr_end = 1e-4
adamw_beta2 = 0.99

epochs = 15
eval_freq = 1
batch_size = 500
gradient_accumulation = 1

train_distribution = "lp"
pickled_dataset = False
pickled_dataset_files = 15
heuristics_enabled = True
heuristic_placement = "append"
solver_include_copy = False
solver_eager = True

curriculum = False
curriculum_eager = True
curriculum_eval_threshold = 0.9

custom_data_shuffle = False
corrective_cot = False
cot = False
direct = False
bidirectional_mask = False

previous_model = ""
initial_eval = False
grpo_enabled = False
flowrl_enabled = False
rl_token_level = True
rl_sparse_reward = False
rl_n_samples = 8
rl_batch_size = 512
rl_epsilon_high = 0.2

dim = 256
vocab_size = 256
n_heads = 4
n_layers = 8
ffn_enabled = False
universal_transformer = False
lnorm_implementation = "RMSNorm"

exec(open('configurator.py').read())

# debug = True
# train_distribution = "rp"
# bidirectional_mask = True
# corrective_cot = True

# previous_model = "/media/e/data/experiments/models/type_llama_no_ffn/rp_epoch=14_ffn_bidir_r2_corrective"
# ffn_enabled = True
# initial_eval = True
# lr_start = 1e-6
# scheduler_lr_end = 1e-6
# cot = True
# flowrl_enabled = True
# rl_batch_size = 32

dtype = getattr(torch, dtype_str)

rl_enabled = grpo_enabled or flowrl_enabled
solver_break_early = not rl_enabled
ref_llm = None
depths = [0, 1, 2, 3, 4, 5, 6]
heuristics = ["r2"] if heuristics_enabled else []

grpo_config = {
    "sparse_reward": rl_sparse_reward,
    "token_level": rl_token_level,
    "n_samples": rl_n_samples,
    "epsilon": 0.2,
    "epsilon_high": rl_epsilon_high,
    "max_gen_len": 32,
    "batch_size": rl_batch_size,
    "kl_coeff": 0.04
}
flowrl_config = {
    "sparse_reward": rl_sparse_reward,
    "token_level": rl_token_level,
    "n_samples": rl_n_samples,
    "epsilon": 0.2,
    "epsilon_high": rl_epsilon_high,
    "max_gen_len": 32,
    "batch_size": rl_batch_size,
    "beta": 15,
}

if flowrl_enabled:
    flowrl_mlp = PartitionZ(dim).to(dtype=dtype).to(device=device)

full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
balanced_distribution = f"{train_distribution}_balanced" if train_distribution == "rp" else train_distribution
validation_ds = full_dataset_dict[f"validation_{balanced_distribution}"]
train_ds_name = f"train_{balanced_distribution}" if curriculum else f"train_{balanced_distribution}_downsampled"
train_ds = [] if pickled_dataset else full_dataset_dict[train_ds_name]
if debug:
    print("DEBUG: training on validation ds for brevity")
    train_ds = validation_ds
print(f"len(train)={len(train_ds)} len(validation)={len(validation_ds)}")
inference_ids_lp = process(train_curriculum(full_dataset_dict["validation_lp_balanced"]))
inference_ids_rp = process(train_curriculum(full_dataset_dict["validation_rp_balanced"]))


def ds_loader(ds, epoch, is_train):
    if is_train and pickled_dataset:
        assert not curriculum and not solver_include_copy and heuristics_enabled
        data_dir = EXPERIMENTS_DIR + f"/datasets/predicate_logic/train/700k/eager_bulk_{train_distribution}_v2/"
        with gzip.open(data_dir + f"epoch={epoch % pickled_dataset_files}.pickle.gzip", 'rb') as f:
            raw_ds = pickle.load(f)
        ds = prepare_ds_train(raw_ds, rl_frac=get_rl_frac(epoch), custom_data_shuffle=custom_data_shuffle, corrective_cot=corrective_cot, cot=cot, direct=direct)
    else:
        select_by_depth = calculate_sample_distribution(epoch_samples=len(depths) * 100_000, current_depth=current_depth, max_depth=len(depths) - 1) if curriculum else None
        print(f"epoch: {epoch} current_depth: {current_depth} select_by_depth: {select_by_depth}")
        ds = train_curriculum(ds, select_by_depth=select_by_depth, eager=curriculum_eager, heuristics=heuristics, heuristic_placement=heuristic_placement, balance=True, preserve_depth="RP" in train_distribution.upper())
        ds = process(ds, max_length=1024, solver_include_copy=solver_include_copy, solver_eager=solver_eager)
        ds = prepare_ds_train(ds, rl_frac=get_rl_frac(epoch), custom_data_shuffle=custom_data_shuffle, corrective_cot=corrective_cot, cot=cot, direct=direct)

    return DataLoader(ds, shuffle=not custom_data_shuffle and is_train, batch_size=batch_size // gradient_accumulation,
                      collate_fn=lambda x: pad_collate(x, padding=pad),
                      pin_memory=True, num_workers=10, prefetch_factor=10)


def get_rl_frac(epoch):
    return 1.0 if rl_enabled and ref_llm is not None else 0


def get_logits_and_states(llm, batch):
    input_ids = batch["input_ids"].to(device)
    type_embeddings = batch["type_embeddings"].to(device)
    hidden_states = llm.forward_train(input_ids=input_ids, type_embeddings=type_embeddings, output_hidden_states=True)
    logits = llm.output_proj(hidden_states[:, -1])
    return logits, hidden_states


def get_loss(llm, batch, backward=True):
    """
    Calculates and backpropagates all loss components.
    Returns a detached, scalar sum of the losses for logging purposes.
    """
    device = next(llm.parameters()).device
    total_log_loss = 0.0
    sample = lambda f: {
        key: value[f] if isinstance(value, torch.Tensor)
        else list(itertools.compress(value, f))
        for key, value in batch.items()
    }

    # --- Part 1: RL loss ---
    is_rl_sample = torch.tensor([d.get("rl", False) for d in batch['extra_data']], dtype=torch.bool)
    if is_rl_sample.any():
        rl_batch = sample(is_rl_sample)
        # These functions perform their own .backward() calls internally
        if grpo_enabled:
            grpo_log_loss = get_grpo_loss(llm, ref_llm, rl_batch, grpo_config, backward=backward)
            total_log_loss += grpo_log_loss.item()
        if flowrl_enabled:
            flowrl_log_loss = get_flowrl_loss(llm, ref_llm, flowrl_mlp, rl_batch, flowrl_config, backward=backward)
            total_log_loss += flowrl_log_loss.item()

    # --- Part 2: Cross-Entropy supervised loss ---
    is_t_sample = ~is_rl_sample
    if is_t_sample.any():
        t_batch = sample(is_t_sample)
        logits, hidden_states = get_logits_and_states(llm, t_batch)
        labels = t_batch["labels"].to(device)

        ce_loss = ce_loss_fn(logits.transpose(1, 2), labels)

        # Gradients will be accumulated with any from the RL part
        if backward:
            ce_loss.backward()
        total_log_loss += ce_loss.item()

    return torch.tensor(total_log_loss)


def run_eval(model, log=lambda x: print(x, end="")):
    if rl_enabled:
        global ref_llm
        ref_llm = copy.deepcopy(model).eval()

    model.eval()
    with torch.no_grad():
        collate_fn = lambda x: pad_collate(x, padding=pad)

        def run_eval_f(is_cot=False, answer_position=-1):
            lp_ds = DataLoader(prepare_ds_inference(inference_ids_lp, is_cot=is_cot), collate_fn=collate_fn)
            rp_ds = DataLoader(prepare_ds_inference(inference_ids_rp, is_cot=is_cot), collate_fn=collate_fn)
            e1 = eval_model(model, lp_ds, vocabulary=special_tokens, answer_position=answer_position, is_cot=is_cot, log=log)
            e3 = eval_model(model, rp_ds, vocabulary=special_tokens, answer_position=answer_position, is_cot=is_cot, log=log)
            log("\n")
            return e1, e3

        if corrective_cot or direct:
            e1, e3 = run_eval_f(is_cot=False)
        if corrective_cot or cot:
            log("cot>\n")
            e1, e3 = run_eval_f(is_cot=True)

    return (e1.get(current_depth, 0) + e3.get(current_depth, 0)) / 2000


def eval_f(model, epoch, log=lambda x: print(x, end="")):
    if epoch % eval_freq != eval_freq - 1:
        return
    acc = run_eval(model, log)
    checkpoint = {'epoch': epoch}

    if curriculum:
        global current_depth
        print(f"depth {current_depth} accuracy: {acc}  threshold: {curriculum_eval_threshold}")
        if acc > curriculum_eval_threshold:
            current_depth += 1
        checkpoint['depth'] = current_depth

    checkpoint['model'] = llm_model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler.state_dict()
    if flowrl_enabled:
        checkpoint['flowrl_mlp'] = flowrl_mlp.state_dict()

    os.makedirs(save_path, exist_ok=True)
    torch.save(checkpoint, save_path + f"epoch={epoch}")


def construct_run_id():
    dataset_flags = []
    dataset_flags.append(train_ds_name)
    dataset_flags.append(f"d{len(depths)}")
    dataset_flags.append(f"bs{batch_size}")
    if curriculum:
        dataset_flags.append("curriculum")
        if curriculum_eager:
            dataset_flags.append("eager")
        else:
            dataset_flags.append("random")
    if heuristics:
        dataset_flags.append("+".join(heuristics))
    if custom_data_shuffle:
        dataset_flags.append("custom_shuffle")

    solver_flags = []
    if corrective_cot:
        solver_flags.append("corrective")
    if cot:
        solver_flags.append("cot")
    if direct:
        solver_flags.append("direct")

    if solver_eager:
        solver_flags.append("eager")
    else:
        solver_flags.append("random")
    if solver_include_copy:
        solver_flags.append("copy")

    _model = f"_model=d{dim}_h{n_heads}_l{n_layers}"
    _ffn = f"_+ffn" if ffn_enabled else ""
    _lnorm = f"_+{lnorm_implementation}"
    _universal = f"_+universal" if universal_transformer else ""
    _bidirectional = f"_+bidirectional" if bidirectional_mask else ""
    _wd = (f"-lnorm" if not wd_on_lnorm else "") + (f"-2d" if not wd_on_2d else "")
    _optimizer = f"_optimizer=lr{lr_start}-{scheduler_lr_end}"
    _optimizer += f"b2{adamw_beta2}wd{wd}{_wd}"
    _dataset = f"_dataset={",".join(dataset_flags)}"
    _solver = f"_solver={",".join(solver_flags)}" if solver_flags else ""
    _rl = f"_grpo={grpo_enabled}_flowrl={flowrl_enabled}_T={rl_token_level}_S={rl_sparse_reward}_n={rl_n_samples}_b={rl_batch_size}" if rl_enabled else ""
    run_id = f"{train_distribution}_adamw{_model}{_ffn}{_lnorm}{_universal}{_bidirectional}{_optimizer}{_dataset}{_solver}{_rl}_epochs={epochs}_seed={seed}_dtype={dtype_str}"
    return run_id


if __name__ == '__main__':
    print("Fixing seed to: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    run_id = construct_run_id()
    print(run_id)

    args = ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size, ffn_enabled=ffn_enabled,
                     dropout_rate=dropout, gradient_checkpointing=gradient_checkpointing,
                     lnorm_implementation=lnorm_implementation, universal_transformer=universal_transformer,
                     generated_type_indices=[0, 1], pad_token_id=pad,
                     bidirectional_stop_tokens=[cot_answer, direct_answer] if bidirectional_mask else None,
                     isolated_subsequence_tokens=[direct_answer, cot_answer] if corrective_cot else None)
    print(args)
    save_path = EXPERIMENTS_DIR + f"/models/type_llama_no_ffn/" + run_id + f"/"
    latest_checkpoint_path, start_epoch = find_latest_epoch_file(save_path)
    llm_model = load_weights_and_init(args, Transformer, state_dict_location=previous_model).to(device)

    for n, p in llm_model.named_parameters():
        p.param_name = n
    large_inner_params = {'params': [p for n, p in llm_model.named_parameters() if p.dim() >= 2 and "layers" in n], 'weight_decay': wd if wd_on_2d else 0.0}
    large_outer_params = {'params': [p for n, p in llm_model.named_parameters() if p.dim() >= 2 and "layers" not in n], 'weight_decay': wd}
    small_params = {'params': [p for p in llm_model.parameters() if p.dim() < 2], 'weight_decay': wd if wd_on_lnorm else 0.0}
    adamw_params = [large_outer_params, small_params]
    if flowrl_enabled:
        adamw_params.append({'params': flowrl_mlp.parameters(), 'weight_decay': wd})
    adamw_params.append(large_inner_params)
    print(f"AdamW(lr={lr_start}, wd={wd}, betas=(0.9, {adamw_beta2}))")
    optimizer = AdamW(adamw_params, lr=lr_start, betas=(0.9, adamw_beta2))
    print(f"CosineAnnealingLR(T_max={epochs - 1}, eta_min={scheduler_lr_end})")
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 1, eta_min=scheduler_lr_end)

    current_depth = 1 if curriculum else len(depths) - 1
    if latest_checkpoint_path:
        print(f"resuming training from: {latest_checkpoint_path} epoch: {start_epoch}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
        llm_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if flowrl_enabled:
            flowrl_mlp.load_state_dict(checkpoint['flowrl_mlp'])
        if curriculum:
            current_depth = checkpoint['depth']

    if previous_model and initial_eval:
        if debug:
            print("DEBUG: skipping initial eval")
            ref_llm = copy.deepcopy(llm_model).eval()
        else:
            print(f"initial acc: {run_eval(llm_model)}")

    print(print_number_of_trainable_model_parameters(llm_model))
    for name, parameter in llm_model.named_parameters():
        parameter.requires_grad = True
        parameter.data = parameter.data.to(dtype=dtype).to(device=device)
        print(f"{name} (device={parameter.device}, requires_grad={parameter.requires_grad}, dtype={parameter.dtype})")
    if flowrl_enabled:
        for name, parameter in flowrl_mlp.named_parameters():
            print(f"{name} (device={parameter.device}, requires_grad={parameter.requires_grad}, dtype={parameter.dtype})")

    train_conf = TrainConf(epochs=epochs, start_epoch=start_epoch,
                           optimizer=optimizer, scheduler=scheduler,
                           ds_loader=ds_loader, eval_model=eval_f, max_grad_norm=3.0 * gradient_accumulation)

    full_file_path = os.path.join(EXPERIMENTS_DIR + "/predicate_runs/", run_id + ".txt")
    with open(full_file_path, 'a', encoding='utf-8', buffering=1) as f_out:
        train(train_conf, llm_model, train_ds, validation_ds, calculate_loss=get_loss, log=f_out.write)
