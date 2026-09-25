from conf import *

from datasets import load_dataset
from dataset.processor import *
from training.train_loop import *
from model.type_llama_no_ffn import Transformer, ModelArgs
from model_repository import ModelRepository
from missed_vs_hallucination import *
from decision_process import *
from attention_maps import *
from linearity import *
from linear_probing_for_provability import *
from weights import *
from axioms import *
from type_relations import *

full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
args = lambda x: ModelArgs(vocab_size=256, pad_token_id=pad, generated_type_indices=[0, 1], **x)
path = EXPERIMENTS_DIR + f"/models/type_llama_no_ffn/"
repo = ModelRepository(base_path=path, model_class=Transformer, loader_func=load_weights_and_init, args_builder=args)

##### monkey patching llm to get embeddings #####

initial_embeddings = []
hidden_states = []
attention_maps = []


def log_forward_constructor(self, ini=False, attn=False, hidden=False):
    def log_forward(x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor | bool] = None,
                    start_pos: int = None, active_indices: Optional[torch.Tensor] = None):
        if not initial_embeddings and ini:
            initial_embeddings.append(x.detach().cpu())
        attn_res, map = self.attention(self.attention_norm(x), freqs_cis, mask, start_pos, active_indices, return_attn_map=True)
        if attn:
            attention_maps.append(map.detach().cpu())
        h = x + attn_res
        if self.ffn_enabled:
            ffn_res = self.feed_forward(self.ffn_norm(h))
            h = h + ffn_res
        if hidden:
            hidden_states.append(h.detach().cpu())
        return h

    return log_forward

# # Appendix: Breakdown of reasoning errors
# for evaluation in [{"rp", "corrective", "r2", "ffn"}, {"rp", "corrective", "r2", "ffn", "bidir"}, {"rp", "corrective", "ffn", "bidir"}, {"rp", "corrective", "ffn"}]:
#     validation_ds = full_dataset_dict["validation_lp_balanced"]
#     ds = prepare_ds_inference(process(train_curriculum(validation_ds)), is_cot=True)
#     entry = repo.get_entry(*evaluation)
#     hallucinated, missed = inspect_eval_missed_and_hallucinations(entry.model.cuda().eval(), ds)
#
#     eval_name = "_".join(sorted(evaluation))
#     filename = f"reasoning_errors_{eval_name}.pdf"
#     visualize_hallucinations(hallucinated, missed, y_lim=40, x_lim=60, filename=filename)

# # Appendix: Tracing the decision process - images
# select_by = lambda x: {y: x for y in range(7)}
# target_depth = [0, 1, 2, 3, 4, 5, 6]
# calibration_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["train_rp_balanced_downsampled"], eager=False, select_by_depth=select_by(1_000))), is_cot=False)
# for eval_ds, evaluations, depths in [
#     (["validation_rp_balanced_1_premise", "validation_rp_balanced", "validation_rp_balanced_3_premise"],
#      [{"rp", "corrective", "r2", "bidir"}], target_depth),
# ]:
#     for evaluation in evaluations:
#         for depth in depths:
#             for eval_d in eval_ds:
#                 ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict[eval_d], select_by_depth={depth: 1_000})), is_cot=False)
#                 entry = repo.get_entry(*evaluation)
#                 for block in entry.model.layers:
#                     block.old_forward = block.forward
#                     block.forward = log_forward_constructor(block, ini=True, hidden=True)
#                 model = entry.model.cuda().eval()
#                 manifold_changes = compute_procrustes_and_similarity(model, calibration_ds, initial_embeddings, hidden_states, target_layer_idx=-1)
#                 cor_probs, cor_logits, incor_logits, accuracy = inspect_eval_decision_process(entry.model.cuda().eval(), ds, initial_embeddings, hidden_states, manifold_changes=manifold_changes)
#                 eval_name = "_".join(sorted(evaluation))
#                 filename = f"decision_trace_{eval_d}_{eval_name}_depth={depth}.pdf"
#                 visualize_decision_process(cor_probs, cor_logits, incor_logits, accuracy, filename=filename)
#
#                 for block in entry.model.layers:
#                     block.forward = block.old_forward


# # tracing the decision process - table (not in paper)
# select_by = lambda x: {y: x for y in range(7)}
# target_depth = [6]
# calibration_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["train_rp_balanced_downsampled"], eager=False, select_by_depth=select_by(1_000))), is_cot=False)
# eval_datasets = ["validation_rp_balanced", "validation_rp_balanced_3_premise", "validation_rp_balanced_2_premise", "validation_rp_balanced_1_premise", "validation_rp_balanced_1_2_premise", "validation_rp_balanced_2_3_premise"]
# folder_configs = [
#     ("baseline_r2_bidir_corrective_123", {"rp", "corrective", "r2", "bidir"}),
#     ("baseline_r2_bidir_corrective_124", {"rp", "corrective", "r2", "bidir"}),
#     ("baseline_r2_bidir_corrective_125", {"rp", "corrective", "r2", "bidir"}),
# ]
# table_results = {}
# for eval_d in eval_datasets:
#     for depth in target_depth:
#         ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict[eval_d], select_by_depth={depth: 1_000})), is_cot=False)
#         for folder, tags in folder_configs:
#             models_by_epoch = repo.load_all_checkpoints(folder, tags)
#             eval_name = "_".join(sorted(tags))
#             for epoch, model in sorted(models_by_epoch.items(), reverse=True):
#                 print(f"eval: {folder} e={epoch}")
#                 for block in model.layers:
#                     block.old_forward = block.forward
#                     block.forward = log_forward_constructor(block, ini=True, hidden=True)
#
#                 model = model.cuda().eval()
#                 manifold_changes = compute_procrustes_and_similarity(model, calibration_ds, initial_embeddings, hidden_states, target_layer_idx=-1)
#                 cor_probs, cor_logits, incor_logits, accuracy = inspect_eval_decision_process(model, ds, initial_embeddings, hidden_states, manifold_changes=manifold_changes)
#                 layer_accs = get_layerwise_accuracies(cor_logits, incor_logits)
#                 table_results[(folder, epoch, eval_d, depth)] = layer_accs
#
#                 for block in model.layers:
#                     block.forward = block.old_forward

# table_results = {('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 59.6, 55.9, 61.4, 59.8, 79.9, 95.3], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 57.6, 54.4, 61.3, 58.2, 80.1, 92.3], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced', 6): [50.0, 50.0, 51.2, 71.2, 58.0, 67.6, 70.3, 80.3, 93.5], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced', 6): [50.2, 50.0, 51.4, 62.6, 57.8, 65.2, 68.0, 69.7, 91.5], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 62.5, 59.7, 66.5, 70.6, 69.3, 89.3], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced', 6): [49.8, 52.5, 56.4, 71.7, 65.4, 68.6, 68.9, 69.6, 85.7], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced', 6): [51.4, 52.7, 57.3, 78.1, 69.7, 69.2, 68.6, 73.8, 87.0], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced', 6): [50.3, 57.1, 57.5, 72.9, 64.6, 66.9, 72.5, 77.1, 87.5], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced', 6): [50.5, 50.2, 50.9, 70.7, 69.0, 67.1, 68.1, 57.7, 81.4], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced', 6): [48.8, 54.7, 72.8, 78.8, 71.9, 70.8, 76.2, 74.9, 87.5], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced', 6): [50.8, 74.1, 75.3, 77.1, 78.7, 71.9, 69.0, 69.8, 82.2], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced', 6): [50.7, 66.4, 75.0, 72.8, 71.6, 74.3, 74.7, 72.5, 70.2], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced', 6): [50.4, 67.2, 73.4, 73.0, 73.5, 73.3, 73.1, 72.9, 72.9], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced', 6): [48.3, 50.0, 50.0, 50.0, 50.0, 49.9, 49.9, 49.9, 49.8], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced', 6): [50.3, 50.0, 50.2, 50.2, 50.5, 50.3, 50.4, 50.0, 50.4], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.0, 51.6, 53.8, 64.9, 65.4, 95.4], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.0, 52.0, 54.5, 62.8, 60.5, 94.1], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced', 6): [50.0, 50.1, 50.1, 50.9, 60.7, 60.2, 70.6, 73.1, 94.6], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.0, 61.4, 59.5, 64.3, 64.7, 94.2], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.0, 51.9, 52.2, 53.8, 62.5, 93.0], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced', 6): [50.0, 50.0, 49.9, 51.0, 61.9, 53.0, 52.9, 56.6, 92.4], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced', 6): [50.0, 50.0, 50.1, 50.3, 57.4, 57.0, 53.1, 57.0, 90.6], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.3, 66.2, 68.9, 72.1, 73.6, 91.9], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced', 6): [50.3, 50.0, 49.8, 52.6, 58.9, 53.8, 55.2, 60.2, 89.3], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced', 6): [50.5, 50.0, 53.2, 56.9, 63.1, 55.1, 55.8, 56.8, 84.8], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced', 6): [50.1, 50.2, 51.0, 54.8, 65.5, 56.3, 62.2, 60.2, 85.6], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced', 6): [51.0, 50.1, 52.1, 61.7, 62.5, 57.6, 66.0, 63.8, 84.9], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced', 6): [49.7, 50.0, 49.3, 50.6, 57.2, 53.1, 68.0, 60.7, 77.3], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced', 6): [50.3, 51.2, 56.9, 53.0, 55.0, 56.5, 74.1, 80.4, 76.0], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced', 6): [49.6, 48.0, 49.0, 49.8, 49.8, 48.8, 53.4, 52.0, 51.5], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.1, 50.0, 50.0, 50.7, 73.9, 93.6], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.2, 70.9, 93.0], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced', 6): [50.0, 50.0, 50.1, 50.1, 50.1, 50.0, 52.0, 67.3, 90.9], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced', 6): [50.3, 50.4, 56.4, 54.1, 51.5, 50.7, 53.2, 74.1, 92.7], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced', 6): [50.0, 50.3, 52.0, 54.1, 52.2, 50.8, 53.7, 68.1, 89.7], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced', 6): [49.5, 60.4, 58.2, 55.7, 51.7, 50.7, 61.8, 84.9, 90.8], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced', 6): [50.3, 56.2, 55.0, 51.1, 51.5, 50.9, 60.5, 72.3, 86.3], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced', 6): [50.6, 57.8, 57.3, 54.1, 52.0, 51.2, 62.5, 63.4, 77.2], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced', 6): [51.1, 56.1, 53.8, 52.0, 50.2, 50.1, 53.5, 71.0, 82.5], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced', 6): [48.4, 59.2, 52.8, 56.4, 55.1, 50.1, 52.1, 58.3, 74.9], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced', 6): [49.2, 50.2, 50.0, 50.3, 51.7, 52.6, 58.9, 60.6, 76.4], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced', 6): [50.4, 50.1, 50.0, 50.0, 50.0, 50.0, 50.9, 57.5, 77.9], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced', 6): [50.0, 49.9, 49.9, 49.9, 50.2, 50.7, 52.4, 49.7, 73.6], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced', 6): [50.0, 51.2, 51.3, 52.8, 58.4, 49.4, 49.7, 49.9, 61.4], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced', 6): [49.5, 50.3, 51.2, 49.6, 51.1, 49.9, 49.4, 48.4, 52.9], ('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.4, 52.8, 55.3, 52.7, 65.7, 82.4], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.3, 51.8, 55.3, 51.7, 63.4, 77.7], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 56.5, 54.8, 61.6, 58.3, 64.0, 83.6], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 49.9, 52.4, 53.3, 56.1, 55.6, 55.1, 73.0], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced_3_premise', 6): [50.2, 50.0, 50.0, 55.5, 54.7, 60.8, 61.4, 56.1, 74.5], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced_3_premise', 6): [49.1, 52.4, 51.7, 61.5, 58.4, 58.9, 57.8, 52.1, 62.6], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced_3_premise', 6): [49.8, 53.5, 55.1, 70.2, 67.9, 65.5, 62.0, 57.4, 70.3], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced_3_premise', 6): [50.3, 58.7, 53.1, 66.7, 59.5, 55.9, 57.9, 57.4, 70.8], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced_3_premise', 6): [50.0, 50.1, 52.3, 61.4, 65.1, 56.4, 52.1, 50.4, 55.7], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced_3_premise', 6): [48.9, 53.6, 69.1, 69.3, 68.1, 60.7, 60.9, 56.2, 68.1], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced_3_premise', 6): [49.6, 70.4, 72.9, 73.7, 74.4, 60.0, 55.0, 55.8, 63.7], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced_3_premise', 6): [51.2, 69.9, 72.6, 67.1, 66.2, 64.3, 60.9, 57.4, 52.1], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced_3_premise', 6): [49.6, 56.8, 71.7, 71.0, 69.4, 68.3, 67.9, 67.3, 59.1], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced_3_premise', 6): [52.4, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 49.8], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced_3_premise', 6): [50.4, 50.1, 50.1, 50.2, 50.4, 50.4, 50.5, 48.5, 49.7], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 49.9, 50.0, 50.1, 50.8, 55.3, 53.5, 77.9], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.2, 52.9, 52.5, 76.2], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced_3_premise', 6): [50.0, 50.1, 51.4, 50.0, 50.7, 51.6, 59.4, 57.7, 80.1], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.1, 50.0, 50.9, 52.4, 54.5, 54.2, 78.2], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.1, 50.6, 52.2, 79.3], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 51.7, 50.0, 54.7, 51.2, 51.0, 52.5, 81.8], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 49.9, 50.0, 50.7, 50.9, 50.3, 50.6, 77.9], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.1, 50.1, 60.6, 65.3, 67.7, 63.2, 80.7], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced_3_premise', 6): [51.3, 50.0, 49.9, 52.5, 60.3, 55.5, 52.8, 56.7, 71.1], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced_3_premise', 6): [50.6, 50.0, 54.6, 58.3, 69.0, 54.7, 54.8, 53.3, 68.2], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced_3_premise', 6): [50.2, 50.1, 54.0, 56.9, 69.7, 55.6, 56.5, 54.3, 64.6], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced_3_premise', 6): [50.9, 50.0, 54.2, 63.9, 64.7, 57.7, 62.6, 57.5, 68.1], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced_3_premise', 6): [50.6, 50.0, 49.7, 50.1, 66.6, 54.6, 59.4, 51.5, 54.7], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced_3_premise', 6): [50.1, 50.9, 54.3, 51.6, 56.9, 52.7, 66.8, 68.3, 56.2], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced_3_premise', 6): [49.7, 47.6, 51.3, 50.0, 51.1, 50.6, 51.1, 50.5, 49.8], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 54.3, 81.9], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 53.6, 78.0], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.1, 51.9, 70.6], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced_3_premise', 6): [50.3, 50.2, 50.7, 50.1, 50.0, 50.0, 50.1, 54.5, 80.7], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced_3_premise', 6): [50.0, 50.1, 50.0, 50.0, 50.0, 50.0, 50.3, 52.2, 73.9], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced_3_premise', 6): [50.9, 57.9, 50.9, 50.4, 50.1, 50.0, 52.7, 58.7, 73.6], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced_3_premise', 6): [50.1, 55.1, 51.5, 50.0, 50.2, 50.1, 51.7, 53.1, 65.6], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced_3_premise', 6): [49.9, 55.3, 50.0, 50.0, 50.6, 50.5, 52.2, 52.6, 60.1], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced_3_premise', 6): [51.2, 54.6, 50.0, 50.0, 51.8, 51.7, 52.1, 57.2, 65.0], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced_3_premise', 6): [50.4, 58.1, 50.2, 51.0, 51.1, 50.1, 50.1, 51.6, 57.5], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced_3_premise', 6): [49.8, 50.2, 50.0, 50.0, 52.1, 51.5, 51.9, 52.9, 62.0], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced_3_premise', 6): [50.9, 50.1, 50.0, 50.0, 50.1, 50.2, 49.8, 50.3, 61.5], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.9, 51.1, 51.2, 51.7, 57.4], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced_3_premise', 6): [50.0, 51.9, 51.1, 51.7, 60.5, 49.8, 50.2, 50.0, 52.8], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced_3_premise', 6): [50.4, 53.0, 53.5, 51.6, 52.9, 51.8, 51.0, 50.2, 51.0], ('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.3, 65.1, 58.3, 64.5, 59.5, 79.4, 93.7], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.3, 62.7, 55.4, 62.9, 58.1, 81.6, 90.8], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 52.6, 75.2, 60.8, 72.7, 72.9, 83.4, 92.5], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced_2_premise', 6): [50.1, 50.0, 51.9, 64.9, 58.4, 64.7, 67.5, 70.3, 90.6], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced_2_premise', 6): [49.7, 50.0, 50.5, 68.2, 62.8, 69.8, 72.1, 70.8, 88.2], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced_2_premise', 6): [51.0, 53.7, 58.6, 77.4, 67.1, 73.3, 72.3, 69.1, 83.2], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced_2_premise', 6): [48.1, 52.6, 60.5, 80.0, 74.4, 74.7, 73.8, 76.9, 88.9], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced_2_premise', 6): [52.6, 58.6, 58.9, 75.8, 68.3, 70.7, 75.3, 77.6, 85.6], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced_2_premise', 6): [50.1, 50.0, 50.9, 72.3, 70.5, 67.0, 66.2, 54.9, 75.7], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced_2_premise', 6): [49.6, 54.8, 74.1, 81.8, 76.1, 72.5, 76.5, 74.8, 86.0], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced_2_premise', 6): [51.4, 74.2, 76.7, 79.6, 80.2, 70.6, 67.1, 68.7, 78.9], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced_2_premise', 6): [47.7, 69.7, 77.4, 72.1, 72.0, 75.3, 74.9, 73.8, 69.0], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced_2_premise', 6): [50.9, 66.0, 75.6, 75.2, 75.5, 74.9, 75.0, 74.7, 72.3], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced_2_premise', 6): [49.5, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.7], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced_2_premise', 6): [50.5, 50.1, 50.1, 50.1, 50.0, 50.0, 49.9, 46.6, 46.8], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.0, 53.1, 55.9, 62.7, 65.9, 93.9], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.0, 53.5, 56.1, 63.2, 61.6, 92.5], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.4, 51.3, 63.5, 61.2, 69.7, 72.2, 93.0], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.3, 65.7, 62.0, 65.5, 65.3, 89.5], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.0, 53.2, 51.8, 53.3, 63.1, 91.2], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 51.1, 63.9, 53.3, 53.2, 57.1, 91.3], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.1, 50.1, 55.9, 53.9, 51.9, 55.1, 90.6], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.5, 70.7, 71.9, 71.6, 73.1, 90.3], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced_2_premise', 6): [49.8, 50.0, 50.5, 52.6, 60.6, 53.6, 54.8, 61.9, 88.9], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced_2_premise', 6): [49.5, 50.0, 56.2, 61.7, 68.6, 56.5, 59.5, 59.5, 85.0], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced_2_premise', 6): [51.8, 50.0, 52.2, 56.0, 67.5, 56.6, 63.6, 59.5, 83.4], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced_2_premise', 6): [51.8, 50.1, 52.9, 64.3, 64.0, 59.6, 66.3, 64.5, 81.0], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 51.2, 59.8, 57.0, 68.5, 59.4, 70.9], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced_2_premise', 6): [50.3, 52.4, 59.9, 54.9, 56.5, 58.8, 76.6, 79.9, 69.7], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced_2_premise', 6): [50.0, 49.4, 51.2, 51.1, 50.9, 51.3, 52.7, 51.8, 49.9], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.1, 50.0, 50.0, 50.0, 51.1, 70.4, 92.3], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.1, 50.0, 50.1, 50.0, 50.7, 66.1, 91.2], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.1, 50.0, 50.0, 50.0, 52.5, 62.7, 86.5], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced_2_premise', 6): [50.4, 50.4, 55.7, 54.3, 50.6, 50.4, 55.0, 71.7, 90.4], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced_2_premise', 6): [50.0, 50.2, 52.2, 53.8, 52.6, 51.2, 54.9, 62.9, 87.4], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced_2_premise', 6): [48.8, 59.8, 58.9, 55.9, 51.5, 50.8, 64.4, 78.1, 89.6], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced_2_premise', 6): [50.0, 55.4, 54.8, 51.6, 51.1, 50.1, 61.6, 69.7, 82.5], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced_2_premise', 6): [50.8, 56.7, 57.2, 53.8, 51.5, 50.2, 60.9, 61.8, 75.0], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced_2_premise', 6): [50.2, 56.9, 52.2, 50.7, 50.1, 50.2, 57.6, 68.4, 77.5], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced_2_premise', 6): [50.9, 60.4, 51.9, 55.5, 52.4, 50.2, 50.8, 55.8, 71.6], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced_2_premise', 6): [49.8, 50.3, 49.9, 50.3, 50.7, 55.2, 59.1, 60.7, 74.4], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced_2_premise', 6): [49.8, 50.0, 50.0, 50.0, 50.5, 50.1, 51.6, 55.8, 73.0], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced_2_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.2, 51.1, 52.9, 50.7, 71.9], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced_2_premise', 6): [50.0, 51.7, 51.9, 53.3, 60.6, 49.9, 50.9, 50.0, 60.1], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced_2_premise', 6): [50.5, 52.6, 53.5, 52.1, 51.0, 51.6, 49.6, 51.4, 53.9], ('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 57.0, 88.8, 53.9, 69.3, 79.2, 91.8, 94.7], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 58.0, 85.3, 52.1, 64.9, 74.6, 88.6, 91.4], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 56.9, 90.0, 56.4, 76.2, 88.0, 92.9, 96.0], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 62.2, 82.8, 57.8, 69.8, 80.8, 81.0, 92.3], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced_1_premise', 6): [49.9, 50.0, 52.9, 82.7, 64.9, 71.4, 78.1, 76.2, 90.2], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced_1_premise', 6): [51.3, 51.5, 56.5, 90.3, 71.2, 81.9, 87.2, 86.7, 93.1], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced_1_premise', 6): [54.2, 50.7, 63.6, 91.6, 80.4, 81.2, 85.4, 87.3, 92.0], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced_1_premise', 6): [51.9, 55.3, 56.3, 88.4, 77.3, 80.0, 91.2, 88.5, 91.3], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced_1_premise', 6): [50.4, 50.0, 50.1, 83.5, 76.3, 77.6, 88.6, 80.7, 88.9], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced_1_premise', 6): [50.5, 52.8, 74.3, 88.3, 81.8, 83.0, 91.0, 90.3, 95.0], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced_1_premise', 6): [52.0, 74.7, 84.3, 90.0, 91.1, 89.1, 88.9, 87.0, 92.0], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced_1_premise', 6): [52.9, 62.0, 52.6, 88.2, 88.3, 87.2, 86.7, 86.9, 85.5], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced_1_premise', 6): [50.5, 78.3, 85.7, 85.6, 85.4, 85.4, 85.4, 85.1, 85.1], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced_1_premise', 6): [49.6, 50.1, 50.1, 50.1, 50.1, 50.2, 50.2, 50.2, 51.1], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced_1_premise', 6): [49.7, 50.0, 50.0, 49.9, 49.9, 50.0, 49.8, 52.5, 50.6], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 50.7, 53.6, 57.7, 65.5, 67.7, 96.5], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 51.0, 55.6, 68.7, 68.8, 66.4, 94.0], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced_1_premise', 6): [50.0, 50.2, 50.0, 71.5, 61.3, 76.7, 75.3, 75.9, 94.7], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 56.5, 76.1, 77.3, 65.6, 71.8, 95.5], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 52.3, 73.4, 64.8, 57.9, 78.5, 96.5], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 49.9, 69.1, 66.4, 59.7, 55.6, 66.4, 95.4], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 65.3, 63.8, 72.2, 59.3, 74.2, 95.7], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced_1_premise', 6): [50.0, 50.1, 50.0, 53.9, 65.5, 84.9, 81.5, 84.2, 96.2], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced_1_premise', 6): [49.8, 50.0, 52.2, 68.7, 76.6, 60.0, 66.8, 75.2, 95.7], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced_1_premise', 6): [50.1, 49.9, 66.8, 76.1, 80.5, 76.6, 82.0, 73.3, 94.0], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced_1_premise', 6): [51.2, 50.3, 54.2, 63.3, 71.3, 69.5, 79.5, 81.8, 95.9], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced_1_premise', 6): [50.2, 50.1, 49.7, 63.3, 61.4, 77.6, 85.7, 85.4, 94.3], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced_1_premise', 6): [50.1, 50.1, 48.0, 53.6, 58.6, 66.0, 87.0, 88.0, 93.7], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced_1_premise', 6): [50.2, 53.8, 69.5, 65.5, 58.3, 81.1, 91.7, 93.4, 90.1], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced_1_premise', 6): [50.3, 49.1, 49.8, 49.5, 49.7, 49.6, 52.4, 52.4, 52.6], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 58.5, 61.3, 57.6, 50.0, 66.1, 93.0, 92.5], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 63.7, 62.6, 58.9, 50.1, 62.1, 91.6, 94.8], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 57.1, 55.9, 63.0, 52.0, 70.1, 86.8, 87.6], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced_1_premise', 6): [50.3, 51.2, 67.6, 66.3, 70.7, 62.2, 73.1, 89.8, 93.9], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced_1_premise', 6): [50.0, 50.2, 62.6, 66.1, 69.8, 55.2, 67.0, 80.7, 94.2], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced_1_premise', 6): [49.2, 62.5, 70.7, 58.5, 62.7, 62.5, 73.6, 93.3, 97.1], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced_1_premise', 6): [49.6, 58.2, 63.5, 56.6, 71.7, 81.5, 80.8, 93.1, 90.7], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced_1_premise', 6): [50.1, 62.2, 73.8, 73.3, 86.0, 85.1, 85.4, 88.9, 86.0], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced_1_premise', 6): [50.2, 58.1, 68.2, 64.6, 65.2, 72.9, 67.8, 80.9, 88.4], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced_1_premise', 6): [50.0, 60.7, 61.5, 65.6, 85.9, 82.2, 65.9, 71.6, 91.1], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced_1_premise', 6): [50.3, 50.1, 50.7, 52.7, 59.9, 68.2, 71.9, 71.0, 88.7], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced_1_premise', 6): [49.4, 50.0, 50.2, 50.6, 60.2, 53.6, 61.6, 77.4, 92.3], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced_1_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.1, 68.0, 53.2, 91.8], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced_1_premise', 6): [50.0, 51.2, 51.9, 54.7, 53.1, 50.0, 55.2, 50.3, 84.2], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced_1_premise', 6): [49.8, 54.4, 54.8, 53.5, 53.6, 52.5, 53.9, 53.8, 53.9], ('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 53.5, 76.8, 57.9, 67.4, 68.0, 87.8, 95.6], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 51.6, 74.9, 54.4, 66.0, 66.5, 86.0, 93.7], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 64.1, 83.3, 59.5, 74.5, 78.2, 87.9, 95.3], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced_1_2_premise', 6): [50.2, 50.0, 57.6, 75.7, 59.8, 68.6, 74.2, 77.8, 93.6], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced_1_2_premise', 6): [49.3, 50.0, 51.3, 75.7, 63.5, 71.4, 76.4, 73.6, 90.7], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced_1_2_premise', 6): [48.7, 52.0, 63.0, 83.0, 67.9, 75.4, 77.7, 79.0, 91.0], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced_1_2_premise', 6): [47.9, 51.6, 66.3, 87.7, 74.7, 76.2, 77.8, 84.1, 91.5], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced_1_2_premise', 6): [50.1, 59.0, 59.6, 80.8, 71.2, 74.9, 82.7, 83.4, 91.0], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced_1_2_premise', 6): [50.5, 50.0, 50.7, 78.0, 71.5, 72.1, 77.4, 66.4, 86.7], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced_1_2_premise', 6): [49.4, 54.1, 77.5, 86.0, 78.1, 78.5, 84.7, 84.8, 92.6], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced_1_2_premise', 6): [50.5, 77.5, 82.3, 85.5, 86.4, 82.7, 81.4, 81.2, 87.2], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced_1_2_premise', 6): [51.1, 65.8, 71.2, 79.8, 78.3, 82.4, 82.3, 82.2, 80.9], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced_1_2_premise', 6): [48.5, 75.5, 81.0, 81.1, 80.8, 80.6, 80.7, 80.5, 79.9], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced_1_2_premise', 6): [50.8, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced_1_2_premise', 6): [50.7, 49.8, 49.8, 49.9, 50.1, 49.8, 49.8, 49.5, 49.5], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.2, 53.3, 54.0, 64.7, 67.5, 96.6], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.2, 53.9, 57.5, 63.9, 66.0, 96.0], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.2, 50.2, 56.3, 65.8, 69.7, 72.4, 75.1, 96.2], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.7, 71.9, 66.5, 64.2, 69.0, 95.4], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.2, 61.6, 56.1, 55.5, 69.3, 94.0], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.1, 55.4, 69.9, 57.1, 54.1, 60.7, 94.2], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.1, 49.9, 51.9, 65.1, 62.8, 54.0, 61.7, 93.4], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.9, 71.2, 77.7, 76.1, 76.4, 93.7], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced_1_2_premise', 6): [49.5, 50.0, 50.1, 55.1, 65.5, 55.1, 58.7, 65.6, 93.3], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced_1_2_premise', 6): [50.5, 50.0, 59.3, 65.9, 71.5, 60.9, 65.5, 63.5, 90.7], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced_1_2_premise', 6): [48.4, 50.3, 53.4, 59.3, 70.0, 61.1, 68.6, 68.4, 90.7], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced_1_2_premise', 6): [49.3, 50.2, 46.6, 61.9, 60.3, 64.0, 72.4, 72.5, 90.0], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced_1_2_premise', 6): [49.4, 50.0, 49.0, 51.1, 57.5, 58.3, 74.8, 70.8, 87.1], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced_1_2_premise', 6): [50.4, 53.6, 63.6, 58.6, 58.7, 65.3, 84.5, 87.0, 81.4], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced_1_2_premise', 6): [50.4, 48.2, 49.0, 48.6, 48.7, 48.6, 52.2, 53.5, 51.8], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 51.0, 52.4, 50.6, 50.0, 54.8, 81.9, 94.8], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 53.1, 54.4, 50.8, 50.0, 53.5, 80.2, 95.5], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 51.8, 53.1, 52.2, 50.4, 58.3, 77.2, 90.5], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced_1_2_premise', 6): [49.9, 50.6, 62.8, 61.9, 57.7, 53.6, 61.8, 80.2, 94.0], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.5, 57.4, 61.6, 58.5, 52.9, 59.7, 75.3, 93.9], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced_1_2_premise', 6): [50.9, 62.6, 67.1, 61.4, 56.4, 53.8, 67.2, 87.7, 94.2], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced_1_2_premise', 6): [49.8, 58.3, 60.8, 54.6, 59.7, 58.4, 68.7, 81.9, 88.6], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced_1_2_premise', 6): [50.3, 60.6, 66.1, 63.5, 70.0, 64.2, 72.2, 77.3, 83.2], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced_1_2_premise', 6): [48.8, 60.2, 61.2, 57.7, 51.9, 52.5, 59.2, 75.0, 86.5], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced_1_2_premise', 6): [48.4, 61.1, 56.7, 62.2, 69.0, 52.3, 54.9, 63.9, 84.4], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced_1_2_premise', 6): [50.6, 50.3, 50.0, 50.8, 54.3, 59.4, 66.6, 66.5, 84.1], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced_1_2_premise', 6): [50.2, 50.0, 50.0, 50.1, 53.1, 50.3, 53.8, 64.5, 85.4], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced_1_2_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.3, 50.9, 58.2, 50.2, 84.3], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced_1_2_premise', 6): [50.0, 51.4, 51.6, 56.0, 57.6, 50.1, 51.5, 50.0, 74.9], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced_1_2_premise', 6): [51.4, 53.2, 53.2, 52.0, 51.7, 52.5, 52.9, 53.6, 52.3], ('baseline_r2_bidir_corrective_123', 14, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 53.3, 55.3, 58.6, 55.3, 71.9, 89.4], ('baseline_r2_bidir_corrective_123', 13, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 52.9, 53.9, 58.5, 54.6, 71.6, 85.0], ('baseline_r2_bidir_corrective_123', 12, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 65.5, 57.5, 67.5, 66.1, 73.6, 92.2], ('baseline_r2_bidir_corrective_123', 11, 'validation_rp_balanced_2_3_premise', 6): [49.9, 50.0, 50.3, 57.9, 56.4, 58.7, 62.0, 62.5, 83.7], ('baseline_r2_bidir_corrective_123', 10, 'validation_rp_balanced_2_3_premise', 6): [49.9, 50.0, 49.9, 61.1, 59.7, 66.4, 68.8, 61.2, 82.2], ('baseline_r2_bidir_corrective_123', 9, 'validation_rp_balanced_2_3_premise', 6): [49.6, 53.8, 56.3, 68.9, 63.9, 65.5, 64.9, 58.3, 75.9], ('baseline_r2_bidir_corrective_123', 8, 'validation_rp_balanced_2_3_premise', 6): [49.9, 53.9, 58.4, 76.2, 69.2, 69.9, 66.9, 66.7, 82.0], ('baseline_r2_bidir_corrective_123', 7, 'validation_rp_balanced_2_3_premise', 6): [48.8, 58.9, 57.2, 71.2, 61.6, 64.0, 67.4, 68.7, 81.5], ('baseline_r2_bidir_corrective_123', 6, 'validation_rp_balanced_2_3_premise', 6): [49.4, 50.1, 53.2, 67.5, 68.4, 62.9, 57.6, 51.3, 64.6], ('baseline_r2_bidir_corrective_123', 5, 'validation_rp_balanced_2_3_premise', 6): [50.7, 56.8, 72.2, 74.2, 69.8, 66.6, 68.4, 67.5, 80.5], ('baseline_r2_bidir_corrective_123', 4, 'validation_rp_balanced_2_3_premise', 6): [50.2, 73.1, 75.6, 76.0, 77.8, 67.4, 60.5, 63.3, 74.9], ('baseline_r2_bidir_corrective_123', 3, 'validation_rp_balanced_2_3_premise', 6): [51.8, 70.0, 75.8, 69.2, 70.0, 68.9, 70.4, 65.8, 60.9], ('baseline_r2_bidir_corrective_123', 2, 'validation_rp_balanced_2_3_premise', 6): [47.2, 60.8, 74.9, 73.8, 73.4, 73.1, 72.4, 72.2, 66.8], ('baseline_r2_bidir_corrective_123', 1, 'validation_rp_balanced_2_3_premise', 6): [49.7, 50.0, 50.0, 50.0, 50.0, 50.0, 50.1, 50.0, 51.7], ('baseline_r2_bidir_corrective_123', 0, 'validation_rp_balanced_2_3_premise', 6): [49.0, 50.0, 50.1, 50.0, 50.1, 50.1, 50.0, 48.2, 47.2], ('baseline_r2_bidir_corrective_124', 14, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 51.5, 53.6, 60.6, 59.7, 90.5], ('baseline_r2_bidir_corrective_124', 13, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 51.2, 52.8, 58.3, 56.8, 87.6], ('baseline_r2_bidir_corrective_124', 12, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.1, 50.7, 50.2, 56.5, 55.7, 65.2, 65.0, 88.2], ('baseline_r2_bidir_corrective_124', 11, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.1, 50.1, 56.1, 56.5, 61.9, 60.8, 86.8], ('baseline_r2_bidir_corrective_124', 10, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.6, 50.5, 51.3, 56.8, 87.9], ('baseline_r2_bidir_corrective_124', 9, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.5, 50.5, 57.5, 51.9, 50.9, 54.8, 87.2], ('baseline_r2_bidir_corrective_124', 8, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.1, 51.2, 51.3, 50.5, 52.6, 85.6], ('baseline_r2_bidir_corrective_124', 7, 'validation_rp_balanced_2_3_premise', 6): [50.0, 49.9, 50.0, 50.1, 62.5, 68.9, 68.9, 68.9, 87.6], ('baseline_r2_bidir_corrective_124', 6, 'validation_rp_balanced_2_3_premise', 6): [50.4, 50.0, 50.4, 52.5, 62.5, 54.3, 54.1, 58.0, 81.8], ('baseline_r2_bidir_corrective_124', 5, 'validation_rp_balanced_2_3_premise', 6): [49.8, 50.0, 56.2, 59.5, 67.9, 54.0, 56.3, 56.1, 78.1], ('baseline_r2_bidir_corrective_124', 4, 'validation_rp_balanced_2_3_premise', 6): [50.8, 50.1, 53.5, 59.4, 69.6, 55.8, 60.4, 56.6, 74.9], ('baseline_r2_bidir_corrective_124', 3, 'validation_rp_balanced_2_3_premise', 6): [46.8, 50.0, 51.9, 62.6, 63.5, 57.2, 64.0, 60.7, 75.0], ('baseline_r2_bidir_corrective_124', 2, 'validation_rp_balanced_2_3_premise', 6): [49.8, 50.0, 49.9, 50.9, 63.7, 55.8, 63.3, 55.3, 62.3], ('baseline_r2_bidir_corrective_124', 1, 'validation_rp_balanced_2_3_premise', 6): [49.3, 52.5, 56.8, 54.2, 58.2, 55.1, 73.9, 72.7, 63.7], ('baseline_r2_bidir_corrective_124', 0, 'validation_rp_balanced_2_3_premise', 6): [50.0, 47.4, 48.3, 48.4, 49.3, 48.5, 52.1, 50.2, 51.0], ('baseline_r2_bidir_corrective_125', 14, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.1, 61.9, 88.6], ('baseline_r2_bidir_corrective_125', 13, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 59.2, 86.4], ('baseline_r2_bidir_corrective_125', 12, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.1, 50.0, 50.0, 50.4, 56.6, 80.3], ('baseline_r2_bidir_corrective_125', 11, 'validation_rp_balanced_2_3_premise', 6): [49.9, 50.2, 52.4, 51.3, 50.1, 50.2, 51.4, 63.8, 86.6], ('baseline_r2_bidir_corrective_125', 10, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.1, 50.2, 50.9, 50.7, 50.1, 51.9, 58.6, 83.5], ('baseline_r2_bidir_corrective_125', 9, 'validation_rp_balanced_2_3_premise', 6): [49.5, 57.5, 54.5, 52.5, 50.7, 50.3, 58.5, 70.3, 84.9], ('baseline_r2_bidir_corrective_125', 8, 'validation_rp_balanced_2_3_premise', 6): [49.8, 53.9, 52.9, 50.2, 50.9, 50.0, 56.1, 59.9, 77.8], ('baseline_r2_bidir_corrective_125', 7, 'validation_rp_balanced_2_3_premise', 6): [50.7, 55.7, 52.0, 50.9, 50.2, 50.2, 57.6, 56.1, 67.8], ('baseline_r2_bidir_corrective_125', 6, 'validation_rp_balanced_2_3_premise', 6): [51.1, 55.4, 50.7, 49.9, 50.2, 50.2, 53.9, 62.1, 71.3], ('baseline_r2_bidir_corrective_125', 5, 'validation_rp_balanced_2_3_premise', 6): [51.0, 59.9, 50.9, 52.5, 51.4, 50.1, 50.4, 54.1, 63.1], ('baseline_r2_bidir_corrective_125', 4, 'validation_rp_balanced_2_3_premise', 6): [50.9, 50.0, 50.0, 50.0, 51.1, 53.6, 55.2, 54.6, 68.5], ('baseline_r2_bidir_corrective_125', 3, 'validation_rp_balanced_2_3_premise', 6): [49.5, 50.0, 50.0, 50.0, 50.1, 50.0, 50.7, 52.8, 68.9], ('baseline_r2_bidir_corrective_125', 2, 'validation_rp_balanced_2_3_premise', 6): [50.0, 50.0, 50.0, 50.0, 50.6, 51.7, 52.4, 52.3, 64.0], ('baseline_r2_bidir_corrective_125', 1, 'validation_rp_balanced_2_3_premise', 6): [50.0, 52.1, 51.9, 54.0, 60.9, 49.9, 50.4, 50.0, 54.0], ('baseline_r2_bidir_corrective_125', 0, 'validation_rp_balanced_2_3_premise', 6): [49.1, 51.5, 51.5, 52.7, 52.3, 51.7, 51.4, 49.9, 51.1]}
# print_results_table(table_results)

# # Appendix: Superposition hypothesis
# evaluations = [
#     {"rp", "corrective", "r2", "bidir", "small", "baseline", "seed", "seed124"},
#     {"rp", "corrective", "r2", "bidir", "small", "baseline", "seed", "seed125"},
#     {"rp", "corrective", "r2", "bidir", "small", "baseline"},
# ]
# select_by = lambda x: {y: x for y in range(7)}
# calibration_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["train_rp_balanced_downsampled"], eager=False, select_by_depth=select_by(1_000))), is_cot=False)
# validation_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["validation_rp_balanced"], select_by_depth=select_by(1_000))), is_cot=False)
# accumulated_type_similarity = defaultdict(list)
# accumulated_other_similarity = defaultdict(list)
# accumulated_aligned_type_similarity = defaultdict(list)
# accumulated_aligned_other_similarity = defaultdict(list)
#
# for evaluation in evaluations:
#     entry = repo.get_entry(*evaluation)
#     for block in entry.model.layers:
#         block.old_forward = block.forward
#         block.forward = log_forward_constructor(block, ini=True, hidden=True)
#
#     eval_name = "_".join(sorted(evaluation))
#     model = entry.model.cuda().eval()
#     manifold_changes = compute_procrustes_and_similarity(model, calibration_ds, initial_embeddings, hidden_states, target_layer_idx=0)
#     aligned_type_sim, aligned_other_sim = inspect_linearity(model, validation_ds, manifold_changes, initial_embeddings, hidden_states)
#     type_sim, other_sim = inspect_linearity(model, validation_ds, None, initial_embeddings, hidden_states)
#
#     for t_id, similarities in type_sim.items():
#         accumulated_type_similarity[t_id].append(similarities)
#     for t_id, similarities in other_sim.items():
#         accumulated_other_similarity[t_id].append(similarities)
#
#     for t_id, similarities in aligned_type_sim.items():
#         accumulated_aligned_type_similarity[t_id].append(similarities)
#     for t_id, similarities in aligned_other_sim.items():
#         accumulated_aligned_other_similarity[t_id].append(similarities)
#
#     for block in entry.model.layers:
#         block.forward = block.old_forward
#
# avg_type_sim = {t_id: np.mean(np.array(sims), axis=0) for t_id, sims in accumulated_type_similarity.items()}
# avg_other_sim = {t_id: np.mean(np.array(sims), axis=0) for t_id, sims in accumulated_other_similarity.items()}
# avg_uncurved_sim = {t_id: np.mean(np.array(sims), axis=0) for t_id, sims in accumulated_aligned_type_similarity.items()}
# avg_uncurved_other_sim = {t_id: np.mean(np.array(sims), axis=0) for t_id, sims in accumulated_aligned_other_similarity.items()}
# filename_type = f"type_similarity_{eval_name}.pdf"
# visualize_type_similarity(avg_type_sim, avg_other_sim, avg_uncurved_sim, avg_uncurved_other_sim, filename=filename_type)

# # Appendix: Superposition hypothesis - visualizing predicates
# evaluations = [
#     frozenset({"rp", "corrective", "r2", "bidir"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=16"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=32"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=64"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=128"})
# ]
# select_by = lambda x: {y: x for y in range(7)}
# calibration_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["train_rp_balanced_downsampled"], eager=False, select_by_depth=select_by(1_000))), is_cot=False)
# validation_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["validation_rp_balanced"], select_by_depth=select_by(1_000))), is_cot=False)
# target_tokens = [1, 10, 100]
# unaligned_tokens = defaultdict(list)
# aligned_tokens = defaultdict(list)
# for evaluation in evaluations:
#     entry = repo.get_entry(*evaluation)
#     for block in entry.model.layers:
#         block.old_forward = block.forward
#         block.forward = log_forward_constructor(block, ini=True, hidden=True)
#
#     eval_name = "_".join(sorted(evaluation))
#     model = entry.model.cuda().eval()
#     manifold_changes = compute_procrustes_and_similarity(model, calibration_ds, initial_embeddings, hidden_states, target_layer_idx=0)
#     unaligned_trajs = inspect_token_linearity(model, validation_ds, None, initial_embeddings, hidden_states, target_tokens)
#     aligned_trajs = inspect_token_linearity(model, validation_ds, manifold_changes, initial_embeddings, hidden_states, target_tokens)
#     for t_id in target_tokens:
#         unaligned_tokens[t_id].extend(unaligned_trajs[t_id])
#         aligned_tokens[t_id].extend(aligned_trajs[t_id])
#
#     for block in entry.model.layers:
#         block.forward = block.old_forward
#
# visualize_combined_trajectories(unaligned_tokens, aligned_tokens, target_tokens, filename=f"rp_token_similarity.pdf")

# # Identifying the successful approximation
# aggregated_results = {}
# for evaluation in [
#     frozenset({"rp", "corrective", "r2", "bidir"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=16"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=32"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=64"}),
#     frozenset({"rp", "corrective", "r2", "bidir", "scaling", "layers=128"})
# ]:
#     for eval_ds in ["validation_rp_balanced_deep_30_pred", "validation_rp_balanced_deep_60_pred", "validation_lp_balanced_deep_30_pred", "validation_lp_balanced_deep_60_pred"]:
#         model_name = " + ".join(evaluation)
#         select_by = lambda x: {y: x for y in range(7)}
#         calibration_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict["train_rp_balanced_downsampled"], eager=False, select_by_depth=select_by(1_000))), is_cot=False)
#         validation_ds = prepare_ds_inference(process(train_curriculum(full_dataset_dict[eval_ds], select_by_depth=select_by(1_000))), is_cot=False)
#         entry = repo.get_entry(*evaluation)
#         for block in entry.model.layers:
#             block.old_forward = block.forward
#             block.forward = log_forward_constructor(block, ini=True, hidden=True)
#         model = entry.model.cuda().eval()
#
#         for curve_manifold in [True, False]:
#             if curve_manifold:
#                 manifold_changes = compute_procrustes_and_similarity(model, calibration_ds, initial_embeddings, hidden_states, target_layer_idx=-1)
#                 linear_probe = ZeroShotDecisionProbe(model, manifold_changes)
#             else:
#                 linear_probe = ZeroShotDecisionProbe(model, None)
#             layer_metrics = extract_and_eval_probe(model, linear_probe, validation_ds, initial_embeddings, hidden_states)
#             aggregated_results[(model_name, eval_ds, curve_manifold)] = layer_metrics
#
#         for block in entry.model.layers:
#             block.forward = block.old_forward
#
# print(aggregated_results)
# plot_provability_cached()

# # Appendix: Justification for corrective
# for evaluation in [{"lp", "corrective", "r2", "ffn"}, {"lp", "cot", "direct", "r2", "ffn"}]:
#     entry = repo.get_entry(*evaluation)
#     eval_name = "_".join(sorted(evaluation))
#     filename = f"corrective_justification_{eval_name}.pdf"
#     plot_justification_for_corrective(entry.model, filename=filename)

# # Appendix: RL errors
# for evaluation in [{"rp", "sparse", "token", "flowrl", "r2", "ffn", "bidir", "cot"}, {"rp", "token", "sparse", "grpo", "r2", "ffn", "bidir", "cot"}]:
#     validation_ds = full_dataset_dict["validation_lp_balanced"]
#     ds = prepare_ds_inference(process(train_curriculum(validation_ds)), is_cot=True)
#     entry = repo.get_entry(*evaluation)
#     hallucinated, missed = inspect_eval_missed_and_hallucinations(entry.model.cuda().eval(), ds)
#     eval_name = "_".join(sorted(evaluation))
#     filename = f"reasoning_errors_{eval_name}.pdf"
#     visualize_hallucinations(hallucinated, missed, y_lim=40, x_lim=100, filename=filename)

# # Appendix: rotating subspaces
#
# for evaluation in [{"rp", "corrective", "r2", "bidir"}]:
#     entry = repo.get_entry(*evaluation)
#     eval_name = "_".join(sorted(evaluation))
#     plot_weight_distribution(entry.model, eval_name, "norm.w")


# not in article

# # magnitude
# for evaluation in [{"rp", "corrective", "r2", "bidir", "scaling", "layers=64"}]:
#     entry = repo.get_entry(*evaluation)
#     spike_df = get_spike_data(entry.model.type_embeddings, TYPE_NAME_MAP, threshold=3)
#     visualize_dimension_relations(spike_df)
#     sim_df = calculate_similarity_matrix(entry.model.type_embeddings, TYPE_NAME_MAP)
#     visualize_similarity_matrix(sim_df)


# # attention maps
# for evaluation in [{"rp", "corrective", "r2", "bidir"}]:
#     validation_ds = full_dataset_dict["validation_lp_balanced"]
#     ds = prepare_ds_inference(process(train_curriculum(validation_ds, select_by_depth={6: 1000})), is_cot=False)
#     entry = repo.get_entry(*evaluation)
#     for block in entry.model.layers:
#         block.old_forward = block.forward
#         block.forward = log_forward_constructor(block, attn=True)
#
#     for aggregated in [True]:
#         mean_attention_df, std_attention_df = inspect_eval_attention_maps(entry.model.cuda().eval(), ds, attention_maps, aggregated=aggregated)
#         eval_name = "_".join(sorted(evaluation))
#         agg_str = "aggregated" if aggregated else "layerwise"
#         filename = f"attention_maps_{eval_name}_{agg_str}.pdf"
#         visualize_aggregate_attention_maps(mean_attention_df, std_attention_df, aggregated=aggregated, filename=filename)
#
#     for block in entry.model.layers:
#         block.forward = block.old_forward
