import random
from conf import *
from training.train_loop import *
from model.type_llama_no_ffn import Transformer, ModelArgs
from dataset import processor
from dataset.data_preprocessing import pad_collate
from training.adamw import AdamW
from dataset.eval import eval_model


def loss_fn(logits, labels):
    return F.cross_entropy(logits.transpose(1, 2), labels)


def get_logits(llm, batch):
    input_ids = batch["input_ids"].to("cuda")
    type_embeddings = batch["type_embeddings"].to("cuda")
    return llm.forward_train(input_ids=input_ids, type_embeddings=type_embeddings)


def get_loss(llm, batch, loss_fn):
    return loss_fn(get_logits(llm, batch), batch["labels"].to("cuda"))

depths = [0]

seed = 123
epochs = 10
eval_freq = 10
beta2 = 0.99

# initial best guesses
dim = 128
n_heads = 4
dropout = 0.0
wd = 0.1

# grid search ranges
lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
batches = [16, 32, 64, 128, 256, 512, 1024]
dims = [32, 64, 128, 256, 512]
heads = [1, 2, 4, 8, 16, 32]
wds = [0.0, 0.05, 0.1, 0.2, 0.4]
dropouts = [0.0, 0.05, 0.1, 0.2, 0.4]
beta2s = [0.95, 0.98, 0.99, 0.995, 0.999]

# grid search results (lr x batch_size)
batch_size = 512
lr = 0.01

# grid search results (dim x n_heads)
dim = 256
n_heads = 8

# for regularization hyperparameters, increase depth
depths = [0, 1]



# prepare dataset for grid search
data_dir = EXPERIMENTS_DIR + "/datasets/predicate_logic/"
train_data = processor.load(data_dir + "train/700k/prop_examples_lp.txt", depths=depths)
validation_data = processor.load(data_dir + "validation/prop_examples_lp.txt", depths=depths)
print(f"len(train)={len(train_data)}, len(validation)={len(validation_data)}")
inference_ids_lp = processor.process(processor.load(data_dir + "validation/prop_examples_lp.txt", depths=depths))
inference_ids_lp_star = processor.process(processor.load(data_dir + "validation/prop_examples_lp_star.txt", depths=depths))
inference_ids_rp = processor.process(processor.load(data_dir + "validation/prop_examples_rp.txt", depths=depths))


# do grid search

# for lr in lrs:
#     for batch_size in batches:
#         print(f"\nbatch_size: {batch_size}, learning_rate: {lr}")

# for dim in dims:
#     for n_heads in heads:
#         if dim // n_heads <= 2:
#             continue
#         print(f"\ndim: {dim}, n_heads: {n_heads}")

for lr in lrs:
    for beta2 in beta2s:
        print(f"\nlr: {lr}, beta2: {beta2}")

# for wd in wds:
#     for dropout in dropouts:
#         print(f"\nwd: {wd}, dropout: {dropout}")

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        args = ModelArgs(
            dim=dim,
            n_layers=2,
            n_heads=n_heads,
            vocab_size=256,
            ffn_dim_multiplier=1,
            max_seq_len=1024,
            dropout_rate=dropout,
            rope_theta=10000,
            use_scaled_rope=True
        )
        llm_model = load_weights_and_init(args, Transformer)
        for name, parameter in llm_model.named_parameters():
            parameter.requires_grad = True
            parameter.data = parameter.data.float().to("cuda")


        def ds_loader(ds, epoch, is_train):
            return DataLoader(ds, shuffle=True, batch_size=batch_size, collate_fn=lambda x: pad_collate(x, padding=processor.pad), pin_memory=True, num_workers=10, prefetch_factor=10)


        def preprocess_data(ds):
            items = 102_400 // (1024 // batch_size)  # scaled
            curriculum = processor.train_curriculum(ds, depths=depths, select_items=items)
            data = processor.process(curriculum, max_length=1024, do_shuffle=False)
            return data


        def eval_f(model, epoch, log=lambda x: print(x, end="")):
            if epoch % eval_freq != eval_freq - 1:
                return

            model.eval()
            with torch.no_grad():
                vocabulary, answer_position = processor.special_tokens, -1
                call = lambda xs: get_logits(model, xs)
                eval_model(call, DataLoader(inference_ids_lp, collate_fn=lambda x: pad_collate(x, padding=processor.pad)), vocabulary=vocabulary, answer_position=answer_position, log=log)
                eval_model(call, DataLoader(inference_ids_lp_star, collate_fn=lambda x: pad_collate(x, padding=processor.pad)), vocabulary=vocabulary, answer_position=answer_position, log=log)
                eval_model(call, DataLoader(inference_ids_rp, collate_fn=lambda x: pad_collate(x, padding=processor.pad)), vocabulary=vocabulary, answer_position=answer_position, log=log)
                log("\n")


        optimizer = AdamW(llm_model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, beta2))
        train_conf = TrainConf(epochs=epochs, eval_modulo=eval_freq, optimizer=optimizer, loss_fn=loss_fn, ds_loader=ds_loader, eval_model=eval_f)
        train(train_conf, llm_model, preprocess_data(train_data), preprocess_data(validation_data), calculate_loss=get_loss)
