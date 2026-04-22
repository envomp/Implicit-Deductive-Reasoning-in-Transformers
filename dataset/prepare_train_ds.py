import gzip
import pickle
from conf import *
from datasets import load_dataset
from dataset.processor import process, train_curriculum

seed = 123
processing_cores = 12
start_epoch = 0
epochs = 25
distribution = "rp"
preserve_depth = True
heuristic_placement = "append"

exec(open('configurator.py').read())

depths = [0, 1, 2, 3, 4, 5, 6]
heuristics = ["r2"]

if ("RP" in distribution.upper()) ^ preserve_depth:
    print("warning: probably a config mismatch. preserve_depth was meant for RP only")

full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning").filter(lambda x: x['depth'] in depths, num_proc=processing_cores)
train_ds_name = f"train_{distribution + ("_balanced" if preserve_depth else "")}_downsampled"
train_ds = full_dataset_dict[train_ds_name]

for epoch in range(start_epoch, epochs):
    print(f"epoch: {epoch}")
    curriculum_ds = train_curriculum(train_ds, heuristics=heuristics, heuristic_placement=heuristic_placement, balance=True, preserve_depth=preserve_depth)

    ds = process(curriculum_ds, max_length=1024)

    samples_by_depth = {}
    for elem in ds:
        if elem["depth"] not in samples_by_depth:
            samples_by_depth[elem["depth"]] = 0
        samples_by_depth[elem["depth"]] += 1
    print(f"len={len(ds)}. distribution={samples_by_depth}")

    data_dir = f"{EXPERIMENTS_DIR}/datasets/predicate_logic/train/700k/eager_bulk_{distribution}_v2/"
    os.makedirs(data_dir, exist_ok=True)
    with gzip.open(data_dir + f"epoch={epoch}.pickle.gzip", 'wb') as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
