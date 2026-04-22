import os
from datasets import load_dataset
from benchmark.hf_models import inference
from benchmark.prompt_formats import construct_prompt


def extract_conclusion(text: str):
    if "Conclusion" in text:
        parts = text.rsplit("Conclusion", 1)
        if len(parts) > 1:
            text = parts[-1]

    index_provable = text.find("provable")
    index_unprovable = text.find("unprovable")

    if index_provable != -1 and index_unprovable != -1:
        if index_unprovable < index_provable:
            return "unprovable"
        else:
            return "provable"

    elif index_provable != -1:
        return "provable"
    elif index_unprovable != -1:
        return "unprovable"
    else:
        return None


def reduce_dataset(dataset, items=100):
    group_counts = {}
    reduced_list = []

    total_processed = 0
    for item in dataset:
        total_processed += 1
        depth = item["depth"]
        label = item["label"]
        group_key = (depth, label)
        current_count = group_counts.get(group_key, 0)
        if current_count < items:
            reduced_list.append(item)
            group_counts[group_key] = current_count + 1

    print(f"Processed {total_processed} items from original dataset.")
    print(f"Returning reduced dataset with {len(reduced_list)} items.")
    print(f"Group distribution in reduced set: {group_counts}")
    return reduced_list


def run_baselines(llm, model_str, datasets, prompt_formats, preprocess_fn, decode_fn):
    print(f"--- 🚀 Starting baseline run for {model_str} ---")
    result_dir = f"baselines/{model_str}"
    os.makedirs(result_dir, exist_ok=True)

    llm.eval()
    for dataset_name in datasets:
        eval_dataset = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")[dataset_name]
        reduced_dataset = reduce_dataset(eval_dataset)

        for is_cot in prompt_formats:
            run_filename = f"results_{dataset_name}_is_cot={is_cot}.txt"
            full_file_path = os.path.join(result_dir, run_filename)
            results = {"correct": {i: 0 for i in range(0, 13)},
                       "incorrect": {i: 0 for i in range(0, 13)},
                       "dnf": {i: 0 for i in range(0, 13)}}
            start_index = 0
            file_mode = 'w'
            if os.path.exists(full_file_path):
                with open(full_file_path, 'r', encoding='utf-8') as f_read:
                    lines = f_read.readlines()
                if any("summary=" in line for line in lines):
                    print(f"Skipping {run_filename} (already complete).")
                    continue

                print(f"Resuming {run_filename}...")
                file_mode = 'a'

                for line in lines:
                    parts = line.strip().split('|')
                    if len(parts) >= 4 and "depth" in parts[1]:
                        try:
                            idx = int(parts[0].strip())
                            depth = int(parts[1].split("'")[1])
                            expected = parts[2].split("'")[1]
                            got_val = parts[3].split("'")[1]

                            if got_val == "None":
                                results["dnf"][depth] += 1
                            elif got_val == expected:
                                results["correct"][depth] += 1
                            else:
                                results["incorrect"][depth] += 1

                            start_index = idx + 1
                        except (IndexError, ValueError):
                            continue

            with open(full_file_path, file_mode, encoding='utf-8', buffering=1) as f_out:
                if file_mode == 'w':
                    f_out.write(f"experiment run details:\n")
                    f_out.write(f"  dataset: {dataset_name}\n")
                    f_out.write(f"  is_cot: {is_cot}\n\n")
                    f_out.write(f"---------------------------------------\n")

                for i, item in enumerate(reduced_dataset):
                    if i < start_index:
                        continue

                    full_prompt = construct_prompt(item, is_cot)
                    depth = item["depth"]
                    label = item["label"]
                    inputs, inference_kwargs = preprocess_fn(full_prompt)
                    generated = inference(model=llm, inputs=inputs, max_tokens=10000 if is_cot else 10, kwargs=inference_kwargs)
                    decoded = decode_fn(generated, inputs)
                    got = extract_conclusion(decoded)
                    expected = ("provable" if label else "unprovable")
                    if got is None:
                        results["dnf"][depth] += 1
                    elif got == expected:
                        results["correct"][depth] += 1
                    else:
                        results["incorrect"][depth] += 1
                    f_out.write(f"{i} | depth:'{depth}' | expected:'{expected}' | got='{got}'\n")

                f_out.write(f"---------------------------------------\n")
                f_out.write(f"summary={results}\n")
                f_out.write(f"---------------------------------------\n")
