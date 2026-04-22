import regex as re
import json


def parse_depth_data(log_text):
    regex = r"\bcorrect by depth:\s*(\{[\d\s:,]+\})"

    matches = re.findall(regex, log_text)
    results = []

    if not matches:
        return results

    for match in matches:
        string_key_match = re.sub(r'(\d+):', r'"\1":', match)
        depth_data = json.loads(string_key_match)
        results.append(depth_data)

    return results


def parse_epoch(log_text):
    regex = r"EPOCH\s+(?P<epoch>\d+).*?len\(train\)=(?P<len_train>\d+).*?\nLOSS train\s+(?P<train_loss>\d+\.\d+).*?\nLOSS validation\s+(?P<val_loss>\d+\.\d+)"

    match = re.search(regex, log_text, re.DOTALL)

    if match:
        return {
            "epoch": int(match.group("epoch")),
            "len_train": int(match.group("len_train")),
            "train_loss": float(match.group("train_loss")),
            "val_loss": float(match.group("val_loss")),
        }
    else:
        return None


def get_last_epoch_accuracy(filename, path, filter_f=lambda x: True, average=False, evals=3):
    with open(path + filename, 'r', encoding='utf-8') as file:
        contents = file.read().strip().split("\n\n")

    eval_results = [{} for x in range(evals)]

    for content_block in contents:
        if filter_f(content_block):
            results_data = parse_depth_data(content_block)
            if results_data and len(results_data) >= evals:
                for eval in range(evals):
                    results = results_data[eval]
                    last_epoch = {int(d): acc / 1000.0 for d, acc in results.items()}
                    eval_results[eval] = last_epoch

    if average:
        avg = lambda x: round(100 * sum(x.values()) / len(x.values()), 1)
        return (avg(x) for x in eval_results)
    return eval_results
