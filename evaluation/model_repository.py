import os
import re
from dataclasses import dataclass
from typing import List, Set, Any

from dataset.processor import direct_answer, cot_answer


@dataclass
class ModelEntry:
    name: str
    pretty_name: str
    tags: Set[str]
    model: Any


class ModelRepository:
    def __init__(self, base_path, model_class, loader_func, args_builder):
        """
        base_path: str -> Directory containing model checkpoints
        model_class: class -> The Transformer class
        loader_func: func -> function(args, model_class, path)
        args_builder: func -> function(dict) -> returns args object
        """
        self.entries: List[ModelEntry] = []
        self.base_path = base_path
        self.model_class = model_class
        self.loader_func = loader_func
        self.args_builder = args_builder
        self._populate()

    def load_model(self, tags, f_name):
        is_ffn = "ffn" in tags
        is_bidir = "bidir" in tags
        is_universal = "universal" in tags

        dim = 256
        n_layers = 8
        for tag in tags:
            if tag.startswith("dim="):
                dim = int(tag.split("=")[1])
            elif tag.startswith("layers="):
                n_layers = int(tag.split("=")[1])

        model_args = self.args_builder({"ffn_enabled": is_ffn, "dim": dim, "n_layers": n_layers, "n_heads": 4,
                                        "bidirectional_stop_tokens": [cot_answer, direct_answer] if is_bidir else None,
                                        "universal_transformer": is_universal})
        print(f"Loading: {f_name} | tags: {tags} | ffn: {is_ffn} | dim: {dim}, L: {n_layers} | universal: {is_universal}")
        return self.loader_func(model_args, self.model_class, os.path.join(self.base_path, f_name))

    def _populate(self):
        files = sorted([f for f in os.listdir(self.base_path) if "epoch=" in f])

        for f_name in files:
            parts = f_name.split('_')
            tags = set(parts)
            tags = {t for t in tags if "epoch=" not in t}
            pretty_name = " ".join(tags)
            model = self.load_model(tags, f_name)
            self.entries.append(ModelEntry(name=f_name, pretty_name=pretty_name, tags=tags, model=model))

    def get_entry(self, *query_tags):
        query_set = set(query_tags)
        matches = [e for e in self.entries if query_set.issubset(e.tags) and e.tags.issubset(query_set)]

        if not matches:
            raise ValueError(f"No model found for tags: {query_set}")
        if len(matches) > 1:
            names = [m.name for m in matches]
            raise ValueError(f"Ambiguous query {query_set}. Matches found: {names}")
        print(f"Loaded model: {matches[0].pretty_name}")
        return matches[0]

    def load_all_checkpoints(self, folder, tags):
        folder_path = os.path.join(self.base_path, folder)
        models_by_epoch = {}
        files = sorted([f for f in os.listdir(folder_path) if "epoch=" in f])

        for f_name in files:
            match = re.search(r'epoch=(\d+)', f_name)
            epoch_num = int(match.group(1))
            relative_path = os.path.join(folder, f_name)
            model = self.load_model(tags, relative_path)
            models_by_epoch[epoch_num] = model
        return models_by_epoch
