import os
import regex as re
import torch

from torch.functional import F
from torch import nn
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, Callable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def find_latest_epoch_file(directory: str):
    os.makedirs(directory, exist_ok=True)
    epoch_pattern = re.compile(r"epoch=(\d+)")
    latest_epoch = -1
    latest_file = None

    for filename in os.listdir(directory):
        match = epoch_pattern.match(filename)
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_file = filename

    if latest_file:
        return os.path.join(directory, latest_file), latest_epoch + 1
    return None, 0


def load_weights_and_init(conf, model_class, state_dict_location=""):
    with torch.device('cpu'):
        llm_model = model_class(params=conf)
        if state_dict_location:
            state_dict = torch.load(state_dict_location, map_location=torch.device("cpu"), weights_only=True)
            res = llm_model.load_state_dict(state_dict["model"], strict=False, assign=True)
            print(res)
    return llm_model


def create_model(root_path, args_class, model_class, state_dict_location=None):
    conf_path = root_path + "/params.json"
    with open(conf_path, 'r') as file:
        conf = args_class.from_json(file.read())
        print(conf)

    return load_weights_and_init(conf, model_class, state_dict_location)


@dataclass_json
@dataclass
class TrainConf:
    epochs: int = None
    start_epoch: int = 0
    gradient_accumulation: int = 1
    max_grad_norm: float = 10000.0
    eval_modulo: int = 1
    patience: int = float('inf')
    optimizer: torch.optim.Optimizer = None
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ds_loader: Callable[[list, int, bool], DataLoader] = None
    eval_model: Optional[Callable[[nn.Module, int, Callable[[str], None]], None]] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


def ce_loss_fn(logits, labels):
    return F.cross_entropy(logits, labels)


def model_loss(llm, batch, backward=True):
    input_ids = batch["input_ids"].to("cuda")
    labels = batch["labels"].to("cuda")

    llm_output = llm.forward_train(input_ids)
    loss = ce_loss_fn(llm_output.transpose(1, 2), labels)
    if backward:
        loss.backward()
    return loss


def train(conf: TrainConf, llm_model: nn.Module, train_ds: list, validation_ds: list, calculate_loss=model_loss, log=lambda x: print(x, end="")):
    torch.autograd.set_detect_anomaly(True)
    best_vloss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(conf.start_epoch, conf.epochs):
        train_loader = conf.ds_loader(train_ds, epoch, True)
        validation_loader = conf.ds_loader(validation_ds, epoch, False)

        llm_model.train()
        training_loss_curve = []
        gradient_norm = []

        train_iterator = iter(train_loader)
        while True:
            step_tloss = 0.0
            micro_tbatches = 0
            for _ in range(conf.gradient_accumulation):
                try:
                    step_tloss += calculate_loss(llm_model, next(train_iterator)).item()
                    micro_tbatches += 1
                except StopIteration:
                    break
            if micro_tbatches == 0:
                break

            current_norm = torch.nn.utils.clip_grad_norm_(llm_model.parameters(), conf.max_grad_norm)
            gradient_norm.append(current_norm)
            conf.optimizer.step()
            conf.optimizer.zero_grad()
            training_loss_curve.append(step_tloss / micro_tbatches)

        if epoch % conf.eval_modulo == conf.eval_modulo - 1:
            validation_loss_curve = []
            llm_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(validation_loader):
                    loss = calculate_loss(llm_model, batch, backward=False)
                    validation_loss_curve.append(loss.item())
            avg_tloss = round(sum(training_loss_curve) / len(training_loss_curve), 4)
            avg_vloss = round(sum(validation_loss_curve) / len(validation_loss_curve), 4)

            log(f"EPOCH {epoch}: len(train)={len(train_loader)}, len(validation)={len(validation_loader)}\n")
            log(f"LOSS train {avg_tloss} lr {conf.scheduler.get_last_lr() if conf.scheduler is not None else get_lr(conf.optimizer)} min gradient norm: {min(gradient_norm)} max gradient norm: {max(gradient_norm)}\n")
            log(f"LOSS validation {avg_vloss}\n\n")

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= conf.patience:
                    log(f"Early stopping at epoch {epoch}\n")
                    break

        if conf.scheduler is not None:
            conf.scheduler.step()

        if conf.eval_model is not None:
            conf.eval_model(llm_model, epoch, log)
    log("Done.")


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(str(param_group['lr']))
    return "\t".join(lrs)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
