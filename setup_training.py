import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from datasets import load_dataset
from torch.types import Device
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quixer.quixer_model import Quixer


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    """
    Computes time elapsed in minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def batchify_s2s(
    data: torch.Tensor,
    batch_size: int,
    window_size: int,
    pad_token_id: int,
    device: Device,
) -> torch.Tensor:
    """
    Takes a flat token-ID tensor and prepares fixed-window next-token batches.
    """
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (data.size(0) - 1) // batch_nr_of_elements

    batched_data = (
        data[: nr_of_batches * batch_nr_of_elements]
        .view(batch_nr_of_elements, nr_of_batches)
        .T
    )

    window_data = torch.cat(
        (
            torch.full((window_size, 1), pad_token_id, device=device),
            batched_data[-window_size:, :-1],
        ),
        dim=1,
    )

    return torch.cat((window_data, batched_data))


def batchify_s2s_text(
    tokens: list[str],
    batch_size: int,
    window_size: int,
) -> np.ndarray:
    """
    Same layout as batchify_s2s, but keeps token strings for teacher prompts.
    """
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (len(tokens) - 1) // batch_nr_of_elements

    flat = np.array(tokens[: nr_of_batches * batch_nr_of_elements], dtype=object)
    batched = flat.reshape(batch_nr_of_elements, nr_of_batches).T

    pad_block = np.full((window_size, 1), "<pad>", dtype=object)
    window_block = np.concatenate((pad_block, batched[-window_size:, :-1]), axis=1)
    return np.concatenate((window_block, batched))


def get_batch_s2s(
    source: torch.Tensor | np.ndarray,
    i: int,
    window_size: int,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """
    Returns the `i`th fixed-window batch.
    """
    return source[i : i + window_size].T, source[i + window_size]


def initialise_weights(model: torch.nn.Module) -> None:
    """
    Initialises model weights.
    """

    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device,
    batch_size: int,
    window_size: int,
) -> Tuple[
    torchtext.vocab.Vocab,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    np.ndarray,
    int,
]:
    """
    Downloads Penn TreeBank and prepares Quixer next-word batches.

    Returns the normal tensor batches plus a matching train_text_iter used only
    for teacher prompt reconstruction during distillation.
    """
    raw_dset = load_dataset("ptb_text_only")
    tokenizer = get_tokenizer("basic_english")

    train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
    val_sents = [s.as_py() for s in raw_dset["validation"].data[0]]
    test_sents = [s.as_py() for s in raw_dset["test"].data[0]]

    vocab = build_vocab_from_iterator(
        map(tokenizer, train_sents),
        specials=["<pad>", "<unk>", "<eos>"],
    )
    vocab.set_default_index(vocab["<unk>"])

    def data_process(raw_text_iter) -> tuple[torch.Tensor, list[str]]:
        token_lists = [tokenizer(item) + ["<eos>"] for item in raw_text_iter]
        token_lists = [tokens for tokens in token_lists if len(tokens) > 1]
        flat_tokens = [token for tokens in token_lists for token in tokens]
        flat_ids = torch.tensor(vocab(flat_tokens), dtype=torch.long).to(device)
        return flat_ids, flat_tokens

    train_flat, train_flat_text = data_process(train_sents)
    val_flat, _ = data_process(val_sents)
    test_flat, _ = data_process(test_sents)

    pad_token = vocab["<pad>"]
    train_iter = batchify_s2s(train_flat, batch_size, window_size, pad_token, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, pad_token, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, pad_token, device)
    train_text_iter = batchify_s2s_text(train_flat_text, batch_size, window_size)

    return vocab, (train_iter, val_iter, test_iter), train_text_iter, pad_token


def create_model(
    hyperparams: dict[str, Any],
    device: Device,
    vocabulary_size: int,
) -> torch.nn.Module:
    """
    Creates the Quixer model.
    """
    return Quixer(
        n_qubits=hyperparams["qubits"],
        n_tokens=hyperparams["window"],
        qsvt_polynomial_degree=hyperparams["layers"],
        n_ansatz_layers=hyperparams["ansatz_layers"],
        vocabulary_size=vocabulary_size,
        embedding_dimension=hyperparams["dimension"],
        dropout=hyperparams["dropout"],
        batch_size=hyperparams["batch_size"],
        device=device,
    )


def load_teacher(
    hyperparams: dict[str, Any],
):
    """
    Loads and freezes the teacher model on CUDA.
    """
    teacher_name = hyperparams["teacher_model_name"]
    teacher_device = torch.device(hyperparams.get("teacher_device", "cuda"))

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_name,
        trust_remote_code=hyperparams.get("trust_remote_code", True),
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype="auto",
        trust_remote_code=hyperparams.get("trust_remote_code", True),
    ).to(teacher_device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Teacher: {teacher_name} on {teacher_device}")
    return tokenizer, model, teacher_device


def clean_teacher_prompt(tokens: np.ndarray) -> str:
    """
    Converts a Quixer word-token context back to plain text.
    """
    specials = {"<pad>", "<unk>", "<eos>", ""}
    text = " ".join(str(token) for token in tokens if token not in specials)
    text = re.sub(r" ([.,!?;:)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    text = re.sub(r" ' ", "'", text)
    text = re.sub(r" 's\b", "'s", text)
    return text.strip()


def uld_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    """
    Universal Logit Distillation loss.

    This is the UDL core: softmax, sort probabilities descending, pad the
    smaller vocabulary, then compute L1 distance averaged over batch positions.
    """
    student_temperature = hyperparams.get("student_temperature", 1.0)
    teacher_temperature = hyperparams.get("teacher_temperature", 1.0)

    student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)

    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    max_vocab_size = max(student_sorted.size(-1), teacher_sorted.size(-1))
    student_sorted = F.pad(
        student_sorted,
        (0, max_vocab_size - student_sorted.size(-1)),
        value=0.0,
    )
    teacher_sorted = F.pad(
        teacher_sorted,
        (0, max_vocab_size - teacher_sorted.size(-1)),
        value=0.0,
    )

    return (student_sorted - teacher_sorted).abs().sum(dim=-1).mean()


def combined_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    hyperparams: dict[str, Any],
    distillation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes CE + weighted UDL for Quixer next-token prediction.
    """
    ce_weight = hyperparams.get(
        "crossentropy_weight",
        hyperparams.get("ce_weight", 1.0),
    )
    ce_loss = F.cross_entropy(student_logits, labels)
    dist_loss = uld_distillation_loss(student_logits, teacher_logits, hyperparams)
    total_loss = ce_weight * ce_loss + distillation_weight * dist_loss
    return total_loss, ce_loss, dist_loss


def precompute_teacher_logits(
    iterator: torch.Tensor,
    text_iter: np.ndarray,
    window_size: int,
    teacher_tokenizer,
    teacher_model: torch.nn.Module,
    teacher_device: Device,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    """
    Precomputes teacher next-token logits for every training batch.
    """
    n_batches = iterator.shape[0] - window_size
    batch_size = iterator.shape[1]
    teacher_batch_size = hyperparams.get("teacher_inference_batch_size", batch_size)
    max_length = hyperparams.get(
        "teacher_max_length",
        hyperparams.get("max_length", window_size),
    )
    teacher_vocab_size = teacher_model.config.vocab_size
    cached_logits = torch.empty(
        (n_batches, batch_size, teacher_vocab_size),
        dtype=next(teacher_model.parameters()).dtype,
        device="cpu",
    )

    print("Precomputing teacher logits...")

    with torch.inference_mode():
        for batch_idx in tqdm(range(n_batches)):
            x_text, _ = get_batch_s2s(text_iter, batch_idx, window_size)
            prompts = [clean_teacher_prompt(row) for row in x_text]
            prompts = [prompt or teacher_tokenizer.eos_token for prompt in prompts]

            batch_logits = []
            for start in range(0, len(prompts), teacher_batch_size):
                sub_prompts = prompts[start : start + teacher_batch_size]
                encoded = teacher_tokenizer(
                    sub_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(teacher_device)

                output = teacher_model(**encoded)
                last_positions = encoded["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(output.logits.size(0), device=teacher_device)
                next_logits = output.logits[rows, last_positions, :]
                batch_logits.append(next_logits.cpu())

            cached_logits[batch_idx] = torch.cat(batch_logits, dim=0)

    return cached_logits


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    window_size: int,
    cached_teacher_logits: torch.Tensor,
    hyperparams: dict[str, Any],
    current_distillation_weight: float,
) -> tuple[float, float, float]:
    """
    Runs one training epoch.
    """
    model.train()
    epoch_loss = 0.0
    epoch_ce_loss = 0.0
    epoch_dist_loss = 0.0

    n_batches = iterator.shape[0] - window_size
    idxs = list(range(n_batches))
    random.shuffle(idxs)
    model_device = next(model.parameters()).device

    for batch_idx in tqdm(idxs, total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad(set_to_none=True)

        student_logits, _ = model(x)
        teacher_logits = cached_teacher_logits[batch_idx].to(model_device).float()
        total_loss, ce_loss, dist_loss = combined_distillation_loss(
            student_logits,
            teacher_logits,
            y,
            hyperparams,
            current_distillation_weight,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        epoch_loss += total_loss.item()
        epoch_ce_loss += ce_loss.item()
        epoch_dist_loss += dist_loss.item()

    return (
        epoch_loss / n_batches,
        epoch_ce_loss / n_batches,
        epoch_dist_loss / n_batches,
    )


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: torch.nn.Module,
    window_size: int,
) -> float:
    """
    Evaluates with CE only. PPL must always be computed from CE.
    """
    model.eval()
    epoch_loss = 0.0
    n_batches = data.shape[0] - window_size

    with torch.inference_mode():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(data, batch_idx, window_size)
            yhat, _ = model(x)
            loss = loss_function(yhat, y)
            epoch_loss += loss.item()

    return epoch_loss / n_batches


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    train_iter: torch.Tensor,
    val_iter: torch.Tensor,
    test_iter: torch.Tensor,
    train_text_iter: np.ndarray,
    teacher_tokenizer,
    teacher_model: torch.nn.Module,
    teacher_device: Device,
) -> float:
    """
    Runs the full train/validate/test cycle.
    """
    folder_path = Path("./trained_models")
    folder_path.mkdir(exist_ok=True, parents=True)
    checkpoint_fpath = (
        folder_path
        / f"q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparams["restart_epochs"],
    )

    loss_function = torch.nn.CrossEntropyLoss()

    cached_teacher_logits = precompute_teacher_logits(
        train_iter,
        train_text_iter,
        hyperparams["window"],
        teacher_tokenizer,
        teacher_model,
        teacher_device,
        hyperparams,
    )
    teacher_model.cpu()
    torch.cuda.empty_cache()

    def _evaluate(data: torch.Tensor):
        return evaluate(model, data, loss_function, hyperparams["window"])

    initial_distillation_weight = hyperparams.get(
        "distillation_weight",
        hyperparams.get("kd_weight", 1.0),
    )
    final_distillation_weight = hyperparams.get(
        "distillation_weight_final",
        initial_distillation_weight,
    )

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        anneal_fraction = epoch / max(1, hyperparams["epochs"] - 1)
        current_distillation_weight = (
            initial_distillation_weight
            + anneal_fraction * (final_distillation_weight - initial_distillation_weight)
        )

        train_loss, train_ce_loss, train_dist_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["window"],
            cached_teacher_logits,
            hyperparams,
            current_distillation_weight,
        )

        valid_loss = _evaluate(val_iter)
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tTrain CE: {train_ce_loss:.3f} | Train ppl: {math.exp(train_ce_loss):.3f}")
        print(
            f"\tTrain UDL: {train_dist_loss:.6f} | "
            f"DistW: {current_distillation_weight:.3f}"
        )
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")

    model.load_state_dict(
        torch.load(checkpoint_fpath, map_location=next(model.parameters()).device)
    )

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss):.3f}")

    return test_loss


def seed(SEED: int) -> None:
    """
    Sets Python, NumPy and PyTorch seeds.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def get_train_evaluate(device: Device) -> Callable:
    """
    Returns a callable that trains and evaluates Quixer.
    """
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        parameterization.setdefault("seed", int.from_bytes(os.urandom(4), "big"))
        parameterization.setdefault("model", "Quixer")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), train_text_iter, _ = setup_dataset(
            device,
            parameterization["batch_size"],
            parameterization["window"],
        )

        model = create_model(parameterization, device, len(vocab))
        initialise_weights(model)
        model = model.to(device)

        teacher_tokenizer, teacher_model, teacher_device = load_teacher(parameterization)
        teacher_model.to(teacher_device)
        teacher_model.eval()

        test_loss = train_cycle(
            model,
            parameterization,
            train_iter,
            val_iter,
            test_iter,
            train_text_iter,
            teacher_tokenizer,
            teacher_model,
            teacher_device,
        )

        return test_loss

    return train_evaluate
