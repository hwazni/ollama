import math
import os
import random
import re
import time
import warnings
import gc
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchtext
from datasets import load_dataset
from torch.nn.modules.loss import _Loss
from torch.types import Device
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from quixer_model import Quixer

warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
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


def get_batch_s2s(
    source: torch.Tensor,
    i: int,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return source[i : i + window_size].T, source[i + window_size]


def initialise_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device,
    batch_size: int,
    window_size: int,
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    try:
        raw_dset = load_dataset("ptb_text_only", "penn_treebank")
    except Exception:
        raw_dset = load_dataset("ptb_text_only")

    def get_sentences(split: str) -> list[str]:
        dset_split = raw_dset[split]
        if "sentence" in dset_split.column_names:
            return list(dset_split["sentence"])
        first_column = dset_split.column_names[0]
        return list(dset_split[first_column])

    train_sents = get_sentences("train")
    val_sents = get_sentences("validation")
    test_sents = get_sentences("test")

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_sents),
        specials=[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN],
    )
    vocab.set_default_index(vocab[UNK_TOKEN])

    def data_process(raw_text_iter) -> torch.Tensor:
        data = [
            torch.tensor(
                vocab(tokenizer(item)) + [vocab[EOS_TOKEN]],
                dtype=torch.long,
            )
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 1, data))).to(device)

    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)

    pad_token_id = vocab[PAD_TOKEN]

    train_iter = batchify_s2s(train_flat, batch_size, window_size, pad_token_id, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, pad_token_id, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, pad_token_id, device)

    return vocab, (train_iter, val_iter, test_iter), pad_token_id


def create_model(
    hyperparams: dict[str, Any],
    device: Device,
    vocabulary_size: int,
) -> torch.nn.Module:
    if hyperparams["model"] != "Quixer":
        raise ValueError(f"Unrecognized model: {hyperparams['model']}")

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
    teacher_model_name: str,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to("cuda")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return tokenizer, model


def decode_contexts(
    input_ids: torch.Tensor,
    vocab: torchtext.vocab.Vocab,
) -> list[str]:
    id_to_token = vocab.get_itos()
    contexts = []

    for row in input_ids.tolist():
        words = []
        for token_id in row:
            token = id_to_token[token_id]
            if token not in {PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, ""}:
                words.append(token)

        text = " ".join(words)
        text = re.sub(r" ([.,!?;:)])", r"\1", text)
        text = re.sub(r"([(]) ", r"\1", text)
        text = re.sub(r" ' ", "'", text)
        text = re.sub(r" 's\b", "'s", text)
        text = text.strip()
        contexts.append(text if text else "\n")

    return contexts


@torch.no_grad()
def teacher_next_logits(
    teacher_model: torch.nn.Module,
    teacher_tokenizer,
    contexts: list[str],
    max_length: int,
) -> torch.Tensor:
    encoded = teacher_tokenizer(
        contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}

    outputs = teacher_model(**encoded)
    last_positions = encoded["attention_mask"].sum(dim=1) - 1
    batch_positions = torch.arange(last_positions.size(0), device="cuda")
    return outputs.logits[batch_positions, last_positions].float()


def uld_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_temperature: float,
) -> torch.Tensor:
    student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)
    k = teacher_probs.size(-1)
    student_k = min(k, student_probs.size(-1))
    student_probs = torch.topk(student_probs, k=student_k, dim=-1).values

    if student_k < k:
        student_probs = F.pad(student_probs, (0, k - student_k), value=0.0)

    return torch.abs(student_probs - teacher_probs).sum(dim=-1).mean()


def _safe_cache_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def cache_teacher_probs(
    iterator: torch.Tensor,
    vocab: torchtext.vocab.Vocab,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    teacher_name = hyperparams["teacher_model_name"]
    window_size = hyperparams["window"]
    top_k = int(hyperparams.get("uld_top_k", 128))
    max_length = int(hyperparams.get("teacher_max_length", 256))
    teacher_temperature = float(hyperparams.get("teacher_temperature", 1.0))

    cache_dir = Path(hyperparams.get("teacher_cache_dir", "teacher_cache"))
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / (
        f"{_safe_cache_name(teacher_name)}"
        f"_w{window_size}_b{hyperparams['batch_size']}"
        f"_k{top_k}_t{teacher_temperature}_m{max_length}.pt"
    )

    if cache_path.exists():
        print(f"Loading teacher cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    if not torch.cuda.is_available():
        raise RuntimeError("Teacher cache precompute requires CUDA.")

    teacher_tokenizer, teacher_model = load_teacher(teacher_name)
    n_batches = iterator.shape[0] - window_size
    cached_batches = []

    print(f"Caching teacher top-{top_k} probabilities: {cache_path}")
    for batch_idx in tqdm(range(n_batches), desc="Teacher cache"):
        x, _ = get_batch_s2s(iterator, batch_idx, window_size)
        contexts = decode_contexts(x, vocab)
        logits = teacher_next_logits(
            teacher_model=teacher_model,
            teacher_tokenizer=teacher_tokenizer,
            contexts=contexts,
            max_length=max_length,
        )
        probs = F.softmax(logits / teacher_temperature, dim=-1)
        k = min(top_k, probs.size(-1))
        cached_batches.append(torch.topk(probs, k=k, dim=-1).values.cpu().half())

    teacher_probs = torch.stack(cached_batches)
    torch.save(teacher_probs, cache_path)

    del teacher_model
    del teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return teacher_probs


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: _Loss,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    window_size: int,
    teacher_probs_cache: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    uld_weight: float = 0.0,
    student_temperature: float = 1.0,
) -> tuple[float, float, float]:
    model.train()

    epoch_loss = 0.0
    epoch_ce_loss = 0.0
    epoch_uld_loss = 0.0

    n_batches = iterator.shape[0] - window_size
    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for _, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad()

        student_logits, _ = model(x)
        ce_loss = loss_function(student_logits, y)

        distill_loss = torch.zeros((), device=student_logits.device)
        if teacher_probs_cache is not None and uld_weight > 0:
            teacher_probs = teacher_probs_cache[batch_idx].to(
                device=student_logits.device,
                dtype=torch.float32,
            )
            distill_loss = uld_loss(
                student_logits=student_logits,
                teacher_probs=teacher_probs,
                student_temperature=student_temperature,
            )

        loss = ce_weight * ce_loss + uld_weight * distill_loss
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()
        epoch_ce_loss += ce_loss.item()
        epoch_uld_loss += distill_loss.item()

    return (
        epoch_loss / n_batches,
        epoch_ce_loss / n_batches,
        epoch_uld_loss / n_batches,
    )


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: _Loss,
    window_size: int,
) -> float:
    model.eval()
    epoch_loss = 0.0
    n_batches = data.shape[0] - window_size

    with torch.no_grad():
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
    vocab: torchtext.vocab.Vocab,
    device: Device,
    teacher_probs_cache: Optional[torch.Tensor] = None,
) -> float:
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

    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparams["restart_epochs"],
        )

    loss_function = torch.nn.CrossEntropyLoss()

    if teacher_probs_cache is not None:
        print(f"Teacher: {hyperparams['teacher_model_name']}")
        print(f"Student vocab: {len(vocab)} | cached teacher top-k: {teacher_probs_cache.size(-1)}")

    def _evaluate(iter: torch.Tensor):
        return evaluate(model, iter, loss_function, hyperparams["window"])

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss, train_ce_loss, train_uld_loss = train_epoch(
            model=model,
            iterator=train_iter,
            optimizer=optimizer,
            loss_function=loss_function,
            clip=hyperparams["max_grad_norm"],
            scheduler=scheduler,
            window_size=hyperparams["window"],
            teacher_probs_cache=teacher_probs_cache,
            ce_weight=hyperparams.get("ce_weight", 1.0),
            uld_weight=hyperparams.get("uld_weight", 0.0),
            student_temperature=hyperparams.get("student_temperature", 1.0),
        )

        valid_loss = _evaluate(val_iter)
        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tTrain CE: {train_ce_loss:.3f} | Train CE ppl: {math.exp(train_ce_loss):.3f}")
        if teacher_probs_cache is not None:
            print(f"\tTrain ULD: {train_uld_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")

    model.load_state_dict(torch.load(checkpoint_fpath, map_location=device))

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss):.3f}")

    return test_loss


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), _ = setup_dataset(
            device=device,
            batch_size=parameterization["batch_size"],
            window_size=parameterization["window"],
        )

        teacher_probs_cache = None
        if parameterization.get("use_uld", False):
            teacher_probs_cache = cache_teacher_probs(
                iterator=train_iter,
                vocab=vocab,
                hyperparams=parameterization,
            )

        model = create_model(parameterization, device, len(vocab))
        initialise_weights(model)
        model = model.to(device)

        return train_cycle(
            model=model,
            hyperparams=parameterization,
            train_iter=train_iter,
            val_iter=val_iter,
            test_iter=test_iter,
            vocab=vocab,
            device=device,
            teacher_probs_cache=teacher_probs_cache,
        )

    return train_evaluate
