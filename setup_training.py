import random
import os
import time
import math
import re
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.types import Device
from torch.nn.modules.loss import _Loss
import torchtext
from transformers import AutoModelForCausalLM, AutoTokenizer

from quixer.quixer_model import Quixer
from quixer.baseline_models import Transformer, LSTM, FNet

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    """
    Computes time elapsed in minutes and seconds when given two UNIX timestamps
    with the starting time and ending time.

    Args:
      start_time: Starting time as a UNIX timestamp.
      end_time: End time as a UNIX timestamp.
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
    Takes in a sequence of token IDs as a torch tensor `data` and returns a torch tensor containing
    the training data with shape `[number of batches + window_size, batch_size]`.

    Each batch is represented by `window_size` contiguous rows in the returned tensor and
    can be extracted using the `get_batch_s2s` function.

    A sequence of pad tokens of length `window_size-1` is prepended to the data so as to
    provide a context window for the first token.

    Args:
      data: A 1D torch tensor containing a sequence of token IDs.
      batch_size: The number of sequences each batch should have.
      window_size: How many tokens are considered in each context window (each of which is a sequence in the batch).
      pad_token_id: The ID of the pad token, as supplied by the tokenizer.
      device: Torch device the returned tensor is to be created on.

    Returns:
      Tensor containing data for each batch prepared for a next token prediction language
      modelling task.
    """
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (data.size(0) - 1) // batch_nr_of_elements

    # Discard tokens at the end of the data that do not fill a whole batch
    batched_data = (
        data[: nr_of_batches * batch_nr_of_elements]
        .view(batch_nr_of_elements, nr_of_batches)
        .T
    )

    # Data for the first batch
    window_data = torch.cat(
        (
            # Adds a sequence of pad tokens of length `window_size-1`
            # to provide a context window for the first token.
            torch.full((window_size, 1), pad_token_id, device=device),
            # Context for the first row of tokens in `batched_data`
            batched_data[-window_size:, :-1],
        ),
        dim=1,
    )

    return torch.cat((window_data, batched_data))


def get_batch_s2s(
    source: torch.Tensor, i: int, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the `i`th batch; expects one of the tensors returned by `setup_dataset`.

    Args:
      source: Tensor containing data.
      i: Index of the batch.
      window_size: Context window size.
    Returns:
      The `i`th batch.
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
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device, batch_size: int, window_size: int
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """
    Downloads and tokenizes the Penn TreeBank dataset, and then sets it up for a
    next-word prediction task.

    Args:
      device: Device to store dataset on.
      batch_size: Size of the batches.
      window_size: Size of the context window.

    Returns:
      Vocabulary represented by a torchtext.vocab.Vocab instance along with
      three torch tensors containing the training, validation and test data.
    """

    # Download dataset from the Hugging Face Hub / load dataset
    raw_dset = load_dataset("ptb_text_only")

    # Get training data in PyArrow format
    train_iter = raw_dset["train"].data[0]
    # Convert from arrow array to native Python list
    train_iter = [s.as_py() for s in train_iter]

    # Get torchtext tokenizer
    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), specials=["<pad>", "<unk>", "<eos>"]
    )
    # Define unknown word as the default index to use
    vocab.set_default_index(vocab["<unk>"])

    def data_process(raw_text_iter) -> torch.Tensor:
        """
        Converts raw text into a flat Tensor of token indices.
        """
        data = [
            torch.tensor(vocab(tokenizer(item)) + [vocab["<eos>"]], dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 1, data))).to(device)

    # Convert from arrow arrays to native Python lists
    train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
    val_sents = [s.as_py() for s in raw_dset["validation"].data[0]]
    test_sents = [s.as_py() for s in raw_dset["test"].data[0]]

    # Flatten datasets into one long tokenised string each
    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)

    # Get padding token
    PAD_TOKEN = vocab["<pad>"]

    # Prepare data for a next-token prediction language modelling task
    train_iter = batchify_s2s(train_flat, batch_size, window_size, PAD_TOKEN, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, PAD_TOKEN, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, PAD_TOKEN, device)

    return vocab, (train_iter, val_iter, test_iter), PAD_TOKEN


def create_model(
    hyperparams: dict[str, Any], device: Device, vocabulary_size: int
) -> torch.nn.Module:
    """
    Selects and creates model based on hyperparameters passed.

    Args:
      hyperparams: Model hyperparameters.
      device: Device the model will be run on.
      vocabulary_size: Size of the vocabulary.
    Returns:
      An instance of a torch model based on the hyperparameters passed.
    """
    model_str = hyperparams["model"]
    model: torch.nn.Module
    if model_str == "Quixer":
        model = Quixer(
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
    elif model_str == "FNet":
        model = FNet(
            vocab_size=vocabulary_size,
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            dropout=hyperparams["dropout"],
        )
    elif model_str == "Transformer":
        model = Transformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    elif model_str == "LSTM":
        model = LSTM(
            emb_dim=hyperparams["dimension"],
            hid_dim=hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model


def get_id_to_token(vocab: torchtext.vocab.Vocab) -> list[str]:
    """
    Returns vocabulary tokens in id order.
    """
    return list(vocab.get_itos())


def clean_teacher_prompt(tokens: list[str]) -> str:
    """
    Converts a Quixer word-token context back to plain text for the teacher.
    """
    specials = {"<pad>", "<unk>", "<eos>", ""}
    text = " ".join(token for token in tokens if token not in specials)
    text = re.sub(r" ([.,!?;:)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    text = re.sub(r" ' ", "'", text)
    text = re.sub(r" 's\b", "'s", text)
    return text.strip()


def contexts_to_teacher_prompts(
    contexts: torch.Tensor,
    id_to_token: list[str],
    pad_token_id: int,
) -> list[str]:
    """
    Reconstructs teacher prompts from student context token ids.
    """
    context_tokens = [
        [
            "<pad>" if token_id == pad_token_id else id_to_token[token_id]
            for token_id in row
        ]
        for row in contexts.detach().cpu().tolist()
    ]
    return [clean_teacher_prompt(tokens) for tokens in context_tokens]


def load_teacher(hyperparams: dict[str, Any]):
    """
    Loads the frozen teacher model used for UDL.
    """
    teacher_name = hyperparams["teacher_model_name"]
    teacher_device = torch.device(hyperparams.get("teacher_device", "cuda"))

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_name,
        trust_remote_code=hyperparams.get("trust_remote_code", True),
    )
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype="auto",
        trust_remote_code=hyperparams.get("trust_remote_code", True),
    ).to(teacher_device)
    teacher_model.config.pad_token_id = teacher_tokenizer.pad_token_id
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    return teacher_tokenizer, teacher_model, teacher_device


def precompute_teacher_logits(
    iterator: torch.Tensor,
    window_size: int,
    id_to_token: list[str],
    pad_token_id: int,
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
        "teacher_max_length", hyperparams.get("max_length", window_size)
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
            x, _ = get_batch_s2s(iterator, batch_idx, window_size)
            prompts = contexts_to_teacher_prompts(x, id_to_token, pad_token_id)
            prompts = [prompt or teacher_tokenizer.eos_token for prompt in prompts]

            batch_logits = []
            for start in range(0, len(prompts), teacher_batch_size):
                sub_prompts = prompts[start : start + teacher_batch_size]
                enc = teacher_tokenizer(
                    sub_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(teacher_device)

                teacher_out = teacher_model(**enc)
                last_pos = enc["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(teacher_out.logits.size(0), device=teacher_device)
                batch_logits.append(teacher_out.logits[rows, last_pos, :].cpu())

            cached_logits[batch_idx] = torch.cat(batch_logits, dim=0)

    return cached_logits


def uld_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    """
    Universal Logit Distillation loss.
    """
    student_temperature = hyperparams.get("student_temperature", 1.0)
    teacher_temperature = hyperparams.get("teacher_temperature", 1.0)

    student_probs = F.softmax(student_logits.float() / student_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / teacher_temperature, dim=-1)

    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    max_vocab_size = max(student_sorted.size(-1), teacher_sorted.size(-1))
    student_sorted = F.pad(
        student_sorted, (0, max_vocab_size - student_sorted.size(-1)), value=0.0
    )
    teacher_sorted = F.pad(
        teacher_sorted, (0, max_vocab_size - teacher_sorted.size(-1)), value=0.0
    )

    return (student_sorted - teacher_sorted).abs().sum(dim=-1).mean()


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: _Loss,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    window_size: int,
    cached_teacher_logits: Optional[torch.Tensor] = None,
    hyperparams: Optional[dict[str, Any]] = None,
    distillation_weight: float = 0.0,
):
    """
    Runs training loop for one epoch.
    """
    model.train()

    epoch_loss = 0
    epoch_ce_loss = 0
    epoch_dist_loss = 0

    n_batches = iterator.shape[0] - window_size

    idxs = list(range(n_batches))
    random.shuffle(idxs)
    model_device = next(model.parameters()).device

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad()

        yhat, norm_avg = model(x)

        ce_loss = loss_function(yhat, y)
        if cached_teacher_logits is None:
            dist_loss = torch.tensor(0.0, device=model_device)
            loss = ce_loss
        else:
            teacher_logits = cached_teacher_logits[batch_idx].to(model_device).float()
            dist_loss = uld_distillation_loss(yhat, teacher_logits, hyperparams)
            ce_weight = hyperparams.get(
                "crossentropy_weight", hyperparams.get("ce_weight", 1.0)
            )
            loss = ce_weight * ce_loss + distillation_weight * dist_loss

        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()
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
    loss_function: _Loss,
    window_size: int,
) -> float:
    """
    Evaluates model on the supplied data.
    """

    model.eval()

    epoch_loss = 0

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
    id_to_token: Optional[list[str]] = None,
    pad_token_id: Optional[int] = None,
) -> float:
    """
    Run a training cycle.

    Args:
      model: The model to train.
      hyperparams: The model hyperparameters.
      train_iter: Tensor containing training data returned by `setup_dataset` function.
      val_iter: Tensor containing validation data returned by `setup_dataset` function.
      test_iter: Tensor containing test data returned by `setup_dataset` function.
    """

    folder_path = Path("./trained_models")
    folder_path.mkdir(exist_ok=True, parents=True)
    checkpoint_fpath = (
        folder_path
        / f"q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )

    # Set up learning rate scheduler
    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    loss_function = torch.nn.CrossEntropyLoss()

    cached_teacher_logits = None
    if "teacher_model_name" in hyperparams:
        teacher_tokenizer, teacher_model, teacher_device = load_teacher(hyperparams)
        cached_teacher_logits = precompute_teacher_logits(
            train_iter,
            hyperparams["window"],
            id_to_token,
            pad_token_id,
            teacher_tokenizer,
            teacher_model,
            teacher_device,
            hyperparams,
        )
        teacher_model.cpu()
        torch.cuda.empty_cache()

    def _evaluate(iter: torch.Tensor):
        return evaluate(model, iter, loss_function, hyperparams["window"])

    initial_distillation_weight = hyperparams.get(
        "distillation_weight", hyperparams.get("kd_weight", 1.0)
    )
    final_distillation_weight = hyperparams.get(
        "distillation_weight_final", initial_distillation_weight
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
            loss_function,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["window"],
            cached_teacher_logits,
            hyperparams,
            current_distillation_weight,
        )

        valid_loss = _evaluate(val_iter)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tTrain CE: {train_ce_loss:.3f} | Train ppl: {math.exp(train_ce_loss)}")
        if cached_teacher_logits is not None:
            print(
                f"\tTrain UDL: {train_dist_loss:.6f} | "
                f"DistW: {current_distillation_weight:.3f}"
            )
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")

    model.load_state_dict(torch.load(checkpoint_fpath))

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss)}")

    return test_loss


def seed(SEED: int) -> None:
    """
    Sets the seed for Python's random module, numpy's RNG and torch's RNG.

    Args:
      SEED: integer specifying the seed
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    """
    Returns a function that runs the training cycle on a specified torch device.

    Args:
      device: Torch device

    Returns:
      Callable taking in a set of parameters as a dict and returning the value of the validation loss
      at the end of the training cycle.
    """

    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and return the test loss.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        initialise_weights(model)

        model = model.to(device)

        valid_loss = train_cycle(
            model,
            parameterization,
            train_iter,
            val_iter,
            test_iter,
            get_id_to_token(vocab),
            PAD_TOK,
        )

        return valid_loss

    return train_evaluate
