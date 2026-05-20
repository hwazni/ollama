import random
import os
import re
import time
import math
from tqdm import tqdm
from typing import Any, Tuple, Callable
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch
from torch.types import Device
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from transformers import AutoTokenizer, AutoModelForCausalLM
from quixer_model import Quixer
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

def batchify_s2s_text(
    tokens: list,
    batch_size: int,
    window_size: int,
) -> np.ndarray:
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (len(tokens) - 1) // batch_nr_of_elements

    flat = np.array(tokens[: nr_of_batches * batch_nr_of_elements], dtype=object)
    batched = flat.reshape(batch_nr_of_elements, nr_of_batches).T

    pad_block = np.full((window_size, 1), "", dtype=object)
    window_block = np.concatenate((pad_block, batched[-window_size:, :-1]), axis=1)
    return np.concatenate((window_block, batched))


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
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray, int]:
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
    raw_dset = load_dataset("ptb_text_only", trust_remote_code=True)

    # DATASET_PATH = r"/home/ccib_dcda/data/Hadi-Main/Distilled-Quixer/ptb-dataset"
    # data_files = {"train": "ptb.train.txt", "test": "ptb.test.txt", "validation": "ptb.valid.txt"}
    # raw_dset = load_dataset(DATASET_PATH, data_files=data_files)

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

    def data_process(raw_text_iter):
        """
        Converts raw text into a flat Tensor of token indices and a parallel flat list of token strings.
        """
        token_lists = [tokenizer(item) + ["<eos>"] for item in raw_text_iter]
        token_lists = [t for t in token_lists if len(t) > 1]
        flat_strings = [tok for sent in token_lists for tok in sent]
        flat_ids = torch.tensor(vocab(flat_strings), dtype=torch.long).to(device)
        return flat_ids, flat_strings

    # Convert from arrow arrays to native Python lists
    train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
    val_sents = [s.as_py() for s in raw_dset["validation"].data[0]]
    test_sents = [s.as_py() for s in raw_dset["test"].data[0]]

    # Flatten datasets into one long tokenised string each
    train_flat, train_flat_text = data_process(train_sents)
    val_flat, _ = data_process(val_sents)
    test_flat, _ = data_process(test_sents)

    # Get padding token
    PAD_TOKEN = vocab["<pad>"]

    # Prepare data for a next-token prediction language modelling task
    train_iter = batchify_s2s(train_flat, batch_size, window_size, PAD_TOKEN, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, PAD_TOKEN, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, PAD_TOKEN, device)

    train_text_iter = batchify_s2s_text(train_flat_text, batch_size, window_size)

    return vocab, (train_iter, val_iter, test_iter), train_text_iter, PAD_TOKEN

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
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model

def uld_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    """
    Universal Logit Distillation loss between student and teacher logits.
    """
    student_temp = hyperparams["student_temperature"]
    teacher_temp = hyperparams["teacher_temperature"]
    if student_temp <= 0 or teacher_temp <= 0:
        raise ValueError("student_temperature and teacher_temperature must be > 0")

    student_probs = F.softmax(student_logits.float() / student_temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / teacher_temp, dim=-1)

    # GOLD-style ULD compares sorted probability ranks.
    student_sorted = student_probs.sort(dim=-1, descending=True).values
    teacher_sorted = teacher_probs.sort(dim=-1, descending=True).values

    # Match vocab dimensions by zero-padding the smaller side.
    student_vocab_size = student_sorted.size(-1)
    teacher_vocab_size = teacher_sorted.size(-1)
    max_vocab_size = max(student_vocab_size, teacher_vocab_size)

    if student_vocab_size < max_vocab_size:
        student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
    if teacher_vocab_size < max_vocab_size:
        teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

    # Normalize by number of positions (batch or batch*time), not by vocab size.
    num_positions = max(1, student_sorted.numel() // student_sorted.size(-1))
    return F.l1_loss(student_sorted, teacher_sorted, reduction="sum") / num_positions

def uld_combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    hyperparams: dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined cross-entropy + ULD distillation loss for next-token prediction.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    dist_loss = uld_distillation_loss(student_logits, teacher_logits, hyperparams)
    total = hyperparams["crossentropy_weight"] * ce_loss + hyperparams["distillation_weight"] * dist_loss
    return total, ce_loss, dist_loss

def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip: float,
    window_size: int,
    cached_teacher_logits: torch.Tensor,
    hyperparams: dict[str, Any],
    current_distillation_weight: float,
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

    for batch_idx in tqdm(idxs, total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        teacher_next_token_logits = cached_teacher_logits[batch_idx].to(model_device)

        optimizer.zero_grad()

        student_logits, _ = model(x)

        total_loss, ce_loss, dist_loss = uld_combined_loss(
            student_logits, teacher_next_token_logits, y, hyperparams
        )
        # Allow epoch-wise annealing of distillation pressure.
        total_loss = total_loss + (current_distillation_weight - hyperparams["distillation_weight"]) * dist_loss
        # print(f"Batch {batch_idx}: Total Loss={total_loss.item():.4f}, CE Loss={ce_loss.item():.4f}, Distillation Loss={dist_loss.item():.4f}")
        total_loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_ce_loss += ce_loss.item()
        epoch_dist_loss += dist_loss.item()

    return epoch_loss / n_batches, epoch_ce_loss / n_batches, epoch_dist_loss / n_batches

def precompute_teacher_logits(
    iterator: torch.Tensor,
    text_iter: np.ndarray,
    window_size: int,
    teacher_tokenizer: AutoTokenizer,
    teacher_model: AutoModelForCausalLM,
    device: Device,
    hyperparams: dict[str, Any],
) -> torch.Tensor:
    """
    Pre-computes teacher next-token logits for all batches and caches them on CPU.
    Returns a tensor of shape [n_batches, batch_size, teacher_vocab_size].
    """
    n_batches = iterator.shape[0] - window_size
    specials = {"<pad>", "<unk>", "<eos>", ""}

    def _clean(tokens):
        text = " ".join(t for t in tokens if t not in specials)
        text = re.sub(r" ([.,!?;:)])", r"\1", text)
        text = re.sub(r"([(]) ", r"\1", text)
        text = re.sub(r" ' ", "'", text)
        text = re.sub(r" 's\b", "'s", text)
        return text

    teacher_inference_batch_size = hyperparams.get("teacher_inference_batch_size", 16)
    all_logits = []
    print("Pre-computing teacher logits...")

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x_text, _ = get_batch_s2s(text_iter, batch_idx, window_size)
            x_text_cleaned = [_clean(row) for row in x_text]

            # Process in smaller sub-batches to avoid OOM
            sub_logits = []
            for i in range(0, len(x_text_cleaned), teacher_inference_batch_size):
                sub = x_text_cleaned[i : i + teacher_inference_batch_size]
                enc = teacher_tokenizer(
                    sub,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=hyperparams["max_length"],
                ).to(device)
                teacher_out = teacher_model(**enc)
                sub_logits.append(teacher_out.logits[:, -1, :].cpu())

            all_logits.append(torch.cat(sub_logits, dim=0))

    return torch.stack(all_logits)  # [n_batches, batch_size, teacher_vocab_size]


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: torch.nn.Module,
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
    train_text_iter: np.ndarray,
    teacher_tokenizer: AutoTokenizer,
    teacher_model: AutoModelForCausalLM,
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hyperparams["epochs"],
            eta_min=hyperparams.get("min_lr", 0.0),
        )
    elif hyperparams["lr_sched"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=hyperparams.get("plateau_factor", 0.5),
            patience=hyperparams.get("plateau_patience", 2),
            min_lr=hyperparams.get("min_lr", 1e-5),
        )

    loss_function = torch.nn.CrossEntropyLoss()

    cached_teacher_logits = precompute_teacher_logits(
        train_iter, train_text_iter, hyperparams["window"],
        teacher_tokenizer, teacher_model, next(teacher_model.parameters()).device,
        hyperparams,
    )

    # Teacher is no longer needed — offload to CPU to free VRAM for student training
    teacher_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _evaluate(data: torch.Tensor):
        return evaluate(model, data, loss_function, hyperparams["window"])

    best_valid_loss = float("inf")
    initial_distillation_weight = hyperparams["distillation_weight"]
    final_distillation_weight = hyperparams.get(
        "distillation_weight_final", initial_distillation_weight
    )

    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        if hyperparams["epochs"] > 1:
            anneal_fraction = epoch / (hyperparams["epochs"] - 1)
        else:
            anneal_fraction = 1.0
        current_distillation_weight = (
            initial_distillation_weight
            + anneal_fraction * (final_distillation_weight - initial_distillation_weight)
        )

        train_loss, train_ce_loss, train_dist_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            hyperparams["max_grad_norm"],
            hyperparams["window"],
            cached_teacher_logits,
            hyperparams,
            current_distillation_weight,
        )

        valid_loss = _evaluate(val_iter)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {math.exp(train_loss)}")
        print(f"\tTrain CE: {train_ce_loss:.3f} | Train Dist: {train_dist_loss:.6f}")
        print(f"\tLR: {current_lr:.6f} | DistW: {current_distillation_weight:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")

    model.load_state_dict(torch.load(checkpoint_fpath, weights_only=True))

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device, teacher_model_id: str) -> Callable:
    """
    Returns a function that runs the training cycle on a specified torch device.

    Args:
      device: Torch device

    Returns:
      Callable taking in a set of parameters as a dict and returning the value of the validation loss
      at the end of the training cycle.
    """

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        torch_dtype="auto",
    ).to(device)
    teacher_model.eval()

    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and return the test loss.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), train_text_iter, _ = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        initialise_weights(model)

        model = model.to(device)

        # train_cycle may offload teacher to CPU to free VRAM; move it back per run.
        teacher_model.to(device)
        teacher_model.eval()

        valid_loss = train_cycle(
            model, parameterization, train_iter, val_iter, test_iter, train_text_iter, teacher_tokenizer, teacher_model
        )

        return valid_loss

    return train_evaluate