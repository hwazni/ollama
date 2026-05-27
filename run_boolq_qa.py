import os
import time
import math
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

from quixer.quixer_model import Quixer


SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<eos>",
    "<context>",
    "<question>",
    "<answer>",
]


quixer_hparams = {
    "model": "Quixer",

    # Quixer architecture
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "dimension": 512,
    "dropout": 0.10,

    # QA / sequence settings
    "window": 64,
    "max_context_tokens": 48,
    "max_question_tokens": 24,
    "max_answer_tokens": 1,

    # Training
    "epochs": 5,
    "batch_size": 16,
    "lr": 0.002,
    "wd": 0.0001,
    "eps": 1e-10,
    "lr_sched": "cos",
    "restart_epochs": 30000,
    "max_grad_norm": 5.0,

    # Dataset limits for quick experiments
    "max_train_examples": 2000,
    "max_val_examples": 300,
    "max_test_examples": 300,

    # Reproducibility
    "seed": 42,
}


class BoolQQADataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def initialise_weights(model):
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def tokenize_boolq_item(
    item,
    tokenizer,
    max_context_tokens: int,
    max_question_tokens: int,
):
    context_tokens = tokenizer(item["passage"])[:max_context_tokens]
    question_tokens = tokenizer(item["question"])[:max_question_tokens]

    answer_token = "yes" if item["answer"] else "no"

    tokens = (
        ["<context>"]
        + context_tokens
        + ["<question>"]
        + question_tokens
        + ["<answer>"]
        + [answer_token]
        + ["<eos>"]
    )

    return tokens


def yield_boolq_tokens(
    split,
    tokenizer,
    max_context_tokens: int,
    max_question_tokens: int,
    limit: Optional[int] = None,
):
    count = 0

    for item in split:
        tokens = tokenize_boolq_item(
            item=item,
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            max_question_tokens=max_question_tokens,
        )

        yield tokens

        count += 1

        if limit is not None and count >= limit:
            break


def boolq_sequence_to_lm_examples(
    tokens,
    vocab,
    window_size: int,
    pad_token_id: int,
):
    """
    Converts:

    <context> passage <question> question <answer> yes <eos>

    into next-token examples:

    Input:  <context> passage <question> question <answer>
    Target: yes

    Input:  <context> passage <question> question <answer> yes
    Target: <eos>
    """

    answer_index = tokens.index("<answer>")

    prompt_tokens = tokens[: answer_index + 1]
    target_tokens = tokens[answer_index + 1 :]

    examples = []

    for t in range(len(target_tokens)):
        prefix = prompt_tokens + target_tokens[:t]
        target_token = target_tokens[t]

        x_tokens = prefix[-window_size:]
        x_ids = vocab(x_tokens)

        if len(x_ids) < window_size:
            x_ids = [pad_token_id] * (window_size - len(x_ids)) + x_ids

        y_id = vocab[target_token]

        examples.append((x_ids, y_id))

    return examples


def build_boolq_lm_examples(
    split,
    tokenizer,
    vocab,
    window_size: int,
    pad_token_id: int,
    max_context_tokens: int,
    max_question_tokens: int,
    limit: Optional[int] = None,
):
    all_examples = []

    for tokens in yield_boolq_tokens(
        split=split,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens,
        max_question_tokens=max_question_tokens,
        limit=limit,
    ):
        examples = boolq_sequence_to_lm_examples(
            tokens=tokens,
            vocab=vocab,
            window_size=window_size,
            pad_token_id=pad_token_id,
        )

        all_examples.extend(examples)

    return all_examples


def setup_boolq_dataset(hparams, device):
    raw_dataset = load_dataset("google/boolq")
    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        yield_boolq_tokens(
            split=raw_dataset["train"],
            tokenizer=tokenizer,
            max_context_tokens=hparams["max_context_tokens"],
            max_question_tokens=hparams["max_question_tokens"],
            limit=hparams["max_train_examples"],
        ),
        specials=SPECIAL_TOKENS,
        min_freq=1,
    )

    vocab.set_default_index(vocab["<unk>"])

    pad_token_id = vocab["<pad>"]

    train_examples = build_boolq_lm_examples(
        split=raw_dataset["train"],
        tokenizer=tokenizer,
        vocab=vocab,
        window_size=hparams["window"],
        pad_token_id=pad_token_id,
        max_context_tokens=hparams["max_context_tokens"],
        max_question_tokens=hparams["max_question_tokens"],
        limit=hparams["max_train_examples"],
    )

    val_examples = build_boolq_lm_examples(
        split=raw_dataset["validation"],
        tokenizer=tokenizer,
        vocab=vocab,
        window_size=hparams["window"],
        pad_token_id=pad_token_id,
        max_context_tokens=hparams["max_context_tokens"],
        max_question_tokens=hparams["max_question_tokens"],
        limit=hparams["max_val_examples"],
    )

    # BoolQ has no official test split in this loader, so use validation as test for now.
    test_examples = build_boolq_lm_examples(
        split=raw_dataset["validation"],
        tokenizer=tokenizer,
        vocab=vocab,
        window_size=hparams["window"],
        pad_token_id=pad_token_id,
        max_context_tokens=hparams["max_context_tokens"],
        max_question_tokens=hparams["max_question_tokens"],
        limit=hparams["max_test_examples"],
    )

    train_loader = DataLoader(
        BoolQQADataset(train_examples),
        batch_size=hparams["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        BoolQQADataset(val_examples),
        batch_size=hparams["batch_size"],
        shuffle=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        BoolQQADataset(test_examples),
        batch_size=hparams["batch_size"],
        shuffle=False,
        drop_last=True,
    )

    print(f"BoolQ vocabulary size: {len(vocab)}")
    print(f"Train LM examples: {len(train_examples)}")
    print(f"Validation LM examples: {len(val_examples)}")
    print(f"Test LM examples: {len(test_examples)}")

    return vocab, tokenizer, train_loader, val_loader, test_loader


def create_quixer_model(hparams, vocab_size, device):
    model = Quixer(
        n_qubits=hparams["qubits"],
        n_tokens=hparams["window"],
        qsvt_polynomial_degree=hparams["layers"],
        n_ansatz_layers=hparams["ansatz_layers"],
        vocabulary_size=vocab_size,
        embedding_dimension=hparams["dimension"],
        dropout=hparams["dropout"],
        batch_size=hparams["batch_size"],
        device=device,
    )

    initialise_weights(model)

    return model.to(device)


def train_epoch(model, loader, optimizer, loss_function, scheduler, hparams, device):
    model.train()

    epoch_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)
        loss = loss_function(logits, y)

        loss.backward()

        if hparams["max_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                hparams["max_grad_norm"],
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        preds = torch.argmax(logits, dim=-1)

        correct += (preds == y).sum().item()
        total += y.numel()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    token_accuracy = correct / max(total, 1)

    return avg_loss, token_accuracy


def evaluate(model, loader, loss_function, device):
    model.eval()

    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = loss_function(logits, y)

            preds = torch.argmax(logits, dim=-1)

            correct += (preds == y).sum().item()
            total += y.numel()

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    token_accuracy = correct / max(total, 1)

    return avg_loss, token_accuracy


def generate_boolq_answer(
    model,
    vocab,
    tokenizer,
    passage: str,
    question: str,
    hparams,
    device,
):
    model.eval()

    tokens = (
        ["<context>"]
        + tokenizer(passage)[: hparams["max_context_tokens"]]
        + ["<question>"]
        + tokenizer(question)[: hparams["max_question_tokens"]]
        + ["<answer>"]
    )

    pad_token_id = vocab["<pad>"]
    itos = vocab.get_itos()

    x_ids = vocab(tokens[-hparams["window"] :])

    if len(x_ids) < hparams["window"]:
        x_ids = [pad_token_id] * (hparams["window"] - len(x_ids)) + x_ids

    x = torch.tensor([x_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(x)
        pred_id = torch.argmax(logits, dim=-1).item()

    return itos[pred_id]


def train_boolq_quixer(hparams, device):
    seed_everything(hparams["seed"])

    vocab, tokenizer, train_loader, val_loader, test_loader = setup_boolq_dataset(
        hparams=hparams,
        device=device,
    )

    model = create_quixer_model(
        hparams=hparams,
        vocab_size=len(vocab),
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["wd"],
        eps=hparams["eps"],
    )

    scheduler = None

    if hparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hparams["restart_epochs"],
        )

    loss_function = torch.nn.CrossEntropyLoss()

    folder_path = Path("./trained_models")
    folder_path.mkdir(exist_ok=True, parents=True)

    checkpoint_path = (
        folder_path
        / f"quixer_boolq_qa_seed_{hparams['seed']}_{int(time.time())}.pt"
    )

    best_val_loss = float("inf")

    for epoch in range(hparams["epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            scheduler=scheduler,
            hparams=hparams,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain loss: {train_loss:.4f} | "
            f"Train ppl: {math.exp(train_loss):.4f} | "
            f"Train token acc: {train_acc:.4f}"
        )
        print(
            f"\tVal loss: {val_loss:.4f} | "
            f"Val ppl: {math.exp(val_loss):.4f} | "
            f"Val token acc: {val_acc:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        loss_function=loss_function,
        device=device,
    )

    print("=" * 80)
    print("Final BoolQ Quixer QA results")
    print(
        f"Test loss: {test_loss:.4f} | "
        f"Test ppl: {math.exp(test_loss):.4f} | "
        f"Test token acc: {test_acc:.4f}"
    )
    print("=" * 80)

    example_passage = (
        "Paris is the capital and most populous city of France."
    )

    example_question = "Is Paris the capital of France?"

    answer = generate_boolq_answer(
        model=model,
        vocab=vocab,
        tokenizer=tokenizer,
        passage=example_passage,
        question=example_question,
        hparams=hparams,
        device=device,
    )

    print("Example generation")
    print(f"Passage: {example_passage}")
    print(f"Question: {example_question}")
    print(f"Predicted answer: {answer}")

    return model, vocab, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Quixer on BoolQ question answering."
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        help="Device: cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=quixer_hparams["epochs"],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=quixer_hparams["batch_size"],
    )

    parser.add_argument(
        "--window",
        type=int,
        default=quixer_hparams["window"],
    )

    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=quixer_hparams["max_train_examples"],
    )

    parser.add_argument(
        "--max-val-examples",
        type=int,
        default=quixer_hparams["max_val_examples"],
    )

    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=quixer_hparams["max_test_examples"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=quixer_hparams["seed"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    hparams = dict(quixer_hparams)

    hparams["epochs"] = args.epochs
    hparams["batch_size"] = args.batch_size
    hparams["window"] = args.window
    hparams["max_train_examples"] = args.max_train_examples
    hparams["max_val_examples"] = args.max_val_examples
    hparams["max_test_examples"] = args.max_test_examples
    hparams["seed"] = args.seed

    device = torch.device(args.device)

    print("=" * 80)
    print("Training Quixer on BoolQ QA")
    print(f"Device: {device}")
    print(hparams)
    print("=" * 80)

    train_boolq_quixer(
        hparams=hparams,
        device=device,
    )
