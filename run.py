import argparse
import math

import torch


DEFAULT_HYPERPARAMS = {
    "model": "Quixer",
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "dimension": 512,
    "crossentropy_weight": 1.0,
    "distillation_weight": 1.0,
    "distillation_weight_final": 0.2,
    "student_temperature": 1.0,
    "teacher_temperature": 1.0,
    "teacher_device": "cuda",
    "teacher_inference_batch_size": 64,
    "trust_remote_code": True,
}


args = argparse.ArgumentParser(
    prog="Quixer UDL",
    description="Train Quixer with Universal Logit Distillation.",
)
args.add_argument("-d", "--device", default="cpu", help="Student device.")
args.add_argument(
    "--teacher-model",
    required=True,
    help="Teacher model name or local path, e.g. Qwen/Qwen3-0.6B-Base.",
)
args.add_argument(
    "--teacher-device",
    default=DEFAULT_HYPERPARAMS["teacher_device"],
    help="Teacher device, e.g. cuda, cuda:0, cuda:1.",
)
args.add_argument("--epochs", type=int, default=DEFAULT_HYPERPARAMS["epochs"])
args.add_argument("--batch-size", type=int, default=DEFAULT_HYPERPARAMS["batch_size"])
args.add_argument("--window", type=int, default=DEFAULT_HYPERPARAMS["window"])
args.add_argument("--dimension", type=int, default=DEFAULT_HYPERPARAMS["dimension"])
args.add_argument("--qubits", type=int, default=DEFAULT_HYPERPARAMS["qubits"])
args.add_argument("--layers", type=int, default=DEFAULT_HYPERPARAMS["layers"])
args.add_argument(
    "--ansatz-layers",
    type=int,
    default=DEFAULT_HYPERPARAMS["ansatz_layers"],
)
args.add_argument("--dropout", type=float, default=DEFAULT_HYPERPARAMS["dropout"])
args.add_argument("--lr", type=float, default=DEFAULT_HYPERPARAMS["lr"])
args.add_argument("--wd", type=float, default=DEFAULT_HYPERPARAMS["wd"])
args.add_argument("--eps", type=float, default=DEFAULT_HYPERPARAMS["eps"])
args.add_argument(
    "--max-grad-norm",
    type=float,
    default=DEFAULT_HYPERPARAMS["max_grad_norm"],
)
args.add_argument(
    "--restart-epochs",
    type=int,
    default=DEFAULT_HYPERPARAMS["restart_epochs"],
)
args.add_argument(
    "--ce-weight",
    type=float,
    default=DEFAULT_HYPERPARAMS["crossentropy_weight"],
)
args.add_argument(
    "--dist-weight",
    type=float,
    default=DEFAULT_HYPERPARAMS["distillation_weight"],
)
args.add_argument(
    "--dist-weight-final",
    type=float,
    default=DEFAULT_HYPERPARAMS["distillation_weight_final"],
)
args.add_argument(
    "--student-temperature",
    type=float,
    default=DEFAULT_HYPERPARAMS["student_temperature"],
)
args.add_argument(
    "--teacher-temperature",
    type=float,
    default=DEFAULT_HYPERPARAMS["teacher_temperature"],
)
args.add_argument(
    "--teacher-inference-batch-size",
    type=int,
    default=DEFAULT_HYPERPARAMS["teacher_inference_batch_size"],
)
args.add_argument(
    "--teacher-max-length",
    type=int,
    default=None,
    help="Teacher tokenizer truncation length. Defaults to the Quixer window.",
)
args.add_argument("--num-seeds", type=int, default=1)
args.add_argument("--seed", type=int, default=None)

parsed = args.parse_args()

torch.backends.cudnn.deterministic = True

device = torch.device(parsed.device)
print(f"Student device: {device}")
print(f"Teacher device: {parsed.teacher_device}")

from quixer.setup_training import get_train_evaluate

train_evaluate = get_train_evaluate(device)

base_hyperparameters = dict(DEFAULT_HYPERPARAMS)
base_hyperparameters.update(
    {
        "teacher_model_name": parsed.teacher_model,
        "teacher_device": parsed.teacher_device,
        "epochs": parsed.epochs,
        "batch_size": parsed.batch_size,
        "window": parsed.window,
        "dimension": parsed.dimension,
        "qubits": parsed.qubits,
        "layers": parsed.layers,
        "ansatz_layers": parsed.ansatz_layers,
        "dropout": parsed.dropout,
        "lr": parsed.lr,
        "wd": parsed.wd,
        "eps": parsed.eps,
        "max_grad_norm": parsed.max_grad_norm,
        "restart_epochs": parsed.restart_epochs,
        "crossentropy_weight": parsed.ce_weight,
        "distillation_weight": parsed.dist_weight,
        "distillation_weight_final": parsed.dist_weight_final,
        "student_temperature": parsed.student_temperature,
        "teacher_temperature": parsed.teacher_temperature,
        "teacher_inference_batch_size": parsed.teacher_inference_batch_size,
    }
)

if parsed.teacher_max_length is not None:
    base_hyperparameters["teacher_max_length"] = parsed.teacher_max_length

if parsed.seed is None:
    seeds = torch.randint(high=1_000_000, size=(parsed.num_seeds,)).tolist()
else:
    seeds = [parsed.seed + offset for offset in range(parsed.num_seeds)]

for seed in seeds:
    hyperparameters = dict(base_hyperparameters)
    hyperparameters["seed"] = seed
    test_loss = train_evaluate(hyperparameters)
    print(f"Seed {seed}: test loss={test_loss:.4f}, test ppl={math.exp(test_loss):.3f}")
