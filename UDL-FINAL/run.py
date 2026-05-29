import argparse
import torch
from setup_training import get_train_evaluate

default_device = "mps" if torch.backends.mps.is_available() else "cpu"

quixer_hparams = {
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 8,
    "max_grad_norm": 5.0,
    "model": "Quixer",
    "print_iter": 50,
    "dimension": 512,

    # Distillation settings
    "use_uld": False,
    "teacher_model_name": "Qwen/Qwen2.5-0.5B",
    "teacher_temperature": 1.0,
    "student_temperature": 1.0,
    "ce_weight": 1.0,
    "uld_weight": 0.3,
}

parser = argparse.ArgumentParser(
    prog="Quixer",
    description="Run Quixer with optional distillation",
)

parser.add_argument("--device", default=default_device)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--use-uld", action="store_true")
parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-1.5B")
parser.add_argument("--teacher-temperature", type=float, default=1.0)
parser.add_argument("--student-temperature", type=float, default=1.0)
parser.add_argument("--ce-weight", type=float, default=1.0)
parser.add_argument("--uld-weight", type=float, default=0.3)

args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(args.device)
print(f"Running on device: {device}")

train_evaluate = get_train_evaluate(device)

hyperparameters = dict(quixer_hparams)
hyperparameters["seed"] = args.seed
hyperparameters["use_uld"] = args.use_uld
hyperparameters["teacher_model_name"] = args.teacher_model
hyperparameters["teacher_temperature"] = args.teacher_temperature
hyperparameters["student_temperature"] = args.student_temperature
hyperparameters["ce_weight"] = args.ce_weight
hyperparameters["uld_weight"] = args.uld_weight

print(
    f"Running Quixer | dim={hyperparameters['dimension']} | seed={hyperparameters['seed']}"
)
print(
    f"use_uld={hyperparameters['use_uld']} | "
    f"teacher={hyperparameters['teacher_model_name']}"
)

train_evaluate(hyperparameters)
