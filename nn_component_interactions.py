"""
Neural Network Component Interaction Experiments on CIFAR-10
============================================================

This script reproduces the core experiments described in the paper:

"A Framework for Understanding Neural Network Component Interactions:
Selection Principles, Guidelines, and Empirical Evidence"

Experiments:
    1) Activation × Initialization (Table 4 in paper)
    2) Normalization × Optimizer  (Table 5 in paper)

Each configuration is run across 5 random seeds for statistical reliability
(Section 6.4). Mean and standard deviation are reported.

Outputs:
    - table4_activation_init_results.csv
    - table5_norm_optimizer_results.csv
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

import pandas as pd


# ---------------------------------------------------------------------------
# 0. Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# ---------------------------------------------------------------------------
# 1. Reproducibility Helper
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Fix all random sources for reproducibility (Section 6.1 & 6.4).
    Base seed used in paper: 42. Additional seeds: 0, 1, 7, 123.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEEDS = [42, 0, 1, 7, 123]   # 5 seeds as described in Section 6.4


# ---------------------------------------------------------------------------
# 2. CIFAR-10 Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size_train: int = 128, batch_size_test: int = 256):
    """
    CIFAR-10 loaders with standard augmentation (Section 6.1):
    random crop (padding=4) + random horizontal flip.
    Batch size 128 as specified in the paper.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=2
    )

    return trainloader, testloader


# ---------------------------------------------------------------------------
# 3. Initialization Functions
# ---------------------------------------------------------------------------

def init_weights_xavier(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_he(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_random(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_weights_orthogonal(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 4. Activation Utilities
# ---------------------------------------------------------------------------

def get_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=True)


def replace_relu_recursive(module: nn.Module, act_name: str):
    """Recursively replace all nn.ReLU with the requested activation."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, get_activation_module(act_name))
        else:
            replace_relu_recursive(child, act_name)


# ---------------------------------------------------------------------------
# 5. Model Builders
# ---------------------------------------------------------------------------

def build_resnet18(
    activation: str = "relu",
    init: str = "he",
    num_classes: int = 10,
) -> nn.Module:
    """
    ResNet-18 for CIFAR-10 with configurable activation and initialization.
    Uses default BatchNorm (Experiment 1 / Table 4).
    Architecture: 4 residual blocks, feature maps [64, 128, 256, 512] (Section 6.1).
    """
    model = resnet18(weights=None, num_classes=num_classes)
    replace_relu_recursive(model, activation)

    init_lower = init.lower()
    if init_lower == "he":
        model.apply(init_weights_he)
    elif init_lower == "xavier":
        model.apply(init_weights_xavier)
    elif init_lower == "random":
        model.apply(init_weights_random)
    elif init_lower == "orthogonal":
        model.apply(init_weights_orthogonal)
    else:
        model.apply(init_weights_he)

    return model.to(device)


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over the channel dimension for 2D feature maps (N x C x H x W).
    Used as a drop-in replacement for BatchNorm2d in Experiment 2 / Table 5.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        n, c, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2).contiguous()


def replace_bn_with_layernorm(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, ChannelLayerNorm(child.num_features))
        else:
            replace_bn_with_layernorm(child)


def remove_batchnorm(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            remove_batchnorm(child)


def build_resnet18_with_norm(
    norm_type: str = "batchnorm",
    activation: str = "relu",
    init: str = "he",
    num_classes: int = 10,
) -> nn.Module:
    """
    ResNet-18 with controllable normalization for Experiment 2 / Table 5.
    norm_type: 'batchnorm' | 'layernorm' | 'none'
    """
    model = resnet18(weights=None, num_classes=num_classes)
    replace_relu_recursive(model, activation)

    init_lower = init.lower()
    if init_lower == "he":
        model.apply(init_weights_he)
    elif init_lower == "xavier":
        model.apply(init_weights_xavier)
    else:
        model.apply(init_weights_he)

    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        replace_bn_with_layernorm(model)
    elif norm_type == "none":
        remove_batchnorm(model)

    return model.to(device)


# ---------------------------------------------------------------------------
# 6. Training & Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    max_epochs: int,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(
        f"  Epoch [{epoch:03d}/{max_epochs}] "
        f"Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.2f}%"
    )
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / total
    return acc, avg_loss


# ---------------------------------------------------------------------------
# 7. Single-Run Helpers
# ---------------------------------------------------------------------------

def run_single_activation_init(
    act_name: str,
    init_name: str,
    seed: int,
    max_epochs: int = 100,
    target_train_acc: float = 90.0,
):
    """One seed run for Experiment 1 (Table 4)."""
    set_seed(seed)
    trainloader, testloader = get_cifar10_loaders()

    model = build_resnet18(activation=act_name, init=init_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )

    best_test_acc = 0.0
    epochs_to_target = None

    for epoch in range(1, max_epochs + 1):
        _, train_acc = train_one_epoch(
            model, optimizer, criterion, trainloader, epoch, max_epochs
        )
        test_acc, _ = evaluate(model, testloader, criterion)
        best_test_acc = max(best_test_acc, test_acc)

        if epochs_to_target is None and train_acc >= target_train_acc:
            epochs_to_target = epoch

    return epochs_to_target, round(best_test_acc, 2)


def run_single_norm_optimizer(
    norm_type: str,
    optimizer_name: str,
    seed: int,
    max_epochs: int = 100,
    target_train_acc: float = 90.0,
):
    """One seed run for Experiment 2 (Table 5)."""
    set_seed(seed)
    trainloader, testloader = get_cifar10_loaders()

    model = build_resnet18_with_norm(norm_type=norm_type, activation="relu", init="he")
    criterion = nn.CrossEntropyLoss()

    optim_lower = optimizer_name.lower()
    if optim_lower == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
    elif optim_lower == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=5e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    best_test_acc = 0.0
    epochs_to_target = None
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        _, train_acc = train_one_epoch(
            model, optimizer, criterion, trainloader, epoch, max_epochs
        )
        test_acc, _ = evaluate(model, testloader, criterion)
        best_test_acc = max(best_test_acc, test_acc)

        if epochs_to_target is None and train_acc >= target_train_acc:
            epochs_to_target = epoch

    training_time = time.time() - start
    return epochs_to_target, round(best_test_acc, 2), int(training_time)


# ---------------------------------------------------------------------------
# 8. Experiment 1: Activation × Initialization  (Table 4)
# ---------------------------------------------------------------------------

def run_table4():
    """
    Reproduce Table 4: Activation-Initialization Convergence Results.
    Each configuration is run with 5 random seeds (Section 6.4).
    Reports mean ± std for epochs_to_90 and best test accuracy.
    """
    configs = [
        ("relu",       "he"),
        ("relu",       "xavier"),
        ("relu",       "random"),
        ("leaky_relu", "he"),
        ("leaky_relu", "xavier"),
        ("gelu",       "he"),
        ("gelu",       "xavier"),
        ("tanh",       "he"),
        ("tanh",       "xavier"),
        ("sigmoid",    "xavier"),
    ]

    rows = []
    for act, init in configs:
        print(f"\n{'='*70}")
        print(f"[Table 4] Activation={act}  Init={init}  ({len(SEEDS)} seeds)")
        print(f"{'='*70}")

        epoch_results = []
        acc_results = []

        for seed in SEEDS:
            print(f"  -- Seed {seed}")
            epochs, acc = run_single_activation_init(act, init, seed)
            epoch_results.append(epochs if epochs is not None else 100)
            acc_results.append(acc)

        mean_epochs = round(float(np.mean(epoch_results)), 1)
        std_epochs  = round(float(np.std(epoch_results)),  1)
        mean_acc    = round(float(np.mean(acc_results)),   2)
        std_acc     = round(float(np.std(acc_results)),    2)

        rows.append({
            "activation":          act,
            "initialization":      init,
            "mean_epochs_to_90":   mean_epochs,
            "std_epochs_to_90":    std_epochs,
            "mean_test_acc":       mean_acc,
            "std_test_acc":        std_acc,
        })
        print(
            f"  >> epochs_to_90 = {mean_epochs} ± {std_epochs}  |  "
            f"test_acc = {mean_acc} ± {std_acc}%"
        )

    df = pd.DataFrame(rows)
    print("\n[Table 4 – Final Results]")
    print(df.to_string(index=False))
    df.to_csv("table4_activation_init_results.csv", index=False)
    print("\n[INFO] Saved → table4_activation_init_results.csv")


# ---------------------------------------------------------------------------
# 9. Experiment 2: Normalization × Optimizer  (Table 5)
# ---------------------------------------------------------------------------

def run_table5():
    """
    Reproduce Table 5: Normalization-Optimizer Interaction Results.
    Each configuration is run with 5 random seeds (Section 6.4).
    Reports mean ± std for epochs_to_90, best test accuracy, and training time.
    """
    configs = [
        ("batchnorm", "sgd"),
        ("batchnorm", "adam"),
        ("layernorm", "sgd"),
        ("layernorm", "adam"),
        ("none",      "sgd"),
        ("none",      "adam"),
    ]

    rows = []
    for norm, opt in configs:
        print(f"\n{'='*70}")
        print(f"[Table 5] Norm={norm}  Optimizer={opt}  ({len(SEEDS)} seeds)")
        print(f"{'='*70}")

        epoch_results = []
        acc_results   = []
        time_results  = []

        for seed in SEEDS:
            print(f"  -- Seed {seed}")
            epochs, acc, t = run_single_norm_optimizer(norm, opt, seed)
            epoch_results.append(epochs if epochs is not None else 100)
            acc_results.append(acc)
            time_results.append(t)

        mean_epochs = round(float(np.mean(epoch_results)), 1)
        std_epochs  = round(float(np.std(epoch_results)),  1)
        mean_acc    = round(float(np.mean(acc_results)),   2)
        std_acc     = round(float(np.std(acc_results)),    2)
        mean_time   = round(float(np.mean(time_results)),  0)

        rows.append({
            "normalization":       norm,
            "optimizer":           opt,
            "mean_epochs_to_90":   mean_epochs,
            "std_epochs_to_90":    std_epochs,
            "mean_test_acc":       mean_acc,
            "std_test_acc":        std_acc,
            "mean_time_sec":       int(mean_time),
        })
        print(
            f"  >> epochs_to_90 = {mean_epochs} ± {std_epochs}  |  "
            f"test_acc = {mean_acc} ± {std_acc}%  |  "
            f"time ≈ {int(mean_time)}s"
        )

    df = pd.DataFrame(rows)
    print("\n[Table 5 – Final Results]")
    print(df.to_string(index=False))
    df.to_csv("table5_norm_optimizer_results.csv", index=False)
    print("\n[INFO] Saved → table5_norm_optimizer_results.csv")


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    تحذير:
    تشغيل كل التجارب (Table 4 + Table 5) مع 5 seeds لكل configuration
    قد يستغرق عدة ساعات على GPU.
    يمكنك تعليق run_table4() أو run_table5() للتجربة السريعة،
    أو تقليل SEEDS إلى seed واحد فقط: SEEDS = [42]
    """

    # Experiment 1 – Activation × Initialization (Table 4)
    run_table4()

    # Experiment 2 – Normalization × Optimizer (Table 5)
    run_table5()

    print("\n[DONE] All experiments finished.")
