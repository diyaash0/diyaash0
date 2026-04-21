# CIFAR-10 with self-pruning via learned gates
# idea: instead of pruning after training, let the network learn which weights to kill
# during training itself via a sparsity penalty on learnable gate values

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# each weight has a gate_score — sigmoid of that score multiplies the weight
# if gate → 0, the weight effectively disappears from the computation

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # init gate_scores slightly negative so gates start around 0.3 rather than 0.5
        # this makes it easier for the L1 penalty to push them all the way to ~0
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -0.5))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        w = self.weight * gates
        return F.linear(x, w, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def gate_l1(self):
        return torch.sigmoid(self.gate_scores).sum()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sparsity_loss(self):
        return self.fc1.gate_l1() + self.fc2.gate_l1() + self.fc3.gate_l1() + self.fc4.gate_l1()

    def all_gates(self):
        vals = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                vals.append(m.get_gates().cpu().numpy().flatten())
        return np.concatenate(vals)


def get_loaders(batch_size=256):
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=test_tfm)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    )


def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total, ce_total = 0.0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        ce = F.cross_entropy(out, labels)
        sp = model.sparsity_loss()
        loss = ce + lam * sp
        loss.backward()
        optimizer.step()
        total += loss.item()
        ce_total += ce.item()
    n = len(loader)
    return total / n, ce_total / n


@torch.no_grad()
def get_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def sparsity_pct(model, thresh=0.1):
    # using 0.1 as threshold — gate < 0.1 means it's contributing less than 10%
    # of the weight, practically pruned
    gates = model.all_gates()
    return 100.0 * (gates < thresh).sum() / len(gates)


def run(lam, epochs, device, train_loader, test_loader):
    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\nlambda = {lam}")
    for epoch in range(1, epochs + 1):
        loss, ce = train_one_epoch(model, train_loader, opt, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = get_acc(model, test_loader, device)
            pruned = sparsity_pct(model)
            print(f"  ep {epoch:3d} | loss {loss:.3f} | ce {ce:.3f} | acc {acc:.1f}% | pruned {pruned:.1f}%")

    acc = get_acc(model, test_loader, device)
    pruned = sparsity_pct(model)
    gates = model.all_gates()
    print(f"  -> final: acc={acc:.2f}%  sparsity={pruned:.2f}%")
    return acc, pruned, gates


def plot_gates(gates_by_lam, best_lam):
    fig, axes = plt.subplots(1, len(gates_by_lam), figsize=(5 * len(gates_by_lam), 4), sharey=True)
    if len(gates_by_lam) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'seagreen']
    for ax, (lam, gates), c in zip(axes, gates_by_lam.items(), colors):
        ax.hist(gates, bins=80, color=c, edgecolor='white', linewidth=0.3)
        title = f'λ={lam}' + (' ← best' if lam == best_lam else '')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('gate value')
        if ax == axes[0]:
            ax.set_ylabel('count')
        ax.axvline(0.1, color='red', linestyle='--', linewidth=1, label='thresh=0.1')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('gate value distributions after training', fontsize=12)
    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=150, bbox_inches='tight')
    print('\nsaved gate_distribution.png')
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_loader, test_loader = get_loaders()

    # lambdas need to be high enough to actually push gates down
    # with ~1.7M gates the sparsity loss term is huge in absolute value,
    # so we need lambda in the range 1e-3 to 1e-1 to see real pruning
    lambdas = [1e-3, 1e-2, 1e-1]
    epochs = 30

    results = {}
    gates_by_lam = {}

    for lam in lambdas:
        acc, pruned, gates = run(lam, epochs, device, train_loader, test_loader)
        results[lam] = (acc, pruned)
        gates_by_lam[lam] = gates

    print('\n' + '-' * 52)
    print(f'  {"lambda":<10} {"accuracy":>12} {"sparsity":>12}')
    print('-' * 52)
    for lam, (acc, sp) in results.items():
        print(f'  {lam:<10} {acc:>11.2f}%  {sp:>10.2f}%')
    print('-' * 52)

    best_lam = max(results, key=lambda l: results[l][0])
    plot_gates(gates_by_lam, best_lam)


if __name__ == '__main__':
    main()
