import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from utils import GBlur, LARSWrapper, accuracy


class ContrastiveLearningViewGenerator:

    def __init__(self, n_patch: int = 4, seed: int = 0):
        self.n_patch = n_patch
        self.rng_ = torch.Generator(device='cpu')
        self.rng_.manual_seed(seed)
        blur_seed = torch.randint(1 << 31, size=(1,), generator=self.rng_).item()
        self.transform_ = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.25, 0.25), ratio=(1, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1, seed=blur_seed),
            transforms.RandomSolarize(threshold=192.0, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

    def __call__(self, x):
        return [self.transform_(x) for _ in range(self.n_patch)]


def get_backbone(arch: str) -> Tuple[nn.Module, int]:
    if arch == 'resnet18-cifar':
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
    elif arch == 'resnet18-imagenet':
        backbone = resnet18()
        backbone.fc = nn.Identity()
    elif arch == 'resnet18-tinyimagenet':
        backbone = resnet18()
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone architecture: {arch}")
    return backbone, 512


class Encoder(nn.Module):

    def __init__(self, z_dim=1024, hidden_dim=4096, norm_p=2, backbone_arch='resnet18-cifar'):
        super().__init__()
        self.backbone, self.backbone_dim = get_backbone(backbone_arch)
        self.z_dim = z_dim
        self.h_dim = hidden_dim
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(
            nn.Linear(self.backbone_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, z_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = self.pre_feature(h)
        z_proj = F.normalize(self.projection(h), p=self.norm_p)
        return h, z_proj


class TotalCodingRateLoss(nn.Module):

    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def _compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  # [d, B]
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, z_proj):
        n_patches = z_proj.shape[0]
        loss = torch.zeros(n_patches)
        for i in range(n_patches):
            loss[i] = -self._compute_discrimn_loss(z_proj[i].T)
        return loss.mean()


class MeanSimilarityLoss(nn.Module):

    def forward(self, z_proj):
        n_patches, bs, _ = z_proj.shape
        z_avg = z_proj.mean(dim=0).repeat((n_patches, 1))
        z_proj = z_proj.reshape(n_patches*bs, -1)
        z_sim = F.cosine_similarity(z_proj, z_avg, dim=1).mean()
        return -z_sim


class BarycenterSphericalUniformityLoss(nn.Module):

    def forward(self, z_proj, t=2):
        z_avg = z_proj.mean(dim=0)
        return torch.pdist(z_avg, p=2).pow(2).mul(-t).exp().mean().log()


def load_dataset(
    dataset_name: str,
    train: bool = True,
    n_patch: int = 4,
    folder: Union[str, os.PathLike] = "./datasets/",
    seed: int = 0
):
    """Loads a dataset for training and testing"""
    folder = Path(folder)
    dataset_name = dataset_name.lower()
    transform = ContrastiveLearningViewGenerator(n_patch=n_patch, seed=seed)
    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(
            root=folder / "CIFAR10",
            train=train,
            download=True,
            transform=transform
        )
        trainset.n_classes = 10
    elif dataset_name == "cifar100":
        trainset = datasets.CIFAR100(
            root=folder / "CIFAR100",
            train=train,
            download=True,
            transform=transform
        )
        trainset.n_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainset


def parse_args():
    parser = argparse.ArgumentParser(description='SSL-in-one-epoch')

    parser.add_argument('--exp_name', type=str, default='default',
                        help='experiment name (default: default)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100'),
                        help='data (default: cifar10)')
    parser.add_argument('--n_patches', type=int, default=100,
                        help='number of patches used in EMP-SSL (default: 100)')
    parser.add_argument('--arch', type=str, default="resnet18-cifar",
                        choices=('resnet18-cifar', 'resnet18-imagenet', 'resnet18-tinyimagenet'),
                        help='network architecture (default: resnet18-cifar)')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='max number of epochs to finish (default: 2)')
    parser.add_argument('--bs', type=int, default=100,
                        help='batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=0.3,
                        help='learning rate (default: 0.3)')
    parser.add_argument('--log_folder', type=str, default='logs/EMP-SSL-Training',
                        help='directory name (default: logs/EMP-SSL-Training)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use for training (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--save_proj', default=False, action='store_true',
                        help='include this flag to save patch embeddings and projections')
    parser.add_argument('--pretrained_proj', default=None, type=str,
                        help='use pretrained weights for the projection network')
    parser.add_argument('--h_dim', default=4096, type=int, help='patch embedding dimensionality')
    parser.add_argument('--z_dim', default=1024, type=int, help='projection dimensionality')
    parser.add_argument('--uniformity_loss', default='tcr', type=str, choices=('tcr', 'vonmises'),
                        help='loss to use for enforcing output space uniformity (default: tcr)')
    parser.add_argument('--emb_pool', default='features', type=str, choices=('features', 'proj'),
                        help='which tensors to pool as a final representation (default: features)')
    parser.add_argument('--invariance_loss_weight', type=float, default=200.,
                        help='coefficient of token similarity (default: 200.0)')
    parser.add_argument('--uniformity_loss_weight', type=float, default=1.,
                        help='coefficient of token uniformity (default: 1.0)')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='if training should be resumed from the latest checkpoint')
    parser.add_argument('--tcr_eps', type=float, default=0.2,
                        help='eps for TCR (default: 0.2)')

    args = parser.parse_args()
    return args


args = parse_args()
print("* Parameters:", args)
torch.manual_seed(args.seed)

# folder for logging checkpoints and metrics
exp_dir = Path(args.log_folder) / f"{args.exp_name}__numpatch{args.n_patches}_bs{args.bs}_lr{args.lr}"
model_dir = exp_dir / "checkpoints"
model_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir = exp_dir / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

# detect available device
device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')
n_workers = min(8, os.cpu_count()-1)

# setup dataset and data loader
train_dataset = load_dataset(args.dataset, train=True, n_patch=args.n_patches, seed=args.seed)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers
)
test_dataset = load_dataset(args.dataset, train=False, n_patch=args.n_patches)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers
)


def train(net: nn.Module, first_epoch: int = 0):
    # setup optimizer and scheduler
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True,)
    n_converge = (len(train_dataloader) // args.bs) * args.n_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=n_converge, eta_min=0, last_epoch=-1)

    # training criterion
    similarity_loss = MeanSimilarityLoss()
    if args.uniformity_loss.lower() == 'tcr':
        uniformity_reg = TotalCodingRateLoss(eps=args.tcr_eps)
    elif args.uniformity_loss.lower() == 'vonmises':
        uniformity_reg = BarycenterSphericalUniformityLoss()
    else:
        raise ValueError(f"Unknown uniformity loss: {args.uniformity_loss}")

    for epoch in range(first_epoch, args.n_epochs):
        for (X, _) in tqdm(train_dataloader, desc=f"Epoch {epoch+1:03d}/{args.n_epochs:03d}"):
            X = torch.stack(X, dim=0).to(device)
            n_patches, bs, C, H, W = X.shape
            X = X.reshape(n_patches*bs, C, H, W)
            _, z_proj = net(X)
            z_proj = z_proj.reshape(n_patches, bs, -1)
            loss_sim = similarity_loss(z_proj)
            loss_unif = uniformity_reg(z_proj)
            loss = args.invariance_loss_weight*loss_sim + args.uniformity_loss_weight*loss_unif

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(
            f"Epoch: {epoch+1:03d} | "
            f"Loss sim: {loss_sim.item():.5f} | "
            f"Loss unif: {loss_unif.item():.5f}"
        )

        # save checkpoint
        # xxx(okachaiev): for resume to work correctly we should also save optim and scheduler
        torch.save(net.state_dict(), model_dir / f"{epoch}.pt")


def encode(net: Encoder, data_loader, subset_file: Union[str, os.PathLike]) -> TensorDataset:
    n_samples = len(data_loader)*args.bs
    if args.emb_pool.lower() == 'features':
        emb_dim = net.h_dim
    elif args.emb_pool.lower() == 'proj':
        emb_dim = net.z_dim
    else:
        raise ValueError(f"Unknown embedding pooling is given: {args.emb_pool}")
    embeddings = torch.zeros((n_samples, emb_dim))
    labels = torch.zeros((n_samples,))
    if args.save_proj:
        features = torch.zeros((n_samples, args.n_patches, net.h_dim))
        projections = torch.zeros((n_samples, args.n_patches, net.z_dim))
    for batch_id, (X, y) in enumerate(tqdm(data_loader)):
        X = torch.stack(X, dim=0).to(device)
        n_patches, bs, C, H, W = X.shape
        X = X.reshape(n_patches*bs, C, H, W)
        with torch.no_grad():
            h, z_proj = net(X)
        h = h.reshape(n_patches, bs, net.h_dim).permute(1, 0, 2)
        z_proj = z_proj.reshape(n_patches, bs, net.z_dim).permute(1, 0, 2)
        if emb_dim == net.h_dim:
            emb = h.mean(1)
        else:
            emb = z_proj.mean(1)
        embeddings[batch_id*bs:(batch_id+1)*bs, :] = emb
        labels[batch_id*bs:(batch_id+1)*bs] = y
        if args.save_proj:
            features[batch_id*bs:(batch_id+1)*bs, :, :] = h
            projections[batch_id*bs:(batch_id+1)*bs, :, :] = z_proj
    artifact = {'embeddings': embeddings, 'labels': labels}
    if args.save_proj:
        artifact.update({'features': features, 'projections': projections})
    torch.save(artifact, subset_file)
    return TensorDataset(embeddings, labels.long())


# xxx(okachaiev): we might also want to store trained classifier
def evaluate(
    train_data,
    test_data,
    report_file: Union[str, os.PathLike],
    n_epochs: int = 100,
    lr: float = 0.0075,
    batch_size: int = 100,
):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2
    )

    # setup model, optimizer, and scheduler
    classifier = nn.Linear(
        train_data.tensors[0].shape[1],
        train_dataset.n_classes
    ).to(device)
    optimizer = SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, 100)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    test_accuracy = torch.zeros(n_epochs)
    test_accuracy_top5 = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_top1 = torch.zeros(len(train_loader))
        # train
        classifier.train()
        for batch_id, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            logits = classifier(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #  batch accuracy
            top1, = accuracy(logits, y, topk=(1,))
            train_top1[batch_id] = top1
        scheduler.step()

        # train accuracy
        avg_train_top1 = train_top1.mean().item()

        # eval on test dataset now
        test_top1 = torch.zeros(len(test_loader))
        test_top5 = torch.zeros(len(test_loader))
        classifier.eval()
        for batch_id, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                logits = classifier(X)
            top1, top5 = accuracy(logits, y, topk=(1, 5))
            test_top1[batch_id] = top1
            test_top5[batch_id] = top5

        avg_test_top1 = test_top1.mean().item()
        avg_test_top5 = test_top5.mean().item()

        test_accuracy[epoch] = avg_test_top1
        test_accuracy_top5[epoch] = avg_test_top5

        print(
            f"Epoch: {epoch:03d} | "
            f"Top1 (train): {avg_train_top1*100:.4f} | "
            f"Top1 (test): {avg_test_top1*100:.4f} | "
            f"Top5 (test): {avg_test_top5*100:.4f}"
        )

    # report best performance
    print(
        f"Best top1 (test): {test_accuracy.max().item()*100:.4f} | "
        f"Last top1 (test): {test_accuracy[-1].item()*100:.4f}"
    )
    # xxx(okachaiev): it might be better to save artifact outside of the
    # function in a main "experiment" loop. or switch to a summary writer
    # like TB or DVC
    with open(report_file, "w") as fd:
        json.dump({
            "top1 (test)": test_accuracy.max().item()*100,
            "top5 (test)": test_accuracy_top5.max().item()*100,
        }, fd, indent=4)


if __name__ == '__main__':
    net = Encoder(z_dim=args.z_dim, hidden_dim=args.h_dim, backbone_arch=args.arch).to(device)
    if args.pretrained_proj:
        net_weights = net.state_dict()
        weights = torch.load(args.pretrained_proj, map_location=device)
        # filter out projection network weights
        net_weights.update({k: v for k, v in weights.items() if k.startswith('projection.')})
        net.load_state_dict(net_weights)
        # freeze training for projection
        for params in net.projection.parameters():
            params.requires_grad = False

    # stage 1: train SSL encoder
    # check if there's a checkpoint that could be loaded,
    # otherwise run training
    checkpoint_files = list(model_dir.glob(f"*.pt"))
    last_checkpoint = model_dir / f"{args.n_epochs-1}.pt"
    if os.path.exists(last_checkpoint):
        weights = torch.load(last_checkpoint, map_location=device)
        net.load_state_dict(weights)
        print(f"* Loaded SSL encoder from the checkpoint {last_checkpoint}")
    elif checkpoint_files and args.resume:
        last_epoch = max(int(file.name.replace(".pt", "")) for file in checkpoint_files)
        last_checkpoint = model_dir / f"{last_epoch}.pt"
        weights = torch.load(last_checkpoint, map_location=device)
        net.load_state_dict(weights)
        print(f"* Resume SSL encoder training from the checkpoint {last_checkpoint} for epoch {last_epoch+1}")
        train(net, first_epoch=last_epoch+1)
    else:
        print("===> Training SSL encoder")
        train(net)

    # stage 2: encode images provided by train/test data loaders
    net.eval()
    eval_datasets = {}
    for subset, dataloader in [('train', train_dataloader), ('test', test_dataloader)]:
        # check if encoded tensor is ready, otherwise run through the network
        subset_file = artifacts_dir / f"{subset}.pt"
        if os.path.exists(subset_file) and not args.resume:
            data = torch.load(subset_file, map_location='cpu')
            eval_datasets[subset] = TensorDataset(data['embeddings'], data['labels'].long())
            print(f"* Loaded encoded {subset} dataset from {subset_file}")
        else:
            print(f"===> Encoding '{subset}' dataset for evaluation")
            eval_datasets[subset] = encode(net, dataloader, subset_file)

    # stage 3: train linear classifier to measure representation performance
    report_file = exp_dir / "linear_accuracy.json"
    if os.path.exists(report_file) and not args.resume:
        print(f"* Loading linear classifier performance from {report_file}")
        with open(report_file, "r") as fd:
            print(fd.read())
    else:
        print("===> Fitting linear classifier")
        evaluate(eval_datasets['train'], eval_datasets['test'], report_file)
