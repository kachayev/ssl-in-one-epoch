import argparse
import os
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import resnet18

from utils import GBlur, LARSWrapper, Solarization


class ContrastiveLearningViewGenerator:

    def __init__(self, n_patch: int = 4):
        self.n_patch = n_patch

    def __call__(self, x):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32,scale=(0.25, 0.25), ratio=(1,1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),  
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        return [transform(x) for _ in range(self.n_patch)]


def get_backbone(arch: str) -> Tuple[nn.Module, int]:
    if arch == "resnet18-cifar":
        backbone = resnet18()
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
    elif arch == "resnet18-imagenet":
        backbone = resnet18()    
        backbone.fc = nn.Identity()
    elif arch == "resnet18-tinyimagenet":
        backbone = resnet18()    
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone architecture: {arch}")
    return backbone, 512


class Encoder(nn.Module): 

    def __init__(self, z_dim=1024, hidden_dim=4096, norm_p=2, backbone_arch="resnet18-cifar"):
        super().__init__()
        backbone, feature_dim  = get_backbone(backbone_arch)
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
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
        p, m = W.shape  #[d, B]
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


class SimilarityLoss(nn.Module):

    def forward(self, z_proj):
        n_patches, bs, _ = z_proj.shape
        z_avg = z_proj.mean(dim=0).repeat((n_patches, 1))
        z_proj = z_proj.reshape(n_patches*bs, -1)
        z_sim = F.cosine_similarity(z_proj, z_avg, dim=1).mean()
        return -z_sim


# xxx(okachaeiev): i guess data_name should be enum
def load_dataset(dataset_name: str, train: bool = True, n_patch: int = 4, path: str = "./datasets/"):
    """Loads a dataset for training and testing"""
    dataset_name = dataset_name.lower()
    transform = ContrastiveLearningViewGenerator(n_patch=n_patch)
    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(
            root=os.path.join(path, "CIFAR10"),
            train=train,
            download=True,
            transform=transform
        )
        trainset.num_classes = 10
    elif dataset_name == "cifar100":
        trainset = datasets.CIFAR100(
            root=os.path.join(path, "CIFAR100"),
            train=train,
            download=True,
            transform=transform
        )
        trainset.num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainset


def parse_args():
    parser = argparse.ArgumentParser(description='SSL-in-one-epoch')

    parser.add_argument('--similarity_loss_weight', type=float, default=200.,
                        help='coefficient of cosine similarity (default: 200.0)')
    parser.add_argument('--tcr_loss_weight', type=float, default=1.,
                        help='coefficient of tcr (default: 1.0)')
    parser.add_argument('--n_patches', type=int, default=100,
                        help='number of patches used in EMP-SSL (default: 100)')
    # xxx(okachaiev): should be CHOICE type
    parser.add_argument('--arch', type=str, default="resnet18-cifar",
                        help='network architecture (default: resnet18-cifar)')
    parser.add_argument('--bs', type=int, default=100,
                        help='batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=0.3,
                        help='learning rate (default: 0.3)')
    parser.add_argument('--eps', type=float, default=0.2,
                        help='eps for TCR (default: 0.2)')
    parser.add_argument('--exp_name', type=str, default='default',
                        help='experiment name (default: default)')
    parser.add_argument('--log_folder', type=str, default='logs/EMP-SSL-Training',
                        help='directory name (default: logs/EMP-SSL-Training)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='data (default: cifar10)')
    parser.add_argument('--n_epoch', type=int, default=2,
                        help='max number of epochs to finish (default: 2)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to use for training (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()
    return args


args = parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# folder for logging checkpoints and metrics
# xxx(okachaiev): switch to pathlib
folder_name = f"{args.log_folder}/{args.exp_name}_numpatch{args.n_patches}_bs{args.bs}_lr{args.lr}"
model_dir = folder_name+"/checkpoints/"
artifacts_dir = folder_name+"/artifacts/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

# detect available device
device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')
n_workers = min(8, os.cpu_count()-1)

# setup dataset and data loader
train_dataset = load_dataset(args.dataset, train=True, n_patch=args.n_patches)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers
)
test_dataset = load_dataset(args.dataset, train=True, n_patch=args.n_patches)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers
)


def train(net: nn.Module):
    # setup optimizer and scheduler
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True,)
    n_converge = (len(train_dataloader) // args.bs) * args.n_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=n_converge, eta_min=0, last_epoch=-1)

    # training criterion
    similarity_loss = SimilarityLoss()
    tcr_loss = TotalCodingRateLoss(eps=args.eps)

    for epoch in range(args.n_epoch):
        for (X, _) in tqdm(train_dataloader):
            X = torch.stack(X, dim=0).to(device)
            n_patches, bs, C, H, W = X.shape
            X = X.reshape(n_patches*bs, C, H, W)
            _, z_proj = net(X)
            z_proj = z_proj.reshape(n_patches, bs, -1)
            loss_sim = similarity_loss(z_proj)
            loss_TCR = tcr_loss(z_proj)
            loss = args.similarity_loss_weight*loss_sim + args.tcr_loss_weight*loss_TCR

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"Epoch: {epoch} | "
              f"Loss sim: {loss_sim.item():.5f} | "
              f"Loss TCR: {loss_TCR.item():.5f}")

        # save checkpoint
        torch.save(net.state_dict(), f"{model_dir}{epoch}.pt")


def encode(net, dataloader, subset_file: str) -> TensorDataset:
    # xxx(okachaiev): if we have access to dimensions, we could
    # pre-allocated tensor to avoid dealing with python lists + cat()
    features, projections, labels = [], [], []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = torch.stack(X, dim = 0).to(device)
            n_patches, bs, C, H, W = X.shape
            X = X.reshape(n_patches*bs, C, H, W)
            z_proj, h = net(X)
            h = h.reshape(-1, n_patches, h.shape[1])
            features.append(h.cpu())
            projections.append(y.cpu())
            labels.append(z_proj.cpu())
    features, projections, labels = (
        torch.cat(features, dim=0),
        torch.cat(projections, dim=0),
        torch.cat(labels, dim=0),
    )
    torch.save({
        'features': features,
        'projections': projections,
        'labels': labels,
    }, subset_file)
    return TensorDataset(features, projections, labels)


if __name__ == '__main__':
    net = Encoder(backbone_arch=args.arch).to(device)
    net = nn.DataParallel(net)

    # stage 1: train SSL encoder
    # check if there's a checkpoint that could be loaded,
    # otherwise run training
    last_checkpoint = f"{model_dir}{args.n_epoch-1}.pt"
    if os.path.exists(last_checkpoint):
        weights = torch.load(last_checkpoint, map_location=device)
        net.load_state_dict(weights)
        print(f"* Loaded SSL encoder from the checkpoint {last_checkpoint}")
    else:
        print("===> Training SSL encoder")
        train(net)

    # stage 2: encode images provided by train/test data loaders
    net.eval()
    eval_datasets = {}
    for dataloader, subset in [(train_dataloader, 'train'), (test_dataloader, 'test')]:
        # check if encoded tensor is ready, otherwise run through the network
        subset_file = f"{artifacts_dir}{subset}.pt"
        if os.path.exists(subset_file):
            data = torch.load(subset_file, map_location='cpu')
            eval_datasets[subset] = TensorDataset(data['features'], data['projections'], data['labels'])
            print(f"* Loaded encoded {subset} dataset from {subset_file}")
        else:
            print(f"===> Encoding {subset} dataset for evaluation")
            eval_datasets[subset] = encode(net, dataloader, subset_file)
