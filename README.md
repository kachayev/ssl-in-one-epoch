# Self-Supervised Learning in One Training Epoch

This is an unofficial implementation of the paper "EMP-SSL: Towards Self-Supervised Learning in One Training Epoch", which you can access on [arXiv](https://arxiv.org/abs/2304.03977).

Original repo could be found here: [EMP-SSL](https://github.com/tsb0601/EMP-SSL). The code here is cleaned up and simplified to facilitate faster iterations. Crucially a lot of operations are reimplemented to minimize re-allocations between GPU and RAM. Extra care was taken about random seeding to ensure that the results of each run are consistently reproducible.

A tip of the hat to the [solo-learn](https://github.com/vturrisi/solo-learn) repository, from which the implementation of the LARS scheduler is borrowed.

The code is rigorously tested using both the `cifar10` and `cifar100` datasets. The quality of the representation is gauged by the performance of a linear classifier (comprising a single linear layer) trained over 100 epochs. As of now, KNN classification for evaluation hasn't been implemented. On the bright side, the entire process — covering both pre-training and final evaluation — takes < 20 minutes when run on a single Nvidia RTX A6000.

## Introduction

The paper "EMP-SSL" introduces a simplistic but efficient self-supervised learning method called Extreme-Multi-Patch Self-Supervised-Learning. The learning framework is schematically shown below (fig. from the paper):

![Training Pipeline](pipeline.png)

Please, refer to the original paper for more details.

## Install

Clone the repo:

```shell
$ git clone https://github.com/kachayev/ssl-in-one-epoch.git
$ cd ssl-in-one-epoch
```

Install libraries using `pip`:

```shell
$ pip install -r requirements.txt
```

Or using `conda`:

```shell
$ conda create -n ssl-in-one-epoch python==3.10
$ conda activate ssl-in-one-epoch
$ conda install pytorch torchvision numpy tqdm Pillow -c pytorch
```

## Run Experiment

To kick off an experiment, utilize the `main.py` script:

```shell
$ python main.py --n_patches 20 --bs 100
Files already downloaded and verified
===> Training SSL encoder
500it [06:38,  1.26it/s]
Epoch: 0 | Loss sim: -0.80343 | Loss TCR: -168.46286
500it [06:38,  1.26it/s]
Epoch: 1 | Loss sim: -0.81066 | Loss TCR: -173.45868
===> Encoding 'train' dataset for evaluation
...
```

Every experiment progresses through these stages:

    1. Train the image encoder leveraging the SSL loss.
    2. Encode all images from the provided train/test dataset.
    3. Fit a linear classifier using the encoded images as features.

Flexible artifacts caching is used to ensure that if an experiment is interrupted, it will automatically pick up from where it left off.

Logs are, by default, stored in the `logs/EMP-SSL-Training/*`` directory. Fancy a different location? Simply use the `--log_folder`` flag when launching your experiment.

Full list of options:

```shell
usage: main.py [-h] [--similarity_loss_weight SIMILARITY_LOSS_WEIGHT] [--tcr_loss_weight TCR_LOSS_WEIGHT] [--n_patches N_PATCHES]
               [--arch ARCH] [--bs BS] [--lr LR] [--eps EPS] [--exp_name EXP_NAME] [--log_folder LOG_FOLDER] [--dataset DATASET]
               [--n_epoch N_EPOCH] [--device DEVICE] [--seed SEED]

SSL-in-one-epoch

optional arguments:
  -h, --help            show this help message and exit
  --similarity_loss_weight SIMILARITY_LOSS_WEIGHT
                        coefficient of cosine similarity (default: 200.0)
  --tcr_loss_weight TCR_LOSS_WEIGHT
                        coefficient of tcr (default: 1.0)
  --n_patches N_PATCHES
                        number of patches used in EMP-SSL (default: 100)
  --arch ARCH           network architecture (default: resnet18-cifar)
  --bs BS               batch size (default: 100)
  --lr LR               learning rate (default: 0.3)
  --eps EPS             eps for TCR (default: 0.2)
  --exp_name EXP_NAME   experiment name (default: default)
  --log_folder LOG_FOLDER
                        directory name (default: logs/EMP-SSL-Training)
  --dataset DATASET     data (default: cifar10)
  --n_epoch N_EPOCH     max number of epochs to finish (default: 2)
  --device DEVICE       device to use for training (default: cuda)
  --seed SEED           random seed
  --save_proj           include this flag to save patch embeddings and projections
  --pretrained_proj PRETRAINED_PROJ
                        use pre-trained weights for the projection network
```


## Additional Experimentation Insights

* The use of the `ReLU` activation within the feature encoder doesn't appear to significantly alter performance. Utilizing alternative activation layers, such as `Tanh`, yields comparable results. Interestingly, while in some cases a `ReLU` post a `BatchNorm1d` can carry nuanced implications, that doesn't seem to be the scenario here.

* The inclusion of `BatchNorm1d` is not paramount (by any means), both for the feature encoder and the projection network. Removing both batch norms drops top1 test performance for `n_patches=50` on CIFAR10 from 89.25% to 88.81%.

* The TCR loss displays a marked sensitivity concerning the number of patches used and the structure of the batch. As one might intuitively expect, when a batch inadvertently contains patches from the same image, there's a considerable dip in performance.

* LARS optimizer is critical. Without it, the top1 accuracy for `n_patches=50` on CIFAR10 is only 57.97%. This fact is worrisome without good theoretical understanding of what exactly leads to such performance gain when applying LARS. It seems one of the original goal was to find an SSL algorithm that doesn't depend drastically on details of the training regime.

* Loading pre-trained weights for projection network and keeping it frozen yields 87.35-86.85% top1 accuracy on CIFAR10 with 50 patches (over multiple runs). This is somewhat surprising, I would expect the network to converge to roughly the same quality of the representation. As we don't have supervision signal, this might mean that existing projection network induce bias in the learning process w.r.t. to the loss function used.

(This list will be updated with more experiments.)