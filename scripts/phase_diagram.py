"""
Script for building phase diagram (Fig. 3) in the paper
This is PyTorch version based on the public code in the Go Decomposition.
Code is linked as "GreBsmo Code" in the News section of the website below.
    https://sites.google.com/site/godecomposition/
    (Last visited on 2021. 07. 06)

    GoDec_test.m is a test script generating phase diagram using Greedy GoDec (GreBsmo).

For safe reproducibility, .zip file the author uploaded is also in this repository without any modification.

Phase diagrams of PCP, IALM, and Greedy GoDec are generated with that matlab code.

In BEAR directory, run as follows
(--D and --d options are not used in this code.)
'''
python scripts/phase_diagram.py --D None --d None
'''
"""

import math
import numpy as np
from scipy import ndimage
import skimage.io as skio
import torch
import os
import sys
sys.path.append('.')
from bear.GreedyBEAR import Greedy_BEAR
from bear.options import parse


def one_phase(args, m, n, rank, card, Smag, Gfac):
    # Generate synthetic data
    mn = min(m, n)
    rank = round(rank * mn)
    L = (torch.randn([m, rank]) / math.sqrt(mn)) @ torch.randn([rank, n])

    card = round(card*m*n)
    supp = torch.randperm(m*n)
    supp = supp[0:card]
    p = torch.ones(card) * 0.5
    s = torch.bernoulli(p)
    c = torch.ones(card) * -1
    s = torch.where(s == 0, c, s)
    s = Smag * s
    S = torch.zeros(m*n)
    S[supp] = s
    S = S.reshape(m, n)
    # G = (Gfac/math.sqrt(mn)) * torch.randn([m, n])
    X = L + S # + G

    normL = torch.norm(L).item()
    # normS = torch.norm(S).item()
    # normG = torch.norm(G).item()

    (L_hat, _, _, _, total_time) = Greedy_BEAR(X, args)
    rel_err = torch.norm(L_hat.cpu() - L).item() / (normL + 1e-8)

    return rel_err, total_time


if __name__ == "__main__":
    args = parse(mode='greedy')
    grid  = 10
    trial = 6
    # we will do 6 trials and save time and error results from
    # 5 trials except first. (discard first experiment)

    _ = args.D
    _ = args.d
    # configuration setting
    args.lr = 3e-3
    args.alpha = 1
    args.epoch = 50
    args.batch_size = 1000
    args.device = 'cuda:0'
    args.lamda = 400
    args.max_rank = 500
    args.rank_step = 10

    time  = torch.zeros(grid, grid, trial)
    error = torch.zeros(grid, grid, trial)

    for i in range(grid):
        for j in range(grid):
            for k in range(trial):
                r = (0.4 / grid) * (i + 1)
                s = (0.5 / grid) * (j + 1)
                (rel_err, one_time) = one_phase(args, 1000, 1000, r, s, 0.1, 0)
                print(i, j, k, rel_err, one_time)
                time[i, j, k] = one_time
                error[i, j, k] = rel_err


    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/phase_diagram"):
        os.mkdir("./results/phase_diagram")

    assert trial >= 2, "Set trial as bigger than 1!"
    error_tif = error[:, :, 1:].permute(2, 0, 1)
    error_avg = np.expand_dims(ndimage.rotate(error_tif.mean(dim=0), 90), axis=0)

    # Time measured from BEAR is in (ms) unit.
    time_tif = time[:, :, 1:].permute(2, 0, 1)
    time_tif /= 1000
    time_avg = np.expand_dims(ndimage.rotate(time_tif.mean(dim=0), 90), axis=0)
    print(time_avg.shape, time_tif.size())

    # Save results
    skio.imsave("./results/phase_diagram/error.tif", error_tif.numpy())
    skio.imsave("./results/phase_diagram/time.tif", time_tif.numpy())
    skio.imsave("./results/phase_diagram/error_avg.tif", error_avg)
    skio.imsave("./results/phase_diagram/time_avg.tif", time_avg)
