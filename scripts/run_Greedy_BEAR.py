"""
Run GreedyBEAR code with several datasets

python scripts/run_Greedy_BEAR.py --D zebrafish_150 --d True

results will saved in results/Greedy_BEAR/{dataset}...
"""

import mat73
import numpy as np
import skimage.io as skio
import time as _time
import torch
import os
import sys
sys.path.append('.')
from bear.GreedyBEAR import Greedy_BEAR
from bear.options import parse
import psutil


def run(args):
    """
    Data description
    Y          : torch.Tensor; [x, y, t] or [x, y, z, t]
    Y_res      : torch.Tensor; [t, xy]   or [t, xyz] (reshape and transpose)
    data_shape : list; [x, y, t] or [x, y, z, t]

    [L_res, S_res, total_loss, model, time] = train_test_BEAR(Y_res, config)
    L_res      : torch.Tensor; [t, xy]   or [t, xyz]
    L          : torch.Tensor; [x, y, t] or [x, y, z, t] (transpose and reshape)
    S_res      : torch.Tensor; [t, xy]   or [t, xyz]
    S          : torch.Tensor; [x, y, t] or [x, y, z, t] (transpose and reshape)
    """
    dataset = args.D
    default = args.d

    if dataset == "zebrafish_150":
        path = "./data/zebrafish_150.mat"
        raw = mat73.loadmat(path, use_attrdict=True)["rawVideo"]
        Y = torch.from_numpy(np.float32(raw))  # [270, 480, 41, 150]
        Y /= Y.max()
        # print(Y.size())
        data_shape = [270, 480, 41, 150]
        Y_res = torch.reshape(Y, [np.prod(data_shape[0:3]), data_shape[3]]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 5e-5
            args.epoch = 45
            args.batch_size = 64
            args.device = 'cuda:0'
            # for greedy bear
            args.lamda = 400
            args.max_rank = 5
            args.rank_step = 1

    elif dataset == "zebrafish_1000":
        path = "./data/zebrafish_1000.mat"
        raw = mat73.loadmat(path, use_attrdict=True)["rawVideo"]
        Y = torch.from_numpy(np.float32(raw))
        del raw
        Y /= Y.max()
        # print(Y.size())
        data_shape = [270, 480, 41, 1000]
        Y_res = torch.reshape(Y, [np.prod(data_shape[0:3]), data_shape[3]]).permute(1, 0)
        del Y  # Need to recover. Intended to use SMALL_MEMORY mode

        if default:
            print("Set to default setting!")
            args.lr = 5e-5
            args.epoch = 45 # 45
            args.batch_size = 64
            args.device = 'cuda:0'

            args.lamda = 400
            args.max_rank = 2
            args.rank_step = 1
            args.zero_rank = False


    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    small_memory = (memory_usage_dict['available'] - (4 * Y_res.numel()) * (5 + 1)) < 0
    print(f"SMALL MEMORY MODE : {small_memory}")

    print("START TIME : " + _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime()) + \
    f"\nT_XY SHAPE : {Y_res.size()}" + \
    f"\nCONFIG     : {args}")

    [L_res, S_res, total_loss, _, time] = Greedy_BEAR(Y_res, args)
    
    if small_memory:
        Y = Y_res.permute(1, 0).reshape(data_shape)
        del Y_res
    L = L_res.permute(1, 0).reshape(data_shape)
    S = S_res.permute(1, 0).reshape(data_shape)

    print(f"""Finished. Elapsed time : {time}, Total Loss : {total_loss:.3f}""")

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/Greedy_BEAR"):
        os.mkdir("./results/Greedy_BEAR")
    if not os.path.exists(f"./results/Greedy_BEAR/{dataset}_Y"):
        os.mkdir(f"./results/Greedy_BEAR/{dataset}_Y")
    if not os.path.exists(f"./results/Greedy_BEAR/{dataset}_L"):
        os.mkdir(f"./results/Greedy_BEAR/{dataset}_L")
    if not os.path.exists(f"./results/Greedy_BEAR/{dataset}_S"):
        os.mkdir(f"./results/Greedy_BEAR/{dataset}_S")

    for i in range(L_res.size(0)):
        if len(Y.size()) == 3:  # 2D video
            Y_frame = Y[:, :, i].numpy()
            L_frame = L[:, :, i].numpy()
            S_frame = S[:, :, i].numpy()

            with torch.no_grad():
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_Y/{i+1}.tif", Y_frame)
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_L/{i+1}.tif", L_frame)
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_S/{i+1}.tif", S_frame)

        elif len(Y.size()) == 4:  # 3D video
            Y_frame = Y[:, :, 5, i]
            L_frame = L[:, :, 5, i]
            S_frame = S[:, :, 5, i]

            Y_f = Y[..., i].permute(2, 0, 1)
            L_f = L[..., i].permute(2, 0, 1)
            S_f = S[..., i].permute(2, 0, 1)
            S_f = torch.nn.functional.relu(S_f)
            with torch.no_grad():
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_Y/{i + 1}.tif", Y_f.numpy())
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_L/{i + 1}.tif", L_f.numpy())
                skio.imsave(f"./results/Greedy_BEAR/{dataset}_S/{i + 1}.tif", S_f.numpy())


if __name__ == "__main__":
    args = parse(mode='greedy')
    run(args)
