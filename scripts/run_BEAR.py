"""
Run BEAR code with several datasets

python scripts/run_BEAR.py --D hall --d True

results will saved in results/BEAR/{dataset}...
"""

from scipy.io import loadmat
import skimage.io as skio
import time as _time
import torch
import os
import numpy as np
import sys
sys.path.append('.')
from bear.BEAR import BEAR
from bear.options import parse


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

    if dataset == "hall":
        path = "./data/hall1-200.mat"
        Y_res = torch.from_numpy(loadmat(path)["XO"]).to(torch.float32).permute(1, 0)
        data_shape = [176, 144, 200]
        Y = torch.reshape(Y_res.permute(1, 0), data_shape)

        if default:
            print("Set to default setting!")
            args.lr = 1e-3
            args.epoch = 80
            args.rank = 1
            args.batch_size = 64
            args.device = 'cuda:0'

    elif dataset == "demoMovie":
        path = "./data/demoMovie.tif"
        Y = torch.from_numpy(skio.imread(path).astype(float)).float().permute(1, 2, 0)
        print(Y.size())
        data_shape = list(Y.size())
        Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 1e-3
            args.epoch = 80
            args.rank = 2
            args.batch_size = 512
            args.device = 'cuda:0'

    elif dataset == "confocal_zebrafish":
        path = "./data/confocal_zebrafish.tif"
        Y = torch.from_numpy(skio.imread(path).astype(float)).float().permute(1, 2, 0)
        print(Y.size())
        data_shape = list(Y.size())
        Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 1e-3
            args.epoch = 80
            args.rank = 2
            args.batch_size = 60
            args.device = 'cuda:0'
            args.constrain_S = 'P'
            args.alpha = 1e3

    elif dataset == "confocal_zebrafish_2":
        path = "./data/confocal_zebrafish_2.tif"
        Y = torch.from_numpy(skio.imread(path).astype(float)).float().permute(1, 2, 0)
        print(Y.size())
        data_shape = list(Y.size())
        Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 1e-3
            args.epoch = 80
            args.rank = 2
            args.batch_size = 60
            args.device = 'cuda:0'
            args.constrain_S = 'P'
            args.alpha = 1e3

    print(f"""
    START TIME : {_time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime())}
    T_XY SHAPE : {Y_res.size()}
    CONFIG     : {args}""")

    [L_res, S_res, total_loss, _, time] = BEAR(Y_res, args)
    Y = Y_res.permute(1, 0).reshape(data_shape)
    del Y_res

    L = L_res.permute(1, 0).reshape(data_shape)
    S = S_res.permute(1, 0).reshape(data_shape)

    print(f"""Finished. Elapsed time : {time}, Total Loss : {total_loss:.3f}""")

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/BEAR"):
        os.mkdir("./results/BEAR")
    if not os.path.exists(f"./results/BEAR/{dataset}_Y"):
        os.mkdir(f"./results/BEAR/{dataset}_Y")
    if not os.path.exists(f"./results/BEAR/{dataset}_L"):
        os.mkdir(f"./results/BEAR/{dataset}_L")
    if not os.path.exists(f"./results/BEAR/{dataset}_S"):
        os.mkdir(f"./results/BEAR/{dataset}_S")


    for i in range(L_res.size(0)):
        if len(Y.size()) == 3:  # 2D video
            Y_frame = Y[:, :, i].numpy()
            L_frame = L[:, :, i].numpy()
            S_frame = S[:, :, i].numpy()

            with torch.no_grad():
                skio.imsave(f"./results/BEAR/{dataset}_Y/{i+1}.tif", Y_frame)
                skio.imsave(f"./results/BEAR/{dataset}_L/{i+1}.tif", L_frame)
                skio.imsave(f"./results/BEAR/{dataset}_S/{i+1}.tif", S_frame)

        if len(Y.size()) == 4:  # 3D video
            Y_frame = Y[:, :, :, i].permute(2, 0, 1).numpy()
            L_frame = L[:, :, :, i].permute(2, 0, 1).numpy()
            S_frame = S[:, :, :, i].permute(2, 0, 1).numpy()

            with torch.no_grad():
                skio.imsave(f"./results/BEAR/{dataset}_Y/{i+1}.tif", Y_frame)
                skio.imsave(f"./results/BEAR/{dataset}_L/{i+1}.tif", L_frame)
                skio.imsave(f"./results/BEAR/{dataset}_S/{i+1}.tif", S_frame)


if __name__ == "__main__":
    args = parse()
    run(args)