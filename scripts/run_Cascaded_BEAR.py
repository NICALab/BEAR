"""
Run BEAR code with several datasets

In BEAR directory, run as follows
python scripts/run_Cascaded_BEAR.py --D demoMovie --d True
"""

import os
import numpy as np
import skimage.io as skio
from tifffile import imsave
import torch
import sys
sys.path.append('.')
from bear.CascadedBEAR import Cascaded_BEAR
from bear.options import parse


def run(args):
    """
    Data Description

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

    if dataset == "demoMovie":
        path = "./data/demoMovie.tif"
        Y = torch.from_numpy(skio.imread(path).astype(float)).float().permute(1, 2, 0)
        print(Y.size())
        data_shape = list(Y.size())
        Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 2e-4
            args.epoch = 5000
            args.batch_size = 512
            args.device = 'cuda:0'
            
            # For cascaded BEAR
            args.sparsity = 3
            args.k = 8

    elif dataset == "spinning_confocal":
        path = "./data/nls_zebrafish.tif"
        Y = torch.from_numpy(skio.imread(path).astype(float)).float().permute(1, 2, 0)
        print(Y.size())
        Y /= Y.max()
        data_shape = list(Y.size())
        Y_res = Y.reshape(np.prod(data_shape[0:2]), data_shape[2]).permute(1, 0)

        if default:
            print("Set to default setting!")
            args.lr = 1e-4
            args.epoch = 1000
            args.batch_size = 64
            args.device = 'cuda:0'

            # For cascaded BEAR
            args.sparsity = 3
            args.k = 8

    [L_res, S_res, total_loss, _, second_model, time] = Cascaded_BEAR(Y_res, args)

    L = L_res.permute(1, 0).reshape(data_shape)
    S = S_res.permute(1, 0).reshape(data_shape)

    print(f"""Finished. Elapsed time : {time}, Total Loss : {total_loss:.3f}""")

    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./results/Cascaded_BEAR"):
        os.mkdir("./results/Cascaded_BEAR")
    if not os.path.exists(f"./results/Cascaded_BEAR/{dataset}"):
        os.mkdir(f"./results/Cascaded_BEAR/{dataset}")

    imsave(f"./results/Cascaded_BEAR/{dataset}/Y.tif", Y.permute(2, 0, 1).cpu().numpy())
    imsave(f"./results/Cascaded_BEAR/{dataset}/L.tif", L.permute(2, 0, 1).cpu().numpy())
    imsave(f"./results/Cascaded_BEAR/{dataset}/S.tif", S.permute(2, 0, 1).cpu().numpy())

    with torch.no_grad():
        second_model = second_model.cpu()
        spatial_mask_res = second_model.ln.weight
        spatial_mask_res /= spatial_mask_res.max()

        temporal_sig_res = second_model.ln(S_res)

        spatial_mask = spatial_mask_res.reshape([spatial_mask_res.size(0), *data_shape[0:2]])

        temporal_sig = temporal_sig_res.permute(1, 0)
        print(spatial_mask.size(), temporal_sig.size())

        imsave(f"./results/Cascaded_BEAR/{dataset}/spatial_footprints.tif", spatial_mask.cpu().detach().numpy())

        # torch.save(temporal_sig, f"./results/Cascaded_BEAR/{dataset}/temporal_footprints.pt")
        imsave(f"./results/Cascaded_BEAR/{dataset}/temporal_footprints.tif", temporal_sig.cpu().numpy())


if __name__ == "__main__":
    args = parse()
    run(args)


