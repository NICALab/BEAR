import argparse


def parse(mode=None):
    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument('--D', type=str, required=True,
                    help='dataset name. And result will saved \
                    in the folder name of this.')
    ap.add_argument('--d', type=bool, required=True,
                    help='Turn on when you want to use default arguments.')

    # Train specification
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
    ap.add_argument('--epoch', type=int, default=50,
                    help='Epochs')
    ap.add_argument('--batch_size', type=int, default=64,
                    help='batch size, for mini-batch training.')
    ap.add_argument('--device', type=str, default='cuda:0',
                    help='CPU or GPU. \'cpu\' to use cpu, \
                    \'cuda:0\' to use gpu. If you have multiple gpu, \
                    can change gpu device such as \'cuda:1\', etc. ')
    ap.add_argument('--bear_type', type=str, default='wwt',
                    help='\'wwt\' to use L=W@W^T@Y setting. \
                    \'aby\' to use L=A@B@Y setting.\
                    wwt setting is preferable. For more detail, \
                    go to our paper.')
    ap.add_argument('--alpha', type=float, default=1e-3,
                    help='Loss = ||S||_1 + alpha * ||Y-L-S||_fro')
    ap.add_argument('--early_stop', type=bool, default=True,
                    help='True for early stopping at convergence. \
                    For best accurate result, set False.')

    # RPCA specification
    ap.add_argument('--rank', type=int, default=10,
                    help='rank of low-rank matrix')
    ap.add_argument('--constrain_L', type=str, default=None,
                    help='Constrain on low-rank matrix L. \
                    P if positive, N if negative, \
                    NMF if you want Non-negative Matrix Factorization')
    ap.add_argument('--constrain_S', type=str, default=None,
                    help='Constrain on sparse matrix S. \
                    P if positive, N if negative.')

    # Additional things
    ap.add_argument('--verbose', type=bool, default=False,
                    help='Print stuffs. (Time/Epoch/loss)')
    ap.add_argument('--tensorboard', type=bool, default=False,
                    help='Set True to use tensorboard, visualization tool. \
                    It will show loss in real time.')

    if mode == None:
        pass
    elif mode == 'greedy':
        # For Greedy BEAR
        ap.add_argument('--rank_step', type=int, default=1,
                        help='step size of rank increment')
        ap.add_argument('--max_rank', type=int, default=None,
                        help='Maximum rank size')
        ap.add_argument('--zero_rank', type=bool, default=True,
                        help='True if low-rank matrix can be 0 rank, \
                        i.e. zero matrix.')
        ap.add_argument('--lamda', type=float, default=1,
                        help='weight in rank between rank and sparsity.')
    elif mode == 'cascaded':
        # For Cascaded BEAR
        ap.add_argument('--sparsity', type=float, default=3,
                        help='sparsity weight for first BEAR of cascaded BEAR')
        ap.add_argument('--k', type=int, default=8,
                        help='number of spatial & temporal components for cascaded BEAR')
    else:
        raise AssertionError


    args = ap.parse_args()

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

        if isinstance(vars(args)[arg], list):
            vars(args)[arg] = [int(s) for s in vars(args)[arg]]

    return args