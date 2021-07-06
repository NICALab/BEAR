import argparse
import torch
from bear.BEAR import BEAR
import warnings
warnings.simplefilter("ignore")
import psutil


def Greedy_BEAR(data: torch.Tensor, args: argparse.Namespace) -> list:
    # Used for Greedy rank estimation BEAR
    '''
    if "rank_step" in config.keys(): rank_step = config["rank_step"]
    else: rank_step = 1
    if "max_rank" in config.keys(): max_rank = config["max_rank"]
    else: max_rank = min(data.size())
    if "zero_rank" in config.keys(): zero_rank = config["zero_rank"]
    else: zero_rank = True
    if "lamda" in config.keys(): lamda = config["lamda"]
    else: lamda = 1
    if "verbose" in config.keys(): verbose = config["verbose"]
    else: verbose = True
    '''
    rank_step = args.rank_step
    max_rank = args.max_rank
    zero_rank = args.zero_rank
    lamda = args.lamda
    verbose = args.verbose

    if zero_rank:
        rank_iter = (max_rank // rank_step)
    else:
        rank_iter = (max_rank // rank_step) - 1

    memory_usage_dict = dict(psutil.virtual_memory()._asdict())
    small_memory = (memory_usage_dict['available'] - (4 * data.numel()) * (5 + 1)) < 0

    if verbose:
        print(f"SMALL MEMORY MODE : {small_memory}")


    total_time = 0.0
    prev_L, prev_S = None, None
    model = None
    prev_loss = float("inf")

    for rank_it in range(rank_iter):
        if zero_rank:
            args.rank = rank_it * rank_step
        else:
            args.rank = (rank_it + 1) * rank_step

        if rank_it == 0:
            weight = None
        else:
            assert args.bear_type == "wwt", "Implemented only for 'wwt' bear_type!"
            weight = model.ln.weight

        [L_res, S_res, total_loss, model, time] = BEAR(data, args, weight)
        if small_memory:
            del L_res, S_res
            best_rank = 0.0
        if total_loss + args.rank * lamda >= prev_loss:
            if not small_memory:
                L_res = prev_L
                S_res = prev_S
            break
        else:
            prev_loss = total_loss + args.rank * lamda
            if not small_memory:
                prev_L = L_res
                prev_S = S_res
            else:
                best_rank = args.rank
        total_time += time

    if small_memory:
        args.rank = best_rank
        [L_res, S_res, total_loss, model, time] = BEAR(data, args, None)
        total_time += time

    if verbose:
        print(f"Estimated rank : {(rank_it-1) * rank_step}")

    return L_res, S_res, total_loss, model, total_time
