import argparse
import time
import torch
from .model import BEAR_WWTY, BEAR_ABY
import warnings
# warnings.simplefilter("ignore")


def BEAR(data: torch.Tensor, args: argparse.Namespace, weight=None):
    """
    Run (train and test) BEAR

    data shape : [time, pixel] in torch.Tensor

    weight is for Greedy BEAR, for weight initialization.
    """
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device
    bear_type = args.bear_type
    alpha = args.alpha
    early_stop = args.early_stop

    rank = args.rank
    constrain_L = args.constrain_L
    constrain_S = args.constrain_S

    verbose = args.verbose
    tensorboard = args.tensorboard

    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    if bear_type == "wwt":
        model = BEAR_WWTY(data.size(1), k=rank).to(device)
    elif bear_type == "aby":
        warnings.warn("W W^T Y modeling is better.", DeprecationWarning)
        model = BEAR_ABY(data.size(1), k=rank).to(device)

    if weight is not None:
        if bear_type == "wwt":
            with torch.no_grad():
                if model.ln.weight.size(0) >= weight.size(0):

                    model.ln.weight[:weight.size(0), :] = weight
                else:
                    print(weight.size(), model.ln.weight.size())
                    print(model.ln.weight.size(0))
                    print(weight[:0, :])
                    # raise RuntimeError
                    model.ln.weight = weight[:model.ln.weight.size(0), :]
        else:
            warnings.warn("Not implemented for ABY modeling", UserWarning)

    if constrain_L == "NMF":
        model.clamper()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    batch_iter = data.size(0) // batch_size

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    error = torch.zeros(epoch)
    torch.cuda.synchronize()
    start_ev.record()
    # Train
    total_loss = None
    for i in range(epoch):
        total_loss = 0.0
        for _ in range(batch_iter):
            r_index = torch.randperm(data.size(0))[:batch_size].to(device)
            batch = data[r_index, :].to(device)
            L = model(batch)
            if constrain_L == "N":
                L = L * (L <= 0)
            elif constrain_L == "P":
                L = L * (L >= 0)

            S = batch - L
            if constrain_S == "N":
                S = S * (S <= 0)
            elif constrain_S == "P":
                S = S * (S >= 0)

            if constrain_S  is None:
                loss = torch.norm(S, p=1)
            else:
                loss = torch.norm(S, p=1) + alpha * torch.norm(batch - (L + S), p='fro')

            optim.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optim.step()

        if early_stop:
            error[i] = total_loss
            if i > epoch/5 and error[i-int(epoch/10)+1:i-1].mean() < 1.005 * error[i]:
                if verbose: print(f"Early stop, {i}th epoch")
                break

        if tensorboard:
            writer.add_scalar('total_loss', total_loss, i)

        if epoch > 4:
            if (i % (epoch // 3) == 0 or i == epoch - 1) and verbose:
                print(f"""[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] EPOCH : [{i}/{epoch}] : {(total_loss):.4f}""")

    # Inference
    test_split = data.split(batch_size, dim=0)
    del data

    L_s, S_s = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_split):
            batch = batch.to(device)
            L = model(batch)
            if constrain_L == "N":
                L = L * (L <= 0)
            elif constrain_L == "P":
                L = L * (L >= 0)

            S = batch - L
            if constrain_S == "N":
                S = S * (S < 0)
            elif constrain_S == "P":
                S = S * (S > 0)

            L = L.to("cpu")
            S = S.to("cpu")

            L_s.append(L)
            S_s.append(S)

        total_L = torch.cat(L_s, dim=0)
        total_S = torch.cat(S_s, dim=0)

    end_ev.record()
    torch.cuda.synchronize()

    del batch, L, S

    if total_loss == None:
        total_loss = 0.0

    return total_L, total_S, total_loss, model, start_ev.elapsed_time(end_ev)