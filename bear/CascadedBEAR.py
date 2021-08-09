import argparse
import time
import torch
from .model import BEAR_WWTY


def Cascaded_BEAR(data: torch.Tensor, args: argparse.Namespace):
    print(args)
    lr = args.lr
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device

    sparsity = args.sparsity
    k = args.k

    model = BEAR_WWTY(data.size(1), k=1).to(device)
    second_model = BEAR_WWTY(data.size(1), k=k).to(device)

    second_model.clamper()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    optim2 = torch.optim.Adam(second_model.parameters(), lr=lr, weight_decay=0)

    batch_iter = data.size(0) // batch_size

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_ev.record()
    # Train
    for i in range(epoch):
        total_loss = 0.0
        for j in range(batch_iter):
            r_index = torch.randperm(data.size(0))[:batch_size].to(device)
            batch = data[r_index, :].to(device)
            L = model(batch)
            S = batch - L
            S = S * (S >= 0)
            loss_a = (sparsity * torch.norm(S, p=2) + torch.norm((batch - L - S), p='fro')) / 2
            nmf_recon = second_model(S)

            loss_b = torch.norm((S - nmf_recon), p='fro')
            loss = 2 * loss_a + loss_b
            if i % 200 == 0 and j == 0:
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{t}] [{i}/{epoch}], loss_a : {loss_a.item():.3f}, loss_b : {loss_b.item():.3f} \
loss {loss.item():.3f}")

            optim.zero_grad()
            optim2.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optim.step()
            optim2.step()

            second_model.clamper()

    # Test
    test_split = data.split(batch_size, dim=0)
    with torch.no_grad():
        for i, batch in enumerate(test_split):
            batch = batch.to(device)
            L = model(batch)
            S = batch - L
            S = S * (S > 0)

            if i == 0:
                total_L = L.cpu()
                total_S = S.cpu()
            else:
                total_L = torch.cat((total_L, L.cpu()), dim=0)
                total_S = torch.cat((total_S, S.cpu()), dim=0)

    end_ev.record()
    torch.cuda.synchronize()

    del batch, L, S

    return total_L, total_S, total_loss, model, second_model, start_ev.elapsed_time(end_ev)