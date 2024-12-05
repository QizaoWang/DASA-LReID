import torch
from tqdm import tqdm


def eval_func(epoch, evaluator, model, test_loader, name):
    evaluator.reset()
    model.eval()
    device = 'cuda'
    pid_list = []
    for n_iter, (imgs, fnames, pids, cids, domains) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            pid_list.append(pids)
            imgs = imgs.to(device)
            cids = cids.to(device)
            feat = model(imgs, domain=int(domains[0]))
            evaluator.update((feat, pids, cids))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print(name, end=' ')
    print("Epoch {}".format(epoch), end=' ')
    print("mAP: {:.1%}".format(mAP), end=' ')
    for r in [1]:
        print("R{:<2}: {:.1%}".format(r, cmc[r - 1]), end=' ')
    print()

    torch.cuda.empty_cache()
    return cmc, mAP
