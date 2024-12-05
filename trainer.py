from __future__ import print_function, absolute_import
import time
from tqdm import trange

from loss.cross_entropy import CrossEntropyLabelSmooth
from utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, model):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth().cuda()

    def train(self, epoch, data_loader_train, optimizer, lr_scheduler, training_phase, train_iters=0):
        self.model.train()
        batch_time = AverageMeter()
        losses_ce = AverageMeter()

        end = time.time()
        for _ in trange(train_iters):
            inputs, targets = self._parse_data(data_loader_train.next())
            cls_out = self.model(inputs, domain=training_phase)

            loss_ce = self.criterion_ce(cls_out, targets)
            losses_ce.update(loss_ce.item())

            optimizer.zero_grad(set_to_none=True)
            loss_ce.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        if epoch % 10 == 0:
            print('Epoch: {} ce {:.3f} '.format(epoch, losses_ce.avg))
        lr_scheduler.step()

    def _parse_data(self, inputs):
        imgs, _, pids, cids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
