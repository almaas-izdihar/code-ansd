"""ANSD-minimal training entry.
[SUMBER: ANSD raw md §3, Eq7-11, Algorithm 2; recipe L234]

Per iteration:
  feats_orig,  z_orig  = model(x)                    # clean TEACHER (detached as target)
  feats_noise, z_noise = model(x, noise=True, lam)   # noisy STUDENT
  L_CE    = CE(z_ce, y)          z_ce = noise (default) or clean  [--ce_view; paper ambiguous]
  L_logit = T^2 KL(P_orig || P_noise)                                  (Eq9)
  L_feat  = sum_{l=2..L} ||F_noise_l - F_orig_l||^2                    (Eq10)
  L       = L_CE + alpha*L_logit + beta*L_feat                         (Eq11)
"""
from __future__ import print_function
import os, argparse, random, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from loader import custom_dataloader
from models.resnet import get_network
from loss.ansd_loss import logit_distillation, feature_distillation
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.dir_maker import DirectroyMaker
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults

C = Colorer.instance()


def parse_args():
    p = argparse.ArgumentParser(description='ANSD-minimal')
    # --- optimization / recipe [SUMBER: ANSD md L234] ---
    p.add_argument('--lr', default=0.1, type=float)
    p.add_argument('--lr_decay_rate', default=0.1, type=float)
    p.add_argument('--lr_decay_schedule', default=[30, 60, 90], nargs='*', type=int)
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--end_epoch', default=100, type=int)
    p.add_argument('--batch_size', type=int, default=128)
    # --- ANSD hyper-params ---
    p.add_argument('--ANSD', action='store_true', help='enable ANSD distillation (off = plain CE baseline)')
    p.add_argument('--lambda_noise', default=1.0, type=float, help='global noise intensity lambda (Eq8)')
    p.add_argument('--alpha', default=1.0, type=float, help='logit distillation weight')
    p.add_argument('--beta', default=1.0, type=float, help='feature distillation weight')
    p.add_argument('--T', default=1.0, type=float, help='distillation temperature (paper searches {1,2})')
    p.add_argument('--ce_view', default='noise', choices=['noise', 'clean'],
                   help='which view gets CE. PAPER AMBIGUOUS: Eq1=student(noise)+teacher-frozen '
                        '=> default noise; validate at Stage 0a, try clean if 76.56 misses.')
    # --- infra ---
    p.add_argument('--data_type', default='cifar100', type=str)
    p.add_argument('--data_path', default='../../Cifar100', type=str)
    p.add_argument('--classifier_type', default='ResNet18', type=str)
    p.add_argument('--experiments_dir', default='experiments', type=str)
    p.add_argument('--experiment_type', default='ansd', type=str)
    p.add_argument('--workers', default=8, type=int)
    p.add_argument('--saveckp_freq', default=100, type=int)
    p.add_argument('--rank', default=-1, type=int)
    p.add_argument('--gpu', default=0, type=int)
    p.add_argument('--seed', default=2024, type=int)
    # unused-but-referenced-by-shared-infra placeholders
    p.add_argument('--multiprocessing_distributed', action='store_true')
    args = p.parse_args()
    args.distributed = False
    args.EHSKD = args.ANSD  # alias: shared infra (dir_maker) expects this attr name
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for m in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= m else 1.
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / bs) for k in topk]


def train(net, criterion_CE, optimizer, scaler, loader, epoch, args):
    net.train()
    top1, losses = AverageMeter(), AverageMeter()
    m_ce, m_logit, m_feat = AverageMeter(), AverageMeter(), AverageMeter()  # component tracking
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            if args.ANSD:
                feats_orig, z_orig = net(inputs, noise=False)
                feats_noise, z_noise = net(inputs, noise=True, noise_lambda=args.lambda_noise)
                z_ce = z_noise if args.ce_view == 'noise' else z_orig
                l_ce = criterion_CE(z_ce, targets)
                l_logit = logit_distillation(z_orig, z_noise, args.T)
                l_feat = feature_distillation(feats_orig, feats_noise)
                loss = l_ce + args.alpha * l_logit + args.beta * l_feat
                logits_for_acc = z_orig  # clean view = deployed path
                # track raw (un-weighted) components → diagnose L_feat reduction / β scale, AMP nan
                m_ce.update(l_ce.item(), inputs.size(0))
                m_logit.update(float(l_logit), inputs.size(0))
                m_feat.update(float(l_feat), inputs.size(0))
            else:
                _, z = net(inputs, noise=False)
                loss = criterion_CE(z, targets)
                logits_for_acc = z
                m_ce.update(loss.item(), inputs.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), inputs.size(0))
        top1.update(accuracy(logits_for_acc.data, targets)[0].item(), inputs.size(0))
        progress_bar(epoch, batch_idx, len(loader), args,
                     'loss: {:.3f} | ce: {:.3f} | logit: {:.3f} | feat: {:.3f} | top1: {:.3f}'.format(
                         losses.avg, m_ce.avg, m_logit.avg, m_feat.avg, top1.avg))
    if is_main_process():
        logging.getLogger('train').info(
            '[Epoch {}] [ANSD {}] [train_loss {:.3f}] [L_CE {:.3f}] [L_logit {:.3f}] '
            '[L_feat {:.3f}] [train_top1 {:.3f}]'.format(
                epoch, args.ANSD, losses.avg, m_ce.avg, m_logit.avg, m_feat.avg, top1.avg))


def validate(net, criterion_CE, loader, epoch, args):
    net.eval()
    top1, top5, losses = AverageMeter(), AverageMeter(), AverageMeter()
    targets_list, confidences = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs, targets = batch[0].cuda(non_blocking=True), batch[1].cuda(non_blocking=True)
            _, outputs = net(inputs, noise=False)   # clean (deployed) path
            targets_list.extend(targets.cpu().numpy().tolist())
            confidences.extend(F.softmax(outputs, dim=1).cpu().numpy().tolist())
            loss = criterion_CE(outputs, targets)
            e1, e5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(e1.item(), inputs.size(0)); top5.update(e5.item(), inputs.size(0))
            progress_bar(epoch, batch_idx, len(loader), args,
                         'val_loss: {:.3f} | val_top1: {:.3f}'.format(losses.avg, top1.avg))
    if is_main_process():
        ece, aurc, eaurc = metric_ece_aurc_eaurc(confidences, targets_list, bin_size=0.1)
        logging.getLogger('val').info(
            '[Epoch {}] [val_loss {:.3f}] [val_top1 {:.3f}] [val_top5 {:.3f}] [ECE {:.3f}]'.format(
                epoch, losses.avg, top1.avg, top5.avg, ece))
    return top1.avg


def main():
    args = parse_args()
    dirs = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True).experiments_dir_maker(args)
    model_dir, log_dir, config_dir = dirs
    paser_config_save(args, config_dir)
    set_logging_defaults(log_dir, args)
    torch.cuda.set_device(args.gpu)
    net = get_network(args).cuda(args.gpu)
    train_loader, valid_loader, _ = custom_dataloader.dataloader(args)
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scaler = GradScaler()
    best_acc = 0
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train(net, criterion_CE, optimizer, scaler, train_loader, epoch, args)
        acc = validate(net, criterion_CE, valid_loader, epoch, args)
        if acc > best_acc:
            best_acc = acc
            save_on_master({'net': net.state_dict(), 'epoch': epoch, 'best_acc': best_acc},
                           os.path.join(model_dir, 'checkpoint_best.pth'))
    print(C.green('[!] Done. best_acc = {:.3f}'.format(best_acc)))


if __name__ == '__main__':
    main()
