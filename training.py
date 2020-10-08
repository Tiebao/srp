# %%
import sys
import argparse
import datetime
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import XrayDateset
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import XrayModel
from tqdm import tqdm
# %%
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
sh = logging.StreamHandler()
log.addHandler(sh)


class XrayTrainApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='工作进程数量',
                            default=8, type=int)
        parser.add_argument('--batch-size', help='每一批次的样本数',
                            default=4, type=int)
        parser.add_argument('--epochs', help='迭代次数',
                            default=1, type=int)

        self.cli_args = parser.parse_args(sys_argv)
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()
        self.total_train_samples_count = 0

    def init_train_dataloader(self):
        train_dataset = XrayDateset(is_val=0, val_stride=6)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dataloader

    def init_val_dataloader(self):
        val_dataset = XrayDateset(is_val=1, val_stride=6)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dataloader

    def init_model(self):
        model = XrayModel()
        if self.use_cuda:
            log.info("Using CUDA.")
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return optimizer

    def train(self, epoch_index, train_dl):
        self.model.train()

        metrics = {}
        metrics_loss = torch.zeros(len(train_dl.dataset), device=self.device)
        metrics_label = torch.zeros(len(train_dl.dataset), device=self.device)
        metrics_pred = torch.zeros(
            len(train_dl.dataset), 11, device=self.device)
        metrics['loss'] = metrics_loss
        metrics['label'] = metrics_label
        metrics['pred'] = metrics_pred

        for batch_index, batch_tuple in tqdm(enumerate(train_dl), total=len(train_dl)):
            self.optimizer.zero_grad()
            loss = self.compute_batch_loss(
                batch_index,
                batch_tuple,
                train_dl.batch_size,
                metrics)
            loss.backward()
            self.optimizer.step()

        self.total_train_samples_count += len(train_dl.dataset)

        metrics['loss'] = metrics['loss'].to('cpu')
        metrics['label'] = metrics['label'].to('cpu')
        metrics['pred'] = metrics['pred'].to('cpu')
        return metrics

    def validate(self, epoch_index, val_dl):
        with torch.no_grad():
            self.model.eval()

            metrics = {}
            metrics_loss = torch.zeros(len(val_dl.dataset), device=self.device)
            metrics_label = torch.zeros(
                len(val_dl.dataset), device=self.device)
            metrics_pred = torch.zeros(
                len(val_dl.dataset), 11, device=self.device)
            metrics['loss'] = metrics_loss
            metrics['label'] = metrics_label
            metrics['pred'] = metrics_pred

            for batch_index, batch_tuple in tqdm(enumerate(val_dl), total=len(val_dl)):
                self.compute_batch_loss(
                    batch_index,
                    batch_tuple,
                    val_dl.batch_size,
                    metrics)

        metrics['loss'] = metrics['loss'].to('cpu')
        metrics['label'] = metrics['label'].to('cpu')
        metrics['pred'] = metrics['pred'].to('cpu')
        return metrics

    def compute_batch_loss(self, batch_index, batch_tuple, batch_size, metrics):
        input_batch, label_batch, candidate_batch = batch_tuple
        slice_batch_gpu = input_batch.to(self.device, non_blocking=True)
        label_batch_gpu = label_batch.to(self.device, non_blocking=True)

        logits_batch_gpu, probability_batch_gpu = self.model(slice_batch_gpu)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_gpu = loss_func(logits_batch_gpu, label_batch_gpu)

        # 记录每个样本的评测数据
        start_index = batch_index * batch_size
        # label_batch.size(0) 主要是考虑到最后一个批次大小可能不等于batch_size
        end_index = start_index + label_batch.size(0)

        metrics['label'][start_index:end_index] = label_batch_gpu.detach()
        metrics['loss'][start_index:end_index] = loss_gpu.detach()
        metrics['pred'][start_index:end_index] = probability_batch_gpu.detach()

        return loss_gpu.mean()

    def log_metric(self, epoch_index, mode, metrics, classification_threshold=0.5):
        normal_label_mask = metrics['label'] == 0
        lighter_label_mask = metrics['label'] == 1
        pressure_label_mask = metrics['label'] == 2
        knife_label_mask = metrics['label'] == 3
        scissors_label_mask = metrics['label'] == 4
        powerbank_label_mask = metrics['label'] == 5
        zippooil_label_mask = metrics['label'] == 6
        handcuffs_label_mask = metrics['label'] == 7
        slingshot_label_mask = metrics['label'] == 8
        firecrackers_label_mask = metrics['label'] == 9
        nailpolish_label_mask = metrics['label'] == 10

        # 如果预测的概率全部小于threshold，则分类为normal
        normal_pred_mask = (metrics['pred'][:, 0] >= classification_threshold) + (
            metrics['pred'] < classification_threshold).all(dim=1)
        lighter_pred_mask = metrics['pred'][:, 1] >= classification_threshold
        pressure_pred_mask = metrics['pred'][:, 2] >= classification_threshold
        knife_pred_mask = metrics['pred'][:, 3] >= classification_threshold
        scissors_pred_mask = metrics['pred'][:, 4] >= classification_threshold
        powerbank_pred_mask = metrics['pred'][:, 5] >= classification_threshold
        zippooil_pred_mask = metrics['pred'][:, 6] >= classification_threshold
        handcuffs_pred_mask = metrics['pred'][:, 7] >= classification_threshold
        slingshot_pred_mask = metrics['pred'][:, 8] >= classification_threshold
        firecrackers_pred_mask = metrics['pred'][:,
                                                 9] >= classification_threshold
        nailpolish_pred_mask = metrics['pred'][:,
                                               10] >= classification_threshold

        normal_count = int(normal_label_mask.sum())
        lighter_count = int(lighter_label_mask.sum())
        pressure_count = int(pressure_label_mask.sum())
        knife_count = int(knife_label_mask.sum())
        scissors_count = int(scissors_label_mask.sum())
        powerbank_count = int(powerbank_label_mask.sum())
        zippooil_count = int(zippooil_label_mask.sum())
        handcuffs_count = int(handcuffs_label_mask.sum())
        slingshot_count = int(slingshot_label_mask.sum())
        firecrackers_count = int(firecrackers_label_mask.sum())
        nailpolish_count = int(nailpolish_label_mask.sum())

        normal_pred_count = int(normal_pred_mask.sum())
        lighter_pred_count = int(lighter_pred_mask.sum())
        pressure_pred_count = int(pressure_pred_mask.sum())
        knife_pred_count = int(knife_pred_mask.sum())
        scissors_pred_count = int(scissors_pred_mask.sum())
        powerbank_pred_count = int(powerbank_pred_mask.sum())
        zippooil_pred_count = int(zippooil_pred_mask.sum())
        handcuffs_pred_count = int(handcuffs_pred_mask.sum())
        slingshot_pred_count = int(slingshot_pred_mask.sum())
        firecrackers_pred_count = int(firecrackers_pred_mask.sum())
        nailpolish_pred_count = int(nailpolish_pred_mask.sum())

        normal_correct = int((normal_label_mask & normal_pred_mask).sum())
        lighter_correct = int((lighter_label_mask & lighter_pred_mask).sum())
        pressure_correct = int(
            (pressure_label_mask & pressure_pred_mask).sum())
        knife_correct = int((knife_label_mask & knife_pred_mask).sum())
        scissors_correct = int(
            (scissors_label_mask & scissors_pred_mask).sum())
        powerbank_correct = int(
            (powerbank_label_mask & powerbank_pred_mask).sum())
        zippooil_correct = int(
            (zippooil_label_mask & zippooil_pred_mask).sum())
        handcuffs_correct = int(
            (handcuffs_label_mask & handcuffs_pred_mask).sum())
        slingshot_correct = int(
            (slingshot_label_mask & slingshot_pred_mask).sum())
        firecrackers_correct = int(
            (firecrackers_label_mask & firecrackers_pred_mask).sum())
        nailpolish_correct = int(
            (nailpolish_label_mask & nailpolish_pred_mask).sum())

        metrics['correct/all'] = (normal_correct + lighter_correct
                                    + pressure_correct + knife_correct
                                    + scissors_correct + powerbank_correct
                                    + zippooil_correct + handcuffs_correct
                                    + slingshot_correct + firecrackers_correct
                                    + nailpolish_correct) / np.float32(metrics['label'].shape[0]) * 100
        metrics['loss/all'] = metrics['loss'].mean()
        metrics['normal_precision'] = normal_correct / \
            np.float32(normal_pred_count) * 100
        metrics['lighter_precision'] = lighter_correct / \
            np.float32(lighter_pred_count) * 100
        metrics['pressure_precision'] = pressure_correct / \
            np.float32(pressure_pred_count) * 100
        metrics['knife_precision'] = knife_correct / \
            np.float32(knife_pred_count) * 100
        metrics['scissors_precision'] = scissors_correct / \
            np.float32(scissors_pred_count) * 100
        metrics['powerbank_precision'] = powerbank_correct / \
            np.float32(powerbank_pred_count) * 100
        metrics['zippooil_precision'] = zippooil_correct / \
            np.float32(zippooil_pred_count) * 100
        metrics['handcuffs_precision'] = handcuffs_correct / \
            np.float32(handcuffs_pred_count) * 100
        metrics['slingshot_precision'] = slingshot_correct / \
            np.float32(slingshot_pred_count) * 100
        metrics['firecrackers_precision'] = firecrackers_correct / \
            np.float32(firecrackers_pred_count) * 100
        metrics['nailpolish_precision'] = nailpolish_correct / \
            np.float32(nailpolish_pred_count) * 100

        log.info(("Epoch{} {:12} {loss/all:.4f} loss, {correct/all:.4f}% correct").format(
            epoch_index, mode, **metrics))
        log.info(
            ("Epoch{} {:12} {normal_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  normal', normal_correct, normal_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {lighter_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  lighter', lighter_correct, lighter_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {pressure_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  pressure', pressure_correct, pressure_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {knife_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  knife', knife_correct, knife_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {scissors_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  scissors', scissors_correct, scissors_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {powerbank_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  powerbank', powerbank_correct, powerbank_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {zippooil_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  zippooil', zippooil_correct, zippooil_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {handcuffs_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  handcuffs', handcuffs_correct, handcuffs_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {slingshot_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  slingshot', slingshot_correct, slingshot_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {firecrackers_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  firecrackers', firecrackers_correct, firecrackers_pred_count, **metrics)
        )
        log.info(
            ("Epoch{} {:12} {nailpolish_precision:.4f}% precision ({} of {})").format(
                epoch_index, mode + '  nailpolish', nailpolish_correct, nailpolish_pred_count, **metrics)
        )

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch_index in range(1, self.cli_args.epochs + 1):
            log.info("------------------------------------------------------")
            log.info("epoch {} of {}, {} training / {} validation batches of size {} ".format(
                epoch_index,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size
            ))
            train_metrics = self.train(epoch_index, train_dl)
            self.log_metric(epoch_index, "training", train_metrics, 0.5)

            val_merics = self.validate(epoch_index, val_dl)
            self.log_metric(epoch_index, "validation", val_merics, 0.5)


if __name__ == '__main__':
    XrayTrainApp().main()
# %%
