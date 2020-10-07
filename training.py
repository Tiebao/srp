# %%
import sys
import argparse
import datetime
import logging
import torch
import torch.nn as nn
from datasets import XrayDateset
from torch.utils.data import DataLoader
from torch.optim import SGD
from model import XrayModel
from tqdm import tqdm
# %%
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class XrayTrainApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='工作进程数量',
                            default=8, type=int)
        parser.add_argument('--batch-size', help='每一批次的样本数',
                            default=4, type=int)

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
            num_workers=self.args.num_workers,
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

        for batch_index, batch_tuple in tqdm(enumerate(train_dl), total=len(train_dl.dataset)):
            self.optimizer.zero_gard()
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

            for batch_index, batch_tuple in tqdm(enumerate(val_dl), total=len(val_dl.dataset)):
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
        

    def main(self):
        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch_index in range(1, self.cli_args.epochs + 1):
            train_metrics = self.train(epoch_index, train_dl)


# if __name__ == '__main__':
#     XrayTrainApp().main()
# %%
