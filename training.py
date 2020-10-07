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

METRICS_LABEL_INDEX = 0
METRICS_PRED_INDEX = 1
METRICS_LOSS_INDEX = 2
METRICS_SIZE = 3


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
        train_metrics = torch.zeros(METRICS_SIZE,
                                    len(train_dl.dataset),
                                    device=self.device)
        for batch_index, batch_tuple in tqdm(enumerate(train_dl), total=len(train_dl.dataset)):
            self.optimizer.zero_gard()
            loss = self.compute_batch_loss(
                batch_index,
                batch_tuple,
                train_dl.batch_size,
                train_metrics)
            loss.backward()
            self.optimizer.step()

        self.total_train_samples_count += len(train_dl.dataset)

        return train_metrics.to('cpu')

    def validate(self, epoch_index, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics = torch.zeros(METRICS_SIZE,
                                      len(val_dl.dataset),
                                      device=self.device)
            for batch_index, batch_tuple in tqdm(enumerate(val_dl), total=len(val_dl.dataset)):
                self.compute_batch_loss(
                    batch_index,
                    batch_tuple,
                    val_dl.batch_size,
                    val_metrics)

        return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_index, batch_tuple, batch_size, train_metrics):
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

        train_metrics[METRICS_LABEL_INDEX,
                      start_index:end_index] = label_batch_gpu.detach()
        train_metrics[METRICS_LOSS_INDEX,
                      start_index:end_index] = loss_gpu.detach()

        # 此处存疑
        for batch_index, metrics_index in enumerate(range(start_index, end_index)):
            train_metrics[METRICS_PRED_INDEX, metrics_index] = \
                probability_batch_gpu[batch_index,
                                      label_batch_gpu[batch_index]].detach()

        return loss_gpu.mean()

    def log_metrics(self, epoch_index, mode, metrics, classification_threshold=0.5):
        normal_label_mask = metrics[METRICS_LABEL_INDEX] == 0
        lighter_label_mask = metrics[METRICS_LABEL_INDEX] == 1
        pressure_label_mask = metrics[METRICS_LABEL_INDEX] == 2
        knife_label_mask = metrics[METRICS_LABEL_INDEX] == 3
        scissors_label_mask = metrics[METRICS_LABEL_INDEX] == 4
        powerbank_label_mask = metrics[METRICS_LABEL_INDEX] == 5
        zippooil_label_mask = metrics[METRICS_LABEL_INDEX] == 6
        handcuffs_label_mask = metrics[METRICS_LABEL_INDEX] == 7
        slingshot_label_mask = metrics[METRICS_LABEL_INDEX] == 8
        firecrackers_label_mask = metrics[METRICS_LABEL_INDEX] == 9
        nailpolish_label_mask = metrics[METRICS_LABEL_INDEX] == 10

        


    def main(self):
        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()

        for epoch_index in range(1, self.cli_args.epochs + 1):
            train_metrics = self.train(epoch_index, train_dl)


# if __name__ == '__main__':
#     XrayTrainApp().main()
# %%
