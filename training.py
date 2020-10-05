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
                            default=32, type=int)

        self.cli_args = parser.parse_args(sys_argv)
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_train_dataloader(self):
        train_dataset = XrayDateset(is_val=0, val_stride=6)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cli_args.batch_size,
            num_workers=self.args.num_workers,
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
        
    def main(self):
        train_dl = self.init_train_dataloader()
        val_dl = self.init_val_dataloader()
        

if __name__ == '__main__':
    XrayTrainApp().main()
# %%
