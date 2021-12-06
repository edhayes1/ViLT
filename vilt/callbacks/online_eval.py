from typing import Optional, Sequence, Tuple, Union
from pytorch_lightning.loggers.base import rank_zero_experiment

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, device
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from vilt.datasets import FeaturesDataset
from vilt.modules import objectives
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from vilt.modules import heads, objectives


class OnlineEvaluator(Callback):  # pragma: no cover

    def __init__(
        self,
        data_module: LightningDataModule,
        name = 'Classification'
    ):
        super().__init__()
        data_module.setup()

        self.train_dataloader = data_module.train_dataloader()
        self.test_dataloader = data_module.val_dataloader()
        self.name = name

    def get_feats_loader(self, pl_module, loader):
        pl_module.eval()

        all_feats = torch.empty(len(loader.dataset), pl_module._hparams.config['hidden_size'])
        all_labels = torch.empty(len(loader.dataset))
        # get features first
        i = 0
        for batch in tqdm(loader):
            img = batch['image'].to(device=pl_module.device, non_blocking=True)
            labels = batch['label']
            text_ids = batch['text_ids'].to(device=pl_module.device, non_blocking=True)
            text_masks = batch['text_masks'].to(device=pl_module.device, non_blocking=True)
            ret = pl_module.infer(img, text_ids, text_masks) 
            
            size = labels.shape[0]

            all_feats[i:i+size] = ret['cls_feats'].cpu()
            all_labels[i:i+size] = labels
            i += size
        
        dataset = FeaturesDataset(all_feats, all_labels)
        dataloader = DataLoader(dataset,
                                batch_size=512,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8,
                                persistent_workers=True)
        
        return dataloader

    def on_validation_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        if not trainer.is_global_zero:
            return

        with torch.no_grad():
            train_feats_loader = self.get_feats_loader(pl_module, self.train_dataloader)
            test_feats_loader = self.get_feats_loader(pl_module, self.test_dataloader)
        
        accuracies, aucrocs, praucs = [], [], []
        # loading_bar = tqdm(total=runs * len(train_feats_loader) * args.num_hateful_training_epochs)
        with torch.enable_grad():
            for i in range(10):

                lin_cls = heads.LinearHead(pl_module._hparams.config['hidden_size']).to(device=pl_module.device)
                # lin_cls.apply(objectives.init_weights)

                optimizer = Adam(lin_cls.classifier.parameters(), lr=0.05)
                
                for epoch in range(20):
                    for batch in train_feats_loader:
                        batch = tuple(t.to(device=pl_module.device, non_blocking=True) for t in batch)
                        feats, labels = batch
                        _, loss = lin_cls(feats, labels)

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # loading_bar.update(1)

                accuracy, aucroc, prauc = self.test_hateful(lin_cls, test_feats_loader, device=pl_module.device)
                accuracies.append(accuracy)
                aucrocs.append(aucroc)
                praucs.append(prauc)

                del lin_cls

        # log metrics
        # print(praucs)
        # print(np.array(praucs).mean())
        print(np.array(praucs).mean(), np.array(aucrocs).mean(), np.array(accuracies).mean())
        pl_module.log(f'{self.name}/pr_auc', np.array(praucs).mean(), sync_dist=False, rank_zero_only=True)
        pl_module.log(f'{self.name}/auroc', np.array(aucrocs).mean(), sync_dist=False, rank_zero_only=True)
        pl_module.log(f'{self.name}/accuracy', np.array(accuracies).mean(), sync_dist=False, rank_zero_only=True)
        del train_feats_loader
        del test_feats_loader
    
    

    def test_hateful(self, lin_cls, dataloader, device='cpu'):
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for batch in dataloader:
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
                feats, labels = batch

                preds, _ = lin_cls(feats, labels)
                all_preds.append(preds.cpu().detach())
                all_labels.append(labels.cpu().detach())

            all_preds = torch.cat(all_preds, 0)
            all_labels = torch.cat(all_labels, 0)

        num_samples = all_preds.shape[0]
        threshed_preds = (torch.sigmoid(all_preds) > 0.5)
        correct = (threshed_preds.squeeze().long() == all_labels.long()).sum().item()
        accuracy = (correct / num_samples)
        aucroc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        prauc = average_precision_score(all_labels.numpy(), all_preds.numpy())

        return accuracy, aucroc, prauc