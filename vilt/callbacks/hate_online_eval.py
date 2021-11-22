from typing import Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, device
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
from vilt.datasets import FeaturesDataset
from vilt.modules import objectives
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam




class HateOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """

    def __init__(
        self,
        train_dataloader: DataLoader = None,
        test_dataloader: DataLoader = None
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_feats_loader(self, pl_module, loader):

        all_feats = torch.empty(len(loader.dataset), pl_module._config.hidden_size)
        all_labels = torch.empty(len(loader.dataset))
        # get features first
        i = 0
        for batch in loader:
            img = batch['image'][0]
            labels = batch['labels']
            text_ids = batch['text_ids']
            text_masks = batch['text_masks']
            ret = self.infer(img, text_ids, text_masks) 
            
            size = labels.shape[0]

            all_feats[i:i+size] = ret['cls_output'].cpu()
            all_labels[i:i+size] = labels.cpu()
            i += size
        
        dataset = FeaturesDataset(all_feats, all_labels)
        dataloader = DataLoader(dataset,
                                batch_size=512,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8,
                                persistent_workers=True)
        
        return dataloader

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        with torch.no_grad():
            train_feats_loader = self.get_feats_loader(pl_module, self.train_dataloader)
            test_feats_loader = self.get_feats_loader(pl_module, self.test_dataloader)
        
        accuracies, aucrocs, praucs = [], [], []
        # loading_bar = tqdm(total=runs * len(train_feats_loader) * args.num_hateful_training_epochs)

        for i in range(10):
            pl_module.lin_cls.apply(objectives.init_weights)

            optimizer = Adam(pl_module.lin_cls.parameters(), lr=0.05)
            
            for epoch in range(20):
                for batch in train_feats_loader:
                    batch = tuple(t.to(device=pl_module.device, non_blocking=True) for t in batch)
                    feats, labels = batch
                    _, loss = pl_module.lin_cls(feats, labels)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # loading_bar.update(1)

            accuracy, aucroc, prauc = self.test_hateful(pl_module, test_feats_loader)
            accuracies.append(accuracy)
            aucrocs.append(aucroc)
            praucs.append(prauc)

        # log metrics
        pl_module.log("online_train_acc", np.array(accuracies).mean(), on_step=True, on_epoch=False)
        pl_module.log("online_train_auc", np.array(aucrocs).mean(), on_step=True, on_epoch=False)
        pl_module.log("online_train_prc", np.array(praucs).mean(), on_step=True, on_epoch=False)
    

    def test_hateful(self, pl_module, dataloader):
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for batch in dataloader:
                batch = tuple(t.to(device=pl_module.device, non_blocking=True) for t in batch)
                feats, labels = batch

                preds, _ = pl_module.lin_cls(feats, labels)
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