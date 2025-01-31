import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import torch.distributions as td


class MMCL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.temperature = self.hparams.config['temperature']
        self.contrastive_dim = self.hparams.config['contrastive_dim']

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        self.mlm_score = heads.MLMHead(bert_config)
        self.mlm_score.apply(objectives.init_weights)

        self.mlp = heads.MLPHead(config["hidden_size"], self.contrastive_dim)
        self.mlp.apply(objectives.init_weights)

        mixup_params = self.hparams.config["mixup_sample"]
        self.beta = td.Beta(torch.tensor([mixup_params[0]]), torch.tensor([mixup_params[1]]))

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def create_cross_negatives(self, img_0, text_ids_0, text_masks_0):
        # creates 2 representations of mismatched modalities
        N = img_0.shape[0]
        with torch.no_grad():
            index = torch.roll(torch.arange(N, device=img_0.device), 1, 0)
            ret_0 = self.infer(img_0, text_ids_0[index], text_masks_0[index], max_image_len=self.hparams.config["max_image_len"], mixup=False)['cls_feats']
        negatives = torch.stack([ret_0, torch.roll(ret_0, -1, 0)], dim=1)
        return negatives
    
    def infer(
        self,
        img, text_ids, text_masks,
        image_token_type_idx=1,
        image_masks=None,
        max_image_len=-1,
        mixup=False
    ):

        text_embeds = self.text_embeddings(text_ids)
        if mixup:
            N = text_embeds.shape[0]
            lam = self.beta.sample((N, 1)).to(self.device)
            index = torch.randperm(N, device=self.device)
            
            text_embeds = lam * text_embeds + (1.0 - lam) * text_embeds[index]
            text_masks = torch.max(text_masks, text_masks[index])

        image_embeds, image_masks, _, _ = self.transformer.visual_embed(
                img,
                max_image_len=max_image_len,
                mask_it=False,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds + self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            'cls_feats': cls_feats,
            'text_feats': text_feats
        }

        return ret

    def forward(self, batch):
        img_0 = batch['image_0'][0]
        img_1 = batch['image_1'][0]

        text_ids_0 = batch['text_0_ids_mlm']
        text_ids_1 = batch['text_1_ids_mlm']
        text_labels_0 = batch['text_0_labels_mlm']
        text_labels_1 = batch['text_1_labels_mlm']
        text_masks_0 = batch['text_0_masks']
        text_masks_1 = batch['text_1_masks']
        
        ret_0 = self.infer(img_0, text_ids_0, text_masks_0, max_image_len=self.hparams.config["max_image_len"], mixup=True) 
        ret_1 = self.infer(img_1, text_ids_1, text_masks_1, max_image_len=self.hparams.config["max_image_len"], mixup=True)

        ret = {}

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret['mlm_loss_0'] = objectives.compute_mlm(self, ret_0['text_feats'], text_labels_0)
            ret['mlm_loss_1'] = objectives.compute_mlm(self, ret_1['text_feats'], text_labels_1)

        # Contrastive Learning
        if "cl" in self.current_tasks:
            cross_negatives = self.create_cross_negatives(img_0, text_ids_0, text_masks_0)
            ret['cl_loss'] = objectives.compute_cl(self, ret_0['cls_feats'], ret_1['cls_feats'], cross_negatives=cross_negatives)

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
