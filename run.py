import os
import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import MMCL
from vilt.datamodules.memes_datamodule import MemesDataModule
from vilt.datamodules.hateful_memes_datamodule import HatefulMemesDataModule
from vilt.callbacks.hate_online_eval import HateOnlineEvaluator
from pytorch_lightning.profiler import AdvancedProfiler

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MemesDataModule(_config)

    model = MMCL(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="Contrastive/loss",
        mode="max",
        save_last=True,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    h_dm = HatefulMemesDataModule(_config)
    hm_callback = HateOnlineEvaluator(h_dm)

    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(grad_steps)

    max_steps = 100 #_config["max_steps"] if _config["max_steps"] is not None else None
    profiler = AdvancedProfiler(output_filename='profile.txt')
    
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        profiler=profiler
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
