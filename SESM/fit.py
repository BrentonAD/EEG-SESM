import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from trainer import PLModel
from sesm import get_data

import os


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def main():
    config = json.load(open("configs/sleep_edf.json", "r"))

    pl.seed_everything(config["seed"])

    # free_gpu_id = get_freer_gpu()
    # print("select gpu:", free_gpu_id)

    train_loader, val_loader, test_loader, class_weights, max_len = get_data(
        config["dataset"], config["batch_size"]
    )

    config.update({"class_weights": class_weights, "max_len": max_len})

    if config.get("train_embedder"):
        model = PLModel(**config)
        early_stop_callback = EarlyStopping(
            monitor="embedder_val_y_loss",
            patience=30,
            verbose=False,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="embedder_val_y_loss", save_last=True
        )
        tb_logger = pl.loggers.TensorBoardLogger(
            name="embedder_cnn_sleep_edf", save_dir="lightning_logs"
        )
        trainer = pl.Trainer(
            max_epochs=35,
            #gpus=[free_gpu_id],
            logger=tb_logger,
            callbacks=[checkpoint_callback,early_stop_callback],
            gradient_clip_val=5,
            gradient_clip_algorithm="value",
        )
        trainer.fit(model, train_loader, val_loader)
        model = PLModel.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path, **config
        )
        torch.save(model.model.cpu().state_dict(), os.path.join("models","trained_embedder.pt"))

    model = PLModel(stage=2, **config)
    model.model.load_state_dict(torch.load(os.path.join("models","trained_embedder.pt")))

    # fix embed
    for p in model.model.embedder.embed.parameters():
        p.requires_grad = False

    # train sesm
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)
    tb_logger = pl.loggers.TensorBoardLogger(
        name="sesm_sleep_edf", save_dir="lightning_logs"
    )
    trainer = pl.Trainer(
        # max_epochs=500,
        # gpus=[0, 1, 2],
        # accelerator="ddp"
        # amp_backend="apex",
        # amp_level="O2",
        # deterministic=True,
        # auto_lr_find=True,
        max_epochs=50,
        #gpus=[free_gpu_id],
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=5,
        gradient_clip_algorithm="value",
    )

    # lr_finder = trainer.tuner.lr_find(model, train_loader)
    # new_lr = lr_finder.suggestion()
    # print("suggested lr", new_lr)
    # model.lr = new_lr
    trainer.fit(model, train_loader, val_loader)

    model = PLModel.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path, **config
    )
    torch.save(model.model.cpu().state_dict(), os.path.join("models","trained_model.pt"))


if __name__ == "__main__":
    main()
