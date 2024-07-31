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
    # Run nvidia-smi command to get GPU memory info
    command = "nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits"
    output = os.popen(command).read().strip().split('\n')

    # Parse the output to find GPU with most free memory
    max_memory = -1
    max_memory_index = -1
    for index, line in enumerate(output):
        free_memory, _ = map(int, line.split(', '))
        if free_memory > max_memory:
            max_memory = free_memory
            max_memory_index = index

    return max_memory_index


def main():
    config = json.load(open("configs/motor_imagery.json", "r"))

    pl.seed_everything(config["seed"])

    free_gpu_id = get_freer_gpu()
    #free_gpu_id=0
    print("select gpu:", free_gpu_id)

    train_loader, val_loader, test_loader, class_weights, max_len = get_data(
        config["dataset"], "E:\s222165064", config["batch_size"],
        [11,3,28,41,39],[25,42,6,1,51]
    )

    config.update({"class_weights": class_weights, "max_len": max_len})

    if config.get("train_embedder"):
        model = PLModel(**config)
        early_stop_callback = EarlyStopping(
            monitor="embedder_val_y_loss",
            patience=20,
            verbose=False,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="embedder_val_y_loss", save_last=True
        )
        tb_logger = pl.loggers.TensorBoardLogger(
            name="embedder_cnn_motor_imagery", save_dir="lightning_logs"
        )
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=[free_gpu_id],
            logger=tb_logger,
            callbacks=[checkpoint_callback,early_stop_callback],
            gradient_clip_val=5,
            gradient_clip_algorithm="value",
        )
        trainer.fit(model, train_loader, val_loader)
        model = PLModel.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path, **config
        )
        torch.save(model.model.cpu().state_dict(), os.path.join("models","trained_embedder_split9.pt"))

    model = PLModel(stage=2, **config)
    model.model.load_state_dict(torch.load(os.path.join("models","trained_embedder_split9.pt")))

    # fix embed
    for p in model.model.embedder.embed.parameters():
        p.requires_grad = False

    # train sesm
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=False,
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_last=True)
    tb_logger = pl.loggers.TensorBoardLogger(
        name="sesm_motor_imagery", save_dir="lightning_logs"
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
        accelerator="gpu",
        devices=[free_gpu_id],
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=5,
        gradient_clip_algorithm="value",
    )
    # tuner = pl.tuner.Tuner(trainer)
    # lr_finder = tuner.lr_find(model, train_loader)
    # new_lr = lr_finder.suggestion()
    # print("suggested lr", new_lr)
    # model.lr = new_lr
    trainer.fit(model, train_loader, val_loader)

    # model = PLModel.load_from_checkpoint(
    #     checkpoint_path=checkpoint_callback.best_model_path, **config
    # )
    torch.save(model.model.cpu().state_dict(), os.path.join("models","trained_model_split9.pt"))


if __name__ == "__main__":
    main()
