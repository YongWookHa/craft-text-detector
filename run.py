"""
pytorch lightning template for model implementation
"""

import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from models.craft import CRAFT
from datasets.craft_dataset import CustomDataset, CustomCollate
from torch.utils.data import DataLoader

from utils.base import load_setting


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=0,
                        help="Train experiment version")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=4,
                        help="batch size")
    parser.add_argument("--preprocessed", action="store_true",
                        help="using preprocessed data")
    args = parser.parse_args()

    # setting
    cfg = load_setting(args.setting)
    cfg.update(vars(args))
    print("setting:", cfg)

    # 다양한 한글 데이터
    custom_collate = CustomCollate(image_size=cfg.craft.image_size,
                                   load_preprocessed_data=args.preprocessed)

    train_set = CustomDataset(data_dir=cfg.train_data_path)
    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                collate_fn=custom_collate)

    val_set = CustomDataset(data_dir=cfg.val_data_path)
    # val_len = int(len(val_set)*0.01)
    # val_set = torch.utils.data.random_split(val_set, lengths=[val_len, len(val_set)-val_len])[0]
    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                collate_fn=custom_collate)
    model = CRAFT(cfg)
    logger = TensorBoardLogger("tb_logs", name="model", version=cfg.version,
                               default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="fscore",
        dirpath=f"checkpoints/version_{cfg.version}",
        filename="checkpoints-{epoch:02d}-{fscore:.3f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    device_cnt = torch.cuda.device_count()
    trainer = pl.Trainer(gpus=device_cnt, max_epochs=cfg.epochs,
                        logger=logger, num_sanity_val_steps=1,
                        strategy="dp" if device_cnt > 1 else None,
                        callbacks=[ckpt_callback, lr_callback],
                        resume_from_checkpoint=cfg.load_chkpt if cfg.load_chkpt else None)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
