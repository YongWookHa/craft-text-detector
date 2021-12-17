import sys
import torch
import argparse
from tqdm import tqdm

from utils.base import load_setting
from datasets.craft_dataset import CustomDataset, CustomCollate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--num_workers", "-nw", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=4,
                        help="batch size")
    args = parser.parse_args()

    cfg = load_setting(args.setting)
    collate = CustomCollate(image_size=cfg.craft.image_size, save_data=True)

    train_set = CustomDataset(data_dir=cfg.train_data_path)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=collate)
    val_set = CustomDataset(data_dir=cfg.val_data_path)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=collate)

    for _ in tqdm(train_dataloader, total=len(train_dataloader), desc="train_data_preprocessing"):
        pass

    for _ in tqdm(val_dataloader, total=len(val_dataloader), desc="valid_data_preprocessing"):
        pass
