import os
from pathlib import Path
import argparse
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
import wandb

from .utils.logging import job_progress, log
from .model import Net
from .eval import evaluate
from .dataset import FashionMNIST, train_transform, test_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on the FashionMNIST dataset."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=Path("results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Weights & Biases run ID for logging"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path of the model to load for training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Path(os.path.dirname(__file__)) / ".." / "data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    return parser.parse_args()


@dataclass
class Checkpoint:
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler
    epoch: int
    best_accuracy: float

    def load(self, path: Path, device: torch.device, epoch_add_one: bool = True):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]

        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])

        if isinstance(self.scheduler, CosineAnnealingLR):
            self.scheduler.last_epoch = self.epoch

        return self.epoch + 1 if epoch_add_one else self.epoch, self.best_accuracy

    def save(self, path: Path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_accuracy": self.best_accuracy,
                "random_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                ),
                "numpy_rng_state": np.random.get_state(),
            },
            path,
        )

    def link_to_best(self, from_ckpt: Path):
        if (from_ckpt.parent / "best.pt").exists():
            (from_ckpt.parent / "best.pt").unlink()
        (from_ckpt.parent / "best.pt").hardlink_to(from_ckpt)
        # from_ckpt.hardlink_to(from_ckpt / ".." / "best.pt")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epochs: int,
    result_dir: Path,
    args: argparse.Namespace,
    start_epoch: int = 1,
    val_every: int = 1,
    best_accuracy: float = 0.0,
):
    with job_progress() as progress:
        prog_epoch = progress.add_task(
            "Epoch", total=start_epoch + epochs - 1, completed=start_epoch - 1
        )

        for epoch in range(start_epoch, start_epoch + epochs):
            log.info(f"Epoch {epoch}/{start_epoch + epochs - 1}")
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            prog_batch = progress.add_task("Batch", total=len(train_loader))
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                progress.advance(prog_batch)
            avg_loss = total_loss / total
            accuracy = correct / total
            log.info(f"[Epoch #{epoch}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            scheduler.step()
            log.info(f"[Epoch #{epoch}] Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            if not args.no_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": accuracy,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=epoch,
                )

            if epoch % val_every == 0:
                val_loss, val_accuracy = evaluate(model, val_loader, device, progress)
                log.info(
                    f"[Epoch #{epoch}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
                )
                if not args.no_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/accuracy": val_accuracy,
                        },
                        step=epoch,
                    )
                checkpoint_path = result_dir / f"checkpoint_epoch_{epoch}.pt"
                checkpoint = Checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_accuracy=best_accuracy,
                )
                checkpoint.save(checkpoint_path)
                log.info(f"Checkpoint saved to {checkpoint_path}")
                if val_accuracy > best_accuracy:
                    log.info(
                        f"New best validation accuracy: {val_accuracy:.4f} (previous: {best_accuracy:.4f})"
                    )
                    best_accuracy = val_accuracy
                    checkpoint.link_to_best(from_ckpt=checkpoint_path)
                    log.info(
                        f"Best checkpoint linked to {checkpoint_path / '..' / 'best.pt'}"
                    )

            progress.advance(prog_epoch)
        progress.remove_task(prog_epoch)
    log.info(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")
    return best_accuracy


def main():
    args = parse_args()

    model = Net(num_classes=10, img_size=28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=0.0)

    epoch = 1
    best_accuracy = 0.0
    if args.model_path:
        epoch, best_accuracy = Checkpoint(
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            epoch=0,
            best_accuracy=0.0,
        ).load(
            path=Path(args.model_path),
            device=device,
        )

    args.result_dir = Path(args.result_dir)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir = Path(args.data_dir)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if not args.no_wandb:
        run = wandb.init(
            project="NCCU_AIBusiness_task3",
            id=args.wandb_id,
            resume="allow",
            config=args,
        )
        args.result_dir /= run.id
        args.result_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"""Start training
             Epochs: {epoch}/{args.epochs}
             Batch Size: {args.batch_size}
             Checkpoint: {args.model_path}"""
    )

    train_dataset = FashionMNIST(
        mode="train",
        root=args.data_dir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = FashionMNIST(
        mode="val",
        root=args.data_dir,
        transform=test_transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optim,
        scheduler=scheduler,
        epochs=args.epochs,
        result_dir=args.result_dir,
        args=args,
        start_epoch=epoch,
        best_accuracy=best_accuracy,
    )

if __name__ == "__main__":
    main()