import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils.logging import log, Progress


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, progress: Progress
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    prog_batch = progress.add_task("Eval", total=len(dataloader))
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress.advance(prog_batch)
        progress.remove_task(prog_batch)
    avg_loss = total_loss / total
    accuracy = correct / total
    log.info(f"Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
