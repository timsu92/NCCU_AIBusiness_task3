"""Read the test csv of pixels and inference the labels using a trained model."""

from pathlib import Path
import argparse
from csv import DictWriter

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .dataset import FashionMNIST, test_transform, labelNames
from .model import Net
from .utils.logging import log, job_progress
from .utils.visualizer import visualize_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions for the FashionMNIST test set."
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path of the model to load for inference",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="my_prediction.csv",
        help="Output CSV file for predictions",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    assert model_path.exists(), f"Model path {model_path} does not exist"

    test_dataset = FashionMNIST("test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=14,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(num_classes=10, img_size=28)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)["model"]
    )
    model.to(device)

    log.info(f"Using device: {device}")
    log.info(f"Loaded model from {model_path}")

    model.eval()
    with job_progress() as progress, open(args.output_file, "w", newline="") as csvfile:
        prog_batch = progress.add_task("Batch", total=len(test_loader))
        writer = DictWriter(csvfile, fieldnames=["imageID", "label"])
        writer.writeheader()
        for i_batch, batch in enumerate(test_loader):
            images = batch.to(device)
            with torch.no_grad():
                preds = model(images)
                preds = preds.argmax(dim=1)
            for i_pred, pred in enumerate(preds):
                writer.writerow(
                    {
                        "imageID": i_batch * args.batch_size + i_pred,
                        "label": pred.item(),
                    }
                )
            if i_batch == 0:
                visualize_images(
                    images,
                    [labelNames[pred.item()] for pred in preds],
                    block=False,
                    save=True,
                )
            progress.advance(prog_batch)
        progress.remove_task(prog_batch)

    # Close all possiply opened matplotlib figures
    if plt.get_fignums():
        input("Inference completed. Press Enter to close the visualization window.")
        plt.close("all")


if __name__ == "__main__":
    main()
