import torch
from torchvision.transforms.v2 import ToPILImage
import matplotlib.pyplot as plt

def visualize_images(images: torch.Tensor, preds, truths=None, *, block=True, save=True) -> None:
    """
    Visualizes a batch of images with their corresponding labels and save to file.

    Args:
        images (torch.Tensor): A batch of images of shape (N, C, H, W).
        preds: Predicted labels for the images.
        truths: True labels for the images.
    """
    images = images[:32]
    if truths is not None:
        truths = truths[:32]
    preds = preds[:32]
    to_pil = ToPILImage()

    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis('off')
            continue
        img = to_pil(images[i].cpu())
        ax.imshow(img, cmap='gray')
        pred = preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i]
        if truths is None:
            ax.set_title(f'Pred: {pred}')
        else:
            truth = truths[i].item() if isinstance(truths[i], torch.Tensor) else truths[i]
            color = 'green' if pred == truth else 'red'
            ax.set_title(f'Truth: {truth}, Pred: {pred}', color=color)
        ax.axis('off')
    fig.tight_layout()
    if save:
        fig.savefig('visualization.png')
    plt.show(block=block)