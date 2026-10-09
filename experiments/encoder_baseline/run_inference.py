# Created on 10/9/25
# Author @Ishaan Kapur

# Run inference using a baseline U-Net model with a ResNet-34 encoder.
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial baseline: ResNet-34 encoder + standard SMP U-Net decoder.
    # No pretrained weights or trained segmentation checkpoint yet.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=5,
    )
    model.eval()

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )[:args.limit]

    if not image_paths:
        raise SystemExit(f"No supported images found in {input_dir}")

    for image_path in image_paths:
        # Read the image using OpenCV.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        # Resize to the baseline input dimensions: width 256, height 192.
        image = cv2.resize(image, (256, 192))

        # OpenCV uses BGR; convert to RGB and scale pixel values to [0, 1].
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Convert H x W x C to PyTorch's 1 x C x H x W format.
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Save a single-channel PNG with class IDs 0 through 4.
        mask = prediction.astype(np.uint8)
        output_path = output_dir / f"{image_path.stem}.png"

        if not cv2.imwrite(str(output_path), mask):
            print(f"Failed to save: {output_path}")
            continue

        saved_mask = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        print(
            f"{image_path.name} -> {output_path.name} | "
            f"shape={saved_mask.shape} | "
            f"classes={np.unique(saved_mask).tolist()}"
        )

    print("Inference smoke test complete.")


if __name__ == "__main__":
    main()