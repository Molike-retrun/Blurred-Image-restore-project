import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a blurred image using a CNN model")
    parser.add_argument("--input", required=True, help="Path to blurred input image")
    parser.add_argument("--output", required=True, help="Path to write restored output image")
    parser.add_argument(
        "--model",
        default="unet",
        choices=["simple", "residual", "unet"],
        help="Model architecture to use",
    )
    parser.add_argument("--weights", help="Optional path to trained .pt weights")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def load_weights(model: torch.nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    model.eval()

    if args.weights:
        load_weights(model, args.weights)

    image = Image.open(args.input).convert("RGB")
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    with torch.no_grad():
        tensor = to_tensor(image).unsqueeze(0).to(device)
        restored = model(tensor).squeeze(0).cpu().clamp(0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_image(restored).save(output_path)


if __name__ == "__main__":
    run_inference(parse_args())
