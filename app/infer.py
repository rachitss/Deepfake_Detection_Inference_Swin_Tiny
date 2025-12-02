import argparse
import os
import shutil
from typing import Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    logging,
)

from extract_faces import get_frames

logging.set_verbosity_error()


def load_model(model_path: str = 'swin_tiny.pth', device: Optional[torch.device] = None) -> Tuple:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    model = AutoModelForImageClassification.from_pretrained(
        'microsoft/swin-tiny-patch4-window7-224',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return processor, model, device


def run_inference(
    video_path: str,
    *,
    model_path: str = 'swin_tiny.pth',
    processor=None,
    model=None,
    device: Optional[torch.device] = None,
    batch_size: int = 36,
) -> dict:
    """Extract frames from the video and return prediction metadata."""

    if processor is None or model is None or device is None:
        processor, model, device = load_model(model_path=model_path, device=device)

    frame_dir = get_frames(video_path)
    try:
        image_files = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith('.png')
            ]
        )
        if not image_files:
            raise RuntimeError('No frames extracted from video; cannot run inference.')

        pred_classes = []
        probabs = []

        print('\n--------------Running Inference------------------')
        for i in tqdm(range(0, len(image_files), batch_size), desc='Image Batches'):
            batch_paths = image_files[i : i + batch_size]
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                batch_logits = outputs.logits
                prob = torch.sigmoid(batch_logits).detach().cpu().view(-1).tolist()
                probabs.extend(prob)
                pred = (torch.sigmoid(batch_logits) > 0.5).int()
                pred_classes.extend(pred.cpu().numpy().flatten().tolist())

        pred = int(sum(pred_classes) > len(pred_classes) / 2)
        return {
            'video_path': video_path,
            'prediction': pred,
            'frame_count': len(image_files),
            'probabilities': probabs,
        }
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Run deepfake inference on a video file.')
    parser.add_argument('--video', required=True, help='Path to the video file to analyze')
    parser.add_argument('--model-path', default='swin_tiny.pth', help='Path to model weights')
    args = parser.parse_args()

    result = run_inference(args.video, model_path=args.model_path)
    print('\n--------------Result------------------')
    print(f"Video: {result['video_path']}")
    print(f"Predicted class: {result['prediction']}")


if __name__ == '__main__':
    main()
