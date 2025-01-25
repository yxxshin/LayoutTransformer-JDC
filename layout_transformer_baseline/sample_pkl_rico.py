import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from data import get_dataset
from model import GPT, GPTConfig
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from utils import sample, trim_tokens
from PIL import Image, ImageDraw

def convert_xywh_to_ltrb(bbox):
    """Convert bbox from [x_center, y_center, width, height] to [x1, y1, x2, y2] format"""
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def convert_layout_to_image(boxes, labels, colors, canvas_size):
    """Convert layout boxes and labels to an image"""
    H, W = canvas_size
    img = Image.new("RGB", (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw from larger boxes to smaller ones
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[int(labels[i])]
        c_fill = color + (100,)  # Add alpha channel for fill
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="samples.pkl")
    parser.add_argument("--precision", type=int, default=8,
                       help="number of bits for coordinate precision (e.g., 8 for 256 levels)")
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--num_context_boxes", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="test")
    parser.add_argument("--num_save", type=int, default=4)
    return parser.parse_args()


def convert_baseline_to_model_input(data, device, precision):
    label, mask = to_dense_batch(data.y, data.batch)
    bbox, _ = to_dense_batch(data.x, data.batch)

    batch_size, seq_length = bbox.size(0), bbox.size(1)
    sequence = torch.zeros(
        batch_size, seq_length * 5 + 2, dtype=torch.long, device=device
    )
    
    scale_factor = (2 ** precision) - 1
    label_offset = 2 ** precision 
    bos_token = label_offset + 13

    for i in range(batch_size):
        curr_idx = 0
        for j in range(seq_length):
            if mask[i, j]:
                label_val = label[i, j].item()
                bbox_coords = bbox[i, j]

                tokens = [label_val + label_offset]
                tokens.extend([(coord * scale_factor).round() for coord in bbox_coords])

                sequence[i, curr_idx : curr_idx + 5] = torch.tensor(
                    tokens, device=device
                )
                curr_idx += 5

    # Add BOS token
    bos_token = torch.full((batch_size, 1), bos_token, device=device)
    sequence = torch.cat([bos_token, sequence], dim=1)
    return sequence, label, mask


def main():
    args = get_args()
    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.dataset_type == "test":
        dataset = get_dataset("rico", "test", data_path="/home/yxxshin/Desktop/CVLab/LayoutTransformer-JDC/data/dataset/rico")

    dataloader = DataLoader(
        dataset, batch_size=512, num_workers=4, pin_memory=True, shuffle=False
    )
    
    vocab_size = 13 + (2 ** args.precision) + 3
    
    label_offset = 2 ** args.precision
    bos_token = label_offset + 13
    eos_token = bos_token + 1 
    pad_token = eos_token + 1

    # Initialize model
    mconf = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(mconf)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    results = []
    saved_count = 0 
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            x, label, mask = convert_baseline_to_model_input(data, device, args.precision)

            # Use only context boxes for conditioning
            context_length = args.num_context_boxes * 5 + 1
            x = x[:, :context_length]

            # Generate layouts
            sampled = sample(
                model,
                x,
                steps=args.max_length,
                temperature=1.0,
                sample=True,
                top_k=None,
            )

            # Process generated sequences
            for layout in sampled:
                layout = layout.cpu().numpy()
                layout = trim_tokens(layout, bos=bos_token, eos=eos_token, pad=pad_token)

                # Skip empty sequences / bad sequences
                if len(layout) == 0 or len(layout) % 5 != 0:
                    continue

                # Reshape to [N, 5] and separate bbox/labels
                layout = layout.reshape(-1, 5)
                bbox = layout[:, 1:].astype(np.float32) / (2 ** args.precision - 1)
                label = (
                    layout[:, 0].astype(np.int64) - (2 ** args.precision)
                ) 

                # Validate outputs
                if len(bbox) > 0 and len(label) > 0:
                    results.append((bbox, label))
                    
                    if saved_count < args.num_save:
                        print(f"bbox: {bbox}")
                        print(f"label: {label}")
                        
                        img = convert_layout_to_image(
                            bbox, label, dataset.colors, (120, 80)
                        )
                        img.save(out_dir / f"generated_{saved_count}.png")
                        saved_count += 1

    # Save results
    with open(args.out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Generated {len(results)} layouts saved to: {args.out_path}")


if __name__ == "__main__":
    main()
