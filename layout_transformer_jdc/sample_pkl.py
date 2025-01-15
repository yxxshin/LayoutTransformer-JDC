import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from data import get_dataset
from model import GPT, GPTConfig
from utils import sample, trim_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=11)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--input_dim", default=12, type=int)
    parser.add_argument("--disc_dim", default=8, type=int)
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_path", type=str, default="output/generated_layouts.pkl")
    parser.add_argument("--num_context_boxes", type=int, default=1)
    parser.add_argument("--num_save", type=int, default=4)
    parser.add_argument("--dataset_type", type=str, default="test")
    return parser.parse_args()


def convert_baseline_to_model_input(data, device):
    """Convert baseline data to model input format"""
    label, mask = to_dense_batch(data.y, data.batch)
    bbox, _ = to_dense_batch(data.x, data.batch)

    # Create sequence tensor
    batch_size, seq_length = bbox.size(0), bbox.size(1)
    sequence = torch.zeros(batch_size, seq_length, 12, device=device)
    sequence[:, :, :4] = bbox  # coordinates

    # Convert labels to one-hot
    for i in range(batch_size):
        for j in range(seq_length):
            if mask[i, j]:  # Only set for valid elements
                sequence[i, j, 4 + label[i, j]] = 1.0  # one-hot labels

    # Create BOS token
    bos_token = torch.zeros(batch_size, 1, 12, device=device)
    bos_token[:, :, -3] = 1.0  # BOS one-hot

    # Combine BOS with sequence
    full_sequence = torch.cat([bos_token, sequence], dim=1)

    return full_sequence, label, mask


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new("RGB", (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[int(labels[i])]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    return img


def main():
    args = get_args()

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = JSONLayout(args.json_path, max_length=args.max_length)
    if args.dataset_type == "test":
        dataset = get_dataset("publaynet", "test")
    elif args.dataset_type == "train":
        full_dataset = get_dataset("publaynet", "train")
        dataset = torch.utils.data.Subset(full_dataset, range(4226))
        dataset.colors = full_dataset.colors
    

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    mconf = GPTConfig(
        vocab_size=8,
        block_size=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        input_dim=args.input_dim,
        disc_dim=args.disc_dim,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
        diffusion_batch_mul=args.diffusion_batch_mul,
        max_length=args.max_length,
    )

    model = GPT(mconf)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)

            x, y, mask = convert_baseline_to_model_input(data, device)

            sampled_layouts = sample(
                model,
                x[:, : args.num_context_boxes + 1, :],
                steps=args.max_length,
            ).cpu()

            for i, layout in enumerate(sampled_layouts):
                if not mask[i].any():
                    continue

                trimmed = trim_tokens(layout)

                bbox = trimmed[:, 1:5].cpu().numpy()
                label = trimmed[:, 0].cpu().numpy()

                if len(results) < args.num_save:
                    print(f"bbox: {bbox}")
                    print(f"label: {label}")

                    convert_layout_to_image(
                        bbox, label, dataset.colors, (120, 80)
                    ).save(out_dir / f"generated_{len(results)}.png")

                results.append((bbox, label))

    with out_path.open("wb") as fb:
        pickle.dump(results, fb)
    print(f"Generated layouts saved to {out_path}")


if __name__ == "__main__":
    main()
