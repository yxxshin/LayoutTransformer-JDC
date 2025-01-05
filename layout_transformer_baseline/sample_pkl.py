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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="samples.pkl")
    parser.add_argument("--max_length", type=int, default=47)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--num_context_boxes", type=int, default=1)
    parser.add_argument("--dataset_type", type=str, default="test")
    return parser.parse_args()


def convert_baseline_to_model_input(data, device):
    label, mask = to_dense_batch(data.y, data.batch)
    bbox, _ = to_dense_batch(data.x, data.batch)

    batch_size, seq_length = bbox.size(0), bbox.size(1)
    sequence = torch.zeros(
        batch_size, seq_length * 5 + 2, dtype=torch.long, device=device
    )

    for i in range(batch_size):
        curr_idx = 0
        for j in range(seq_length):
            if mask[i, j]:
                label_val = label[i, j].item()
                bbox_coords = bbox[i, j]

                tokens = [label_val + 256]
                tokens.extend([(coord * 255).round() for coord in bbox_coords])

                sequence[i, curr_idx : curr_idx + 5] = torch.tensor(
                    tokens, device=device
                )
                curr_idx += 5

    # Add BOS token
    bos_token = torch.full((batch_size, 1), 261, device=device)
    sequence = torch.cat([bos_token, sequence], dim=1)
    return sequence, label, mask


def main():
    args = get_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.dataset_type == "test":
        dataset = get_dataset("publaynet", "test")
    elif args.dataset_type == "train":
        full_dataset = get_dataset("publaynet", "train")
        dataset = torch.utils.data.Subset(full_dataset, range(4226))
        dataset.colors = full_dataset.colors

    dataloader = DataLoader(
        dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False
    )

    # Initialize model
    mconf = GPTConfig(
        vocab_size=264,
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
    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            x, label, mask = convert_baseline_to_model_input(data, device)

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
                layout = trim_tokens(layout, bos=261, eos=262, pad=263)

                # Skip empty sequences / bad sequences
                if len(layout) == 0 or len(layout) % 5 != 0:
                    continue

                # Reshape to [N, 5] and separate bbox/labels
                layout = layout.reshape(-1, 5)
                # For categorical coordinates, we're already getting indices 0-255
                # Convert back to [0,1] range by dividing by 255 (not 256)
                bbox = layout[:, 1:].astype(np.float32) / 255.0
                label = (
                    layout[:, 0].astype(np.int64) - 256
                )  # Convert back to dataset labels

                # Validate outputs
                if len(bbox) > 0 and len(label) > 0:
                    results.append((bbox, label))

    # Save results
    with open(args.out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Generated {len(results)} layouts saved to: {args.out_path}")


if __name__ == "__main__":
    main()
