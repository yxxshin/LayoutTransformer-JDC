import argparse
import os

import numpy as np
import torch
from data import get_dataset
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed


class BaselineDatasetWrapper:
    def __init__(self, dataset, max_length, precision):
        self.dataset = dataset
        self.max_length = max_length
        self.precision = precision
        self.vocab_size = 13 + (2 ** precision) + 3
        self.colors = dataset.colors
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1
        self.scale_factor = (2 ** precision) - 1

        self.valid_indices = []
        for idx in range(len(self.dataset)):
            try:
                sequence, _ = self._get_sequence(idx)
                max_token = sequence.max().item()
                if max_token < self.vocab_size:
                    self.valid_indices.append(idx)
                else:
                    print(f"Skipping sample {idx}: Token {max_token} exceeds vocab size {self.vocab_size}")
            except Exception as e:
                print(f"Skipping invalid sample {idx}: {e}")

        if not self.valid_indices:
            raise ValueError("No valid samples found in the dataset!")

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Use valid_indices to skip invalid samples
        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} is out of range for valid_indices (length {len(self.valid_indices)})")
        valid_idx = self.valid_indices[idx]
        try:
            return self._get_sequence(valid_idx)
        except Exception as e:
            print(f"Error loading sample {valid_idx}: {e}")
            raise  # Raise the exception to skip this sample

    def _get_sequence(self, idx):
        data = self.dataset[idx]

        # Get the sequence for current layout
        sequence = []
        sequence.append(self.bos_token)  # Add BOS token

        # Add bbox coordinates and labels in sequence
        for y, box in zip(data.y, data.x):
            y_clamped = max(0, min(y.item(), 12))
            box_clamped = torch.clamp((box * self.scale_factor).round(), 0, self.scale_factor)
            # Adjust label to proper vocab range (assuming 13 classes)
            # sequence.extend([y.item() + (2 ** self.precision)] + np.long((box * self.scale_factor).round()).tolist())
            
            class_token = y_clamped + (2 ** self.precision)
            sequence.extend([class_token] + box_clamped.long().tolist())

        sequence.append(self.eos_token)  # Add EOS token

        # Pad sequence if needed
        if len(sequence) < self.max_length:
            sequence.extend([self.pad_token] * (self.max_length - len(sequence)))
        else:
            sequence = sequence[: self.max_length]

        # Convert to tensor
        sequence = torch.tensor(sequence).long()

        # Create targets by shifting input
        targets = torch.full_like(sequence, self.pad_token)
        targets[:-1] = sequence[1:]

        return sequence, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout Transformer")
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")

    # MNIST options
    parser.add_argument("--data_dir", default=None, help="/path/to/mnist/data")
    parser.add_argument(
        "--threshold", type=int, default=16, help="threshold for grayscale values"
    )

    # Layout options
    parser.add_argument("--max_length", type=int, default=80, help="batch size")
    parser.add_argument("--precision", type=int, default=8, 
                       help="number of bits for coordinate precision (e.g., 8 for 256 levels)")
    parser.add_argument("--element_order", default="raster")
    parser.add_argument("--attribute_order", default="cxywh")

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument(
        "--lr_decay", action="store_true", help="use learning rate decay"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="linear lr warmup iters"
    )
    parser.add_argument(
        "--final_iters", type=int, default=0, help="cosine lr final iters"
    )
    parser.add_argument(
        "--sample_every", type=int, default=1, help="sample every epoch"
    )

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset_base = get_dataset("rico", "train", data_path="/home/yxxshin/Desktop/CVLab/LayoutTransformer-JDC/data/dataset/rico")
    valid_dataset_base = get_dataset("rico", "val", data_path="/home/yxxshin/Desktop/CVLab/LayoutTransformer-JDC/data/dataset/rico")

    train_dataset = BaselineDatasetWrapper(train_dataset_base, args.max_length, args.precision)
    valid_dataset = BaselineDatasetWrapper(valid_dataset_base, args.max_length, args.precision)

    mconf = GPTConfig(
        train_dataset.vocab_size,
        train_dataset.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )  # a GPT-1
    model = GPT(mconf)
    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_decay=args.lr_decay,
        learning_rate=args.lr * args.batch_size,
        warmup_iters=args.warmup_iters,
        final_iters=args.final_iters,
        ckpt_dir=ckpt_dir,
        samples_dir=samples_dir,
        sample_every=args.sample_every,
    )
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    trainer.train()
