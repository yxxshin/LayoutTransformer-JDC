import argparse
import os

import torch
from data import get_dataset
from model_rico import GPT, GPTConfig
from trainer_rico import Trainer, TrainerConfig
from utils_rico import set_seed


class BaselineDatasetWrapper:
    def __init__(self, dataset, max_length=11):
        self.dataset = dataset
        self.max_length = max_length
        self.vocab_size = 16  # BOS, EOS, PAD + 12 classes for Rico

        # Define special tokens
        self.bos_token = torch.tensor(
            [0.0] * 4 + [0] * 13 + [1, 0, 0], dtype=torch.float32
        )
        self.eos_token = torch.tensor(
            [0.0] * 4 + [0] * 13 + [0, 1, 0], dtype=torch.float32
        )
        self.pad_token = torch.tensor(
            [0.0] * 4 + [0] * 13 + [0, 0, 1], dtype=torch.float32
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        label = data.y.long()
        bbox = data.x
        
        label = torch.clamp(label, 0, 12)
        
        chunk = torch.stack([self.pad_token] * (self.max_length + 2))
        chunk[0] = self.bos_token
        
        seq_length = min(len(bbox), self.max_length)
        sequence = torch.zeros(seq_length, 20)
        sequence[:, :4] = bbox[:seq_length]
        sequence[range(seq_length), 4 + label[:seq_length]] = 1.0  # one-hot labels

        chunk[1 : seq_length + 1] = sequence
        chunk[seq_length + 1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]

        expanded_pad = self.pad_token.expand(y.size())
        mask = ~torch.all(y == expanded_pad, dim=-1)

        return x, y, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layout Transformer")
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs", help="/path/to/logs/dir")

    # Layout options
    parser.add_argument("--max_length", type=int, default=18, help="max seq length")
    parser.add_argument("--input_dim", type=int, default=20, help="input dim")
    parser.add_argument("--disc_dim", type=int, default=16, help="discrete dim")

    # DiffLoss options
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--loss_weight", type=int, default=1)

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
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
    ckpt_dir = os.path.join(log_dir, "ckpt")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset_base = get_dataset("rico", "train", data_path="/home/yxxshin/Desktop/CVLab/LayoutTransformer-JDC/data/dataset/rico")
    val_dataset_base = get_dataset("rico", "val", data_path="/home/yxxshin/Desktop/CVLab/LayoutTransformer-JDC/data/dataset/rico")

    train_dataset = BaselineDatasetWrapper(train_dataset_base, args.max_length)
    val_dataset = BaselineDatasetWrapper(val_dataset_base, args.max_length)

    mconf = GPTConfig(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        input_dim=args.input_dim,
        disc_dim=args.disc_dim,
        diffloss_w=args.diffloss_w,
        diffloss_d=args.diffloss_d,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
        diffusion_batch_mul=args.diffusion_batch_mul,
        max_length=args.max_length,
    )

    model = GPT(mconf)
    model.to(device)
    tconf = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay=args.lr_decay,
        warmup_iters=args.warmup_iters,
        final_iters=args.final_iters,
        sample_every=args.sample_every,
        ckpt_dir=ckpt_dir,
        samples_path=samples_dir,
        loss_weight=args.loss_weight,
    )

    trainer = Trainer(model, train_dataset, val_dataset, tconf, args)
    trainer.train()
