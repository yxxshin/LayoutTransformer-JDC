import argparse
import os

import torch
from dataset import JSONLayout, MNISTLayout
from model import GPT, GPTConfig
from utils import sample, set_seed, transfer_to_category, trim_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--input_dim", default=12, type=int)
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = JSONLayout(args.json_path, max_length=args.max_length)

    # Initialize model config and load checkpoint
    mconf = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=dataset.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        input_dim=args.input_dim,
        diffloss_w=args.diffloss_w,
        diffloss_d=args.diffloss_d,
        num_sampling_steps=args.num_sampling_steps,
        grad_checkpointing=args.grad_checkpointing,
        diffusion_batch_mul=args.diffusion_batch_mul,
        max_length=args.max_length,
    )

    model = GPT(mconf)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    batch_size = args.batch_size
    batch_tensors = []
    for i in range(batch_size):
        batch_item = dataset[i][0]  # [1] gets from (x, y, mask tuple)
        batch_tensors.append(batch_item)

    x = torch.stack(batch_tensors, dim=0)
    x = x.to(device)

    with torch.no_grad():
        # Random 
        print("Sampling: Random")
        random_layout = (
            sample(
                model, 
                x[:, :1, :], 
                steps=dataset.max_length,
            )
            .cpu() 
            .numpy()
        )

        # Completion with one box
        print("Sampling: Completion with one box")
        completion_one = (
            sample(
                model,
                x[:, :2, :],
                steps=dataset.max_length,
            )
            .cpu()
            .numpy()
        )

        # Completion with two boxes
        print("Sampling: Completion with two boxes")
        completion_two = (
            sample(
                model,
                x[:, :3, :],
                steps=dataset.max_length,
            )
            .cpu()
            .numpy()
        )
        
        # Completion with three boxes
        print("Sampling: Completion with three boxes")
        completion_three = (
            sample(
                model,
                x[:, :4, :],
                steps=dataset.max_length,
            )
            .cpu()
            .numpy()
        )
        
        # Save results
        x = transfer_to_category(x)

        sampling_types = {
            "random": random_layout,
            "completion_one": completion_one,
            "completion_two": completion_two,
            "completion_three": completion_three,
        }

        # Save input layouts
        input_dir = os.path.join(args.output_dir, "inputs")
        os.makedirs(input_dir, exist_ok=True)
        for i in range(batch_size):
            dataset.render(trim_tokens(x[i]).cpu().numpy()).save(
                os.path.join(input_dir, f"input_{i}.png")
            )

        # Save generated layouts
        for sample_type, layouts in sampling_types.items():
            type_dir = os.path.join(args.output_dir, sample_type)
            os.makedirs(type_dir, exist_ok=True)

            for i in range(batch_size):
                dataset.render(trim_tokens(layouts[i])).save(
                    os.path.join(type_dir, f"sample_{i}.png")
                )


if __name__ == "__main__":
    main()
