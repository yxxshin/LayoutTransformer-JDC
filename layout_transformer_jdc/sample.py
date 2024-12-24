import argparse
import os

import torch
from dataset import JSONLayout, MNISTLayout
from model import GPT, GPTConfig
from utils import sample, set_seed, transfer_to_category


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--input_dim", default=12, type=int)
    parser.add_argument("--diffloss_d", type=int, default=6)
    parser.add_argument("--diffloss_w", type=int, default=512)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
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
        dataset.vocab_size,
        dataset.max_length,
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

    # Get first few samples from dataset
    x = dataset[0][1].unsqueeze(0)  # [0][0] gets x from (x,y,mask) tuple
    x = x.to(device)

    with torch.no_grad():
        # Random
        rand_layouts = (
            sample(
                model, 
                x[:, :1, :],
                steps=dataset.max_length,
            )
            .cpu() 
            .numpy()
        )
        
        # Completion with one box
        completion_one_layouts = (
            sample(
                model,
                x[:, :2, :],
                steps=dataset.max_length,
            )
            .cpu()
            .numpy()
        )
        
        # Completion with two boxes
        completion_two_layouts = (
            sample(
                model, 
                x[:, :3, :],
                steps=dataset.max_length,
            )
            .cpu() 
            .numpy() 
        )
                    

        # Save results

        x = transfer_to_category(x)

        dataset.render(x[0].cpu().numpy()).save(
            os.path.join(args.output_dir, "input.png")
        )
        dataset.render(rand_layouts[0]).save(
            os.path.join(args.output_dir, "random.png")
        )
        dataset.render(completion_one_layouts[0]).save(
            os.path.join(args.output_dir, "completion_one_box.png")
        )
        dataset.render(completion_two_layouts[0]).save(
            os.path.join(args.output_dir, "completion_two_box.png")
        )
        


if __name__ == "__main__":
    main()

