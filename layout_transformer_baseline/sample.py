import os
import argparse
import torch
from torch.nn import functional as F
from dataset import JSONLayout
from model import GPT, GPTConfig
from utils import sample, set_seed
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='samples', help='output directory')
    
    # Dataset options
    parser.add_argument('--json_path', default=None, help='path to json data')
    
    # Model configuration
    parser.add_argument('--max_length', type=int, default=517)
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # set_seed(40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    dataset = JSONLayout(args.json_path, max_length=args.max_length)
    
    # Initialize model with same configuration as training
    mconf = GPTConfig(
        dataset.vocab_size,
        dataset.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd
    )
    
    model = GPT(mconf)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Get first few samples from dataset for conditioning
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    x, y = next(iter(data_loader))
    x = x[:1].to(device)  # Take first 1 samples, matching training code
    
    with torch.no_grad():        
        # Completion: one box (randomness)
        completion_one_layouts = sample(
            model, 
            x[:, :6],
            steps=dataset.max_length,
            temperature=1.0,
            sample=True,
            top_k=5
        ).detach().cpu().numpy()
        
        # Completion: one box (deterministic)
        completion_one_box_deter_layouts = sample(
            model,
            x[:, :6],
            steps=dataset.max_length,
            temperature=1.0,
            sample=False,
            top_k=None
        ).detach().cpu().numpy()
        
        # Completion: two box (randomness)
        completion_two_box_random_layouts = sample(
            model,
            x[:, :11],
            steps=dataset.max_length,
            temperature=1.0,
            sample=True,
            top_k=None
        ).detach().cpu().numpy()
        
        # Completion: two box (deterministic)
        completion_two_box_deter_layouts = sample(
            model,
            x[:, :11],
            steps=dataset.max_length,
            temperature=1.0,
            sample=False,
            top_k=None
        ).detach().cpu().numpy()
        
        # Random sampling
        random_layouts = sample(
            model,
            x[:, :1],
            steps=dataset.max_length,
            temperature=1.0,
            sample=False,
            top_k=None
        ).detach().cpu().numpy()
        
        # Save all results
        for i in range(len(x)):
            dataset.render(x[i].cpu().numpy()).save(
                os.path.join(args.output_dir, f'input.png'))
            
            dataset.render(completion_one_layouts[i]).save(
                os.path.join(args.output_dir, f'completion_one_box_random.png'))
            
            dataset.render(completion_one_box_deter_layouts[i]).save(
                os.path.join(args.output_dir, f'completion_one_box_deter.png'))
            
            dataset.render(completion_two_box_random_layouts[i]).save(
                os.path.join(args.output_dir, f'completion_two_box_random.png'))    
                 
            dataset.render(completion_two_box_deter_layouts[i]).save(
                os.path.join(args.output_dir, f'completion_two_box_deter.png'))
            
            dataset.render(random_layouts[i]).save(
                os.path.join(args.output_dir, f'random.png'))

if __name__ == '__main__':
    main()