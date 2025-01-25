import argparse
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import sys

# Add both implementation paths
sys.path.append("layout_transformer_baseline")
sys.path.append("layout_transformer_jdc")

from data import get_dataset
from layout_transformer_baseline.model import GPT as GPT_baseline
from layout_transformer_baseline.model import GPTConfig as GPTConfig_baseline
from layout_transformer_baseline.utils import sample as sample_baseline
from layout_transformer_baseline.utils import trim_tokens as trim_tokens_baseline

from layout_transformer_jdc.model import GPT as GPT_jdc
from layout_transformer_jdc.model import GPTConfig as GPTConfig_jdc
from layout_transformer_jdc.utils import sample as sample_jdc
from layout_transformer_jdc.utils import trim_tokens as trim_tokens_jdc

import torch
import torch.nn.functional as F

def convert_xywh_to_ltrb(boxes):
    """Convert [x_center, y_center, width, height] to [left, top, right, bottom]"""
    # Make sure input is a tensor
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    
    # Get the box parameters (handling different possible dimensions)
    if len(boxes.shape) == 1 and len(boxes) == 4:
        # Single box as 1D tensor
        xc, yc, w, h = boxes
    else:
        # Batch of boxes - use slicing instead of unpacking
        xc = boxes[..., 0]
        yc = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]
    
    left = xc - w/2
    top = yc - h/2
    right = xc + w/2
    bottom = yc + h/2
    
    return left, top, right, bottom

def compute_overlap(boxes, mask=None):
    """Compute overlap loss between boxes"""
    # Convert to tensor if needed
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    
    # Create mask if none provided
    if mask is None:
        mask = torch.ones(len(boxes), dtype=torch.bool)
    
    # Get box corners for all pairs
    N = len(boxes)
    boxes_i = boxes.unsqueeze(1).expand(N, N, 4)  # [N, N, 4]
    boxes_j = boxes.unsqueeze(0).expand(N, N, 4)  # [N, N, 4]
    
    l1, t1, r1, b1 = convert_xywh_to_ltrb(boxes_i)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(boxes_j)
    
    # Compute areas of boxes_i
    area1 = (r1 - l1) * (b1 - t1)
    
    # Compute intersection
    inter_left = torch.maximum(l1, l2)
    inter_right = torch.minimum(r1, r2)
    inter_top = torch.maximum(t1, t2)
    inter_bottom = torch.minimum(b1, b2)
    
    inter_w = torch.clamp(inter_right - inter_left, min=0)
    inter_h = torch.clamp(inter_bottom - inter_top, min=0)
    intersection = inter_w * inter_h
    
    # Compute overlap ratio
    overlap = intersection / (area1 + 1e-6)
    
    # Mask diagonal and invalid boxes
    mask_matrix = mask.unsqueeze(0) & mask.unsqueeze(1)
    diag_mask = torch.eye(N, dtype=torch.bool, device=boxes.device)
    overlap = overlap.masked_fill(diag_mask, 0)
    overlap = overlap.masked_fill(~mask_matrix, 0)
    
    return overlap.sum() / (mask.float().sum() + 1e-6)

def compute_alignment(boxes, mask=None):
    """Compute alignment loss between boxes - same mechanism as original"""
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    
    if mask is None:
        mask = torch.ones(len(boxes), dtype=torch.bool)
    
    # Get coordinates
    l, t, r, b = convert_xywh_to_ltrb(boxes)
    xc = boxes[:, 0]
    yc = boxes[:, 1]
    
    # Stack same features as original: [xl, xc, xr, yt, yc, yb]
    features = torch.stack([l, xc, r, t, yc, b], dim=1)
    
    # Compute pairwise distances for each feature
    dists = (features.unsqueeze(0) - features.unsqueeze(1)).abs()
    
    # Handle masking and diagonal
    N = len(boxes)
    mask_matrix = mask.unsqueeze(0) & mask.unsqueeze(1)
    diag_mask = torch.eye(N, dtype=torch.bool, device=boxes.device)
    
    # Set masked and diagonal elements to 1
    dists = dists.masked_fill(~mask_matrix.unsqueeze(-1), 1.0)
    dists = dists.masked_fill(diag_mask.unsqueeze(-1), 1.0)
    
    # Get minimum distance across all pairs and features
    min_dists = dists.min(dim=2)[0].min(dim=1)[0]
    
    # Zero out the masked elements that were set to 1
    min_dists = min_dists.masked_fill(min_dists.eq(1.), 0.)
    min_dists = torch.where(min_dists < 1/512, torch.zeros_like(min_dists), min_dists)
    
    # Compute alignment loss
    alignment = -torch.log(1 - min_dists.clamp(max=0.9999))
    
    return alignment.sum() / (mask.float().sum() + 1e-6)

def compute_and_print_losses(boxes, name):
    """Compute and print both overlap and alignment losses"""
    if len(boxes) == 0:
        print(f"\n{name} Metrics: No boxes to evaluate")
        return
    
    overlap_loss = compute_overlap(boxes)
    alignment_loss = compute_alignment(boxes)
    
    print(f"\n{name} Metrics:")
    print(f"Overlap Loss: {overlap_loss.item():.4f}")
    print(f"Alignment Loss: {alignment_loss.item():.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--proposed_ckpt", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, choices=['train', 'test'], required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--num_context_boxes", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="comparison_results")
    return parser.parse_args()

def convert_layout_to_image(boxes, labels, colors, z_indexes=None, canvas_size=(480, 320)):
    H, W = canvas_size
    # For supersampling, use 3x scale (to ensure better subpixel precision)
    scale = 3
    img_aa = Image.new("RGB", (int(W*scale), int(H*scale)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_aa, "RGBA")
    
    # Sort by z_indexes if provided
    indices = range(len(boxes))
    if z_indexes is not None:
        indices = sorted(indices, key=lambda i: z_indexes[i])
    
    for i in indices:
        bbox, label = boxes[i], labels[i]
        color = colors[int(label)]
        c_fill = color + (100,)  # Alpha=100 for fill
        
        # Calculate coordinates at higher resolution
        x_center, y_center, width, height = bbox
        
        # Do the division before multiplication to match Version 1's rounding
        xs = max(0, int((x_center - width/2) * W * scale))
        ys = max(0, int((y_center - height/2) * H * scale))
        xe = min(W*scale, int((x_center + width/2) * W * scale))
        ye = min(H*scale, int((y_center + height/2) * H * scale))
        
        # Skip if box has no area
        if xe <= xs or ye <= ys:
            continue
        
        # Draw rectangle with slight rounding
        draw.rounded_rectangle([xs, ys, xe, ye], radius=scale * 2, 
                             outline=color, fill=c_fill, width=scale * 3)
    
    # Resize back to original size with antialiasing
    img_final = img_aa.resize((W, H), Image.Resampling.LANCZOS)
    return img_final

def setup_baseline_model(ckpt_path, device):
    mconf = GPTConfig_baseline(
        vocab_size=264,
        block_size=47,
        n_layer=6,
        n_head=8,
        n_embd=512
    )
    model = GPT_baseline(mconf)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()

def setup_proposed_model(ckpt_path, device):
    mconf = GPTConfig_jdc(
        vocab_size=8,
        block_size=12,
        n_layer=12,
        n_head=16,
        n_embd=1024,
        input_dim=12,
        disc_dim=8,
        diffloss_d=6,
        diffloss_w=1024,
        num_sampling_steps="100",
        grad_checkpointing=False,
        diffusion_batch_mul=4,
        max_length=11
    )
    model = GPT_jdc(mconf)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()

def convert_baseline_input(data, device):
    from torch_geometric.utils import to_dense_batch
    label, mask = to_dense_batch(data.y, data.batch)
    bbox, _ = to_dense_batch(data.x, data.batch)
    
    batch_size, seq_length = bbox.size(0), bbox.size(1)
    sequence = torch.zeros(batch_size, seq_length * 5 + 2, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        curr_idx = 0
        for j in range(seq_length):
            if mask[i, j]:
                label_val = label[i, j].item()
                bbox_coords = bbox[i, j]
                tokens = [label_val + 256]
                tokens.extend([(coord * 255).round() for coord in bbox_coords])
                sequence[i, curr_idx:curr_idx + 5] = torch.tensor(tokens, device=device)
                curr_idx += 5
    
    bos_token = torch.full((batch_size, 1), 261, device=device)
    sequence = torch.cat([bos_token, sequence], dim=1)
    return sequence, label, mask

def convert_proposed_input(data, device):
    from torch_geometric.utils import to_dense_batch
    label, mask = to_dense_batch(data.y, data.batch)
    bbox, _ = to_dense_batch(data.x, data.batch)
    
    batch_size, seq_length = bbox.size(0), bbox.size(1)
    sequence = torch.zeros(batch_size, seq_length, 12, device=device)
    sequence[:, :, :4] = bbox
    
    for i in range(batch_size):
        for j in range(seq_length):
            if mask[i, j]:
                sequence[i, j, 4 + label[i, j]] = 1.0
    
    bos_token = torch.zeros(batch_size, 1, 12, device=device)
    bos_token[:, :, -3] = 1.0
    full_sequence = torch.cat([bos_token, sequence], dim=1)
    
    return full_sequence, label, mask

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup datasets (using baseline dataset for visualization)
    dataset = get_dataset("publaynet", args.dataset_type)

    # Get specific sample
    data = dataset[args.index].to(device)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    
    # Load models
    baseline_model = setup_baseline_model(args.baseline_ckpt, device)
    proposed_model = setup_proposed_model(args.proposed_ckpt, device)
    
    # Create output directory
    out_dir = Path(args.out_dir) / f"{args.dataset_type}_{args.index}"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        # Ground truth
        gt_boxes = data.x.cpu().numpy()
        gt_labels = data.y.cpu().numpy()
        print(gt_boxes)
        print(gt_labels)
        gt_img = convert_layout_to_image(gt_boxes, gt_labels, dataset.colors)
        gt_img.save(out_dir / f"gt_{args.index}.png")
        compute_and_print_losses(gt_boxes, "Ground Truth")
        
        # Input (partial)
        input_boxes = gt_boxes[:args.num_context_boxes]
        input_labels = gt_labels[:args.num_context_boxes]
        input_img = convert_layout_to_image(input_boxes, input_labels, dataset.colors)
        input_img.save(out_dir / f"input_{args.index}.png")
        compute_and_print_losses(input_boxes, "Input")
        
        # Baseline generation
        x_baseline, _, _ = convert_baseline_input(data, device)
        context_length = args.num_context_boxes * 5 + 1
        x_baseline = x_baseline[:, :context_length]
        
        baseline_output = sample_baseline(baseline_model, x_baseline, steps=47, temperature=1.0, sample=True, top_k=None)[0]
        baseline_output = baseline_output.cpu().numpy()
        baseline_output = trim_tokens_baseline(baseline_output, bos=261, eos=262, pad=263)
        
        if len(baseline_output) > 0 and len(baseline_output) % 5 == 0:
            baseline_output = baseline_output.reshape(-1, 5)
            baseline_boxes = baseline_output[:, 1:].astype(np.float32) / 255.0
            baseline_labels = baseline_output[:, 0].astype(np.int64) - 256
            baseline_img = convert_layout_to_image(baseline_boxes, baseline_labels, dataset.colors)
            baseline_img.save(out_dir / f"baseline_{args.index}.png")
            compute_and_print_losses(baseline_boxes, "Baseline")
        
        # Proposed method generation
        x_proposed, _, _ = convert_proposed_input(data, device)
        x_proposed = x_proposed[:, :args.num_context_boxes + 1, :]
        
        proposed_output = sample_jdc(proposed_model, x_proposed, steps=11)[0]
        trimmed = trim_tokens_jdc(proposed_output)
        proposed_boxes = trimmed[:, 1:5].cpu().numpy()
        proposed_labels = trimmed[:, 0].cpu().numpy()
        
        proposed_img = convert_layout_to_image(proposed_boxes, proposed_labels, dataset.colors)
        proposed_img.save(out_dir / f"proposed_{args.index}.png")
        compute_and_print_losses(proposed_boxes, "Proposed")

if __name__ == "__main__":
    main()