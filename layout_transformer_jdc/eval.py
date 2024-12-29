import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from dataset import JSONLayout
from fid_model import FIDNetV3, load_fidnet_v3
from model import GPT, GPTConfig
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import sample, set_seed, trim_tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--json_path",
        type=str,
        default="/131_data/yeonsang/PubLayNet/publaynet/val.json",
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--input_dim", default=12, type=int)
    parser.add_argument("--diffloss_d", type=int, default=3)
    parser.add_argument("--diffloss_w", type=int, default=256)
    parser.add_argument("--num_sampling_steps", type=str, default="100")
    parser.add_argument("--grad_checkpointing", type=bool, default=False)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_context_boxes", type=int, default=1)
    parser.add_argument(
        "--fid_weight_dir", type=str, default="fid_weights/FIDNetV3/publaynet-max25"
    )
    return parser.parse_args()


def preprocess_layout(layout, device):
    """Preprocess single layout with proper formatting
    Args:
        layout: numpy array of shape (seq_len, 5) where first dim is category
               and remaining 4 dims are coordinates
        device: torch device
    """
    # Split into category and coordinates
    layout = torch.tensor(layout, dtype=torch.float)
    label = layout[:, 0].long()  # Get categories
    bbox = layout[:, 1:]         # Get coordinates

    # Add batch dimension
    bbox = bbox.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)
    
    # Create mask (ignore padding tokens)
    mask = (label != 7).to(device)  # 7 is padding token
    padding_mask = ~mask
    
    return bbox, label, padding_mask, mask


def compute_fid_score(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Compute FID score between real and generated feature distributions"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)



def compute_alignment_score(bbox):
    """Compute alignment metrics for a single layout"""
    bbox = bbox.cpu().numpy()
    S = bbox.shape[1]

    xc = bbox[..., 0] + bbox[..., 2] / 2
    yc = bbox[..., 1] + bbox[..., 3] / 2
    xl = bbox[..., 0]
    xr = bbox[..., 0] + bbox[..., 2]
    yt = bbox[..., 1]
    yb = bbox[..., 1] + bbox[..., 3]

    # AC-LayoutGAN score
    X = (
        np.stack((xl[0], xc[0], xr[0], yt[0], yc[0], yb[0]))[:, :, None]
        - np.stack((xl[0], xc[0], xr[0], yt[0], yc[0], yb[0]))[:, None, :]
    )
    indices = np.arange(S)
    X[:, indices, indices] = 1.0
    X = np.abs(X)
    X = X.min(axis=0)
    
    epsilon = 1e-8
    X = np.clip(X, 0.0, 0.95)
    
    score_ac = -np.log(1 - X).sum()

    # LayoutGAN++ score
    score_layoutgan = score_ac / S if S > 0 else 0.0
    
    return {
        "alignment-ACLayoutGAN": score_ac,
        "alignment-LayoutGAN++": score_layoutgan,
    }

def compute_overlap(bbox, mask):
    """Compute overlap between layout elements for a single layout"""
    bbox = bbox.cpu().numpy()[0]  # Remove batch dimension
    mask = mask.cpu().numpy()[0]  # Remove batch dimension
    
    valid_boxes = bbox[mask]
    if len(valid_boxes) <= 1:
        return {"overlap": 0.0}

    total_iou = 0
    count = 0
    for i in range(len(valid_boxes)):
        for j in range(i + 1, len(valid_boxes)):
            box1 = valid_boxes[i]
            box2 = valid_boxes[j]

            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = box1[2] * box1[3]
                area2 = box2[2] * box2[3]
                union = area1 + area2 - intersection
                iou = intersection / union
                total_iou += iou
                count += 1

    return {"overlap": total_iou / count if count > 0 else 0.0}


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = JSONLayout(args.json_path, max_length=args.max_length)

    # Initialize models
    # Layout generation model
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

    # FIDNet model
    # Override dataset parameters to match pretrained model
    class PretrainedDatasetConfig:
        def __init__(self):
            self.name = "publaynet"
            self.max_seq_length = 25  
            self.num_classes = 5 

    pretrained_config = PretrainedDatasetConfig()
    fidnet = load_fidnet_v3(pretrained_config, args.fid_weight_dir, device)

    # Extract real features
    real_features = []
    print("Extracting features from real layouts...")
    for i in tqdm(range(min(args.num_samples, len(dataset)))):
        x = dataset[i][1].unsqueeze(0).to(device)
        bbox = x[:, :, :4]
        label = x[:, :, 4].long()
        padding_mask = label == 0
        mask = ~padding_mask

        with torch.no_grad():
            feat = fidnet.extract_features(bbox, label, padding_mask)
            real_features.append(feat.cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)

    # Generate and evaluate layouts
    gen_features = []
    alignment_scores = defaultdict(list)
    overlap_scores = defaultdict(list)

    print("Generating and evaluating layouts...")
    for i in tqdm(range(args.num_samples)):
        # Get random sample from dataset for conditioning
        idx = np.random.randint(0, len(dataset))
        x = dataset[idx][0].unsqueeze(0).to(device)

        with torch.no_grad():
            # Generate completion
            completion = sample(
                model,
                x[:, : args.num_context_boxes + 1, :],  # +1 for BOS token
                steps=dataset.max_length,
            )
            
            trimmed_completion = trim_tokens(completion[0])

            # Process generated layout
            bbox, label, padding_mask, mask = preprocess_layout(trimmed_completion.cpu().numpy(), device)

            # Extract features for FID
            feat = fidnet.extract_features(bbox, label, padding_mask)
            gen_features.append(feat.cpu().numpy())

            # Compute other metrics
            for k, v in compute_alignment_score(bbox).items():
                alignment_scores[k].append(v)
            for k, v in compute_overlap(bbox, mask).items():
                overlap_scores[k].append(v)

    gen_features = np.concatenate(gen_features, axis=0)

    # Compute final metrics
    metrics = {}

    # FID score
    metrics["FID"] = compute_fid_score(real_features, gen_features)

    # Alignment scores
    for k, v in alignment_scores.items():
        metrics[k] = np.mean(v)

    # Overlap scores
    for k, v in overlap_scores.items():
        metrics[k] = np.mean(v)

    # Print results
    print("\nEvaluation Results:")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "eval_results.npy"), metrics)

    return metrics


def main():
    args = get_args()
    set_seed(42)
    evaluate_model(args)


if __name__ == "__main__":
    main()
