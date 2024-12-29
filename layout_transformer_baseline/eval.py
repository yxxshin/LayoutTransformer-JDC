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
    # Required arguments
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint")

    # Dataset options
    parser.add_argument(
        "--json_path",
        type=str,
        default="/131_data/yeonsang/PubLayNet/publaynet/val.json",
    )
    parser.add_argument("--max_length", type=int, default=517)

    # Model configuration
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)

    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_context_boxes", type=int, default=1)
    parser.add_argument(
        "--fid_weight_dir", type=str, default="fid_weights/FIDNetV3/publaynet-max25"
    )

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--sample_mode", type=str, default="random", choices=["random", "deterministic"]
    )

    return parser.parse_args()


def preprocess_layout(layout, device):
    """Preprocess single layout with proper formatting"""
    # Split into category and coordinates
    layout = torch.tensor(layout, dtype=torch.float)
    label = layout[0].long()  # Get categories
    bbox = layout[1:]  # Get coordinates

    # Add batch dimension
    bbox = bbox.unsqueeze(0).to(device)
    label = label.unsqueeze(0).to(device)

    # Create mask (ignore padding tokens)
    mask = (label != 0).to(device)  # 0 is padding token in new version
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


def compute_overlap(bbox):
    """Compute overlap between layout elements for a single layout"""
    bbox = bbox.cpu().numpy()[0]

    if len(bbox) <= 1:
        return {"overlap": 0.0}

    total_iou = 0
    count = 0
    for i in range(len(bbox)):
        for j in range(i + 1, len(bbox)):
            box1 = bbox[i]
            box2 = bbox[j]

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
    print(f"Using device: {device}")

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
    )
    model = GPT(mconf)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    # FIDNet model
    class PretrainedDatasetConfig:
        def __init__(self):
            self.name = "publaynet"
            self.max_seq_length = 25
            self.num_classes = 5

    pretrained_config = PretrainedDatasetConfig()
    fidnet = load_fidnet_v3(pretrained_config, args.fid_weight_dir, device)

    # Extract real features
    real_features = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Extracting features from real layouts...")

    # Get the base category ID (should be dataset.size = 256 for 8-bit precision)
    base_category_id = dataset.size
    num_categories = len(dataset.categories)
    print(f"Base category ID: {base_category_id}, Num categories: {num_categories}")

    for i, (x, y) in enumerate(tqdm(data_loader)):
        if i >= args.num_samples:
            break
        y = y.to(device)

        # Convert to layout format
        boxes = []
        labels = []

        for j in range(0, len(y[0]), 5):
            if j + 4 < len(y[0]):
                box = y[0][j : j + 5]
                if (
                    box[0] >= base_category_id
                    and box[0] < base_category_id + num_categories
                ):
                    # Convert category ID back to 0-based index for FIDNet
                    label_val = box[0].item() - base_category_id
                    labels.append(label_val)

                    # Get box coordinates
                    box_coords = box[1:].float()  # Convert to float first

                    # Convert quantized coordinates back to [0,1] range
                    box_coords = box_coords / (dataset.size - 1)
                    boxes.append(box_coords)

        # Create tensors
        bbox = (
            torch.stack(boxes).to(torch.float32).to(device).unsqueeze(0)
        )  # [1, num_boxes, 4]
        label = (
            torch.tensor(labels, dtype=torch.long).to(device).unsqueeze(0)
        )  # [1, num_boxes]

        if not boxes:  # Skip if no valid boxes found
            continue

        # Move tensors to the correct device and type
        bbox = (
            torch.stack(boxes).to(torch.float32).to(device).unsqueeze(0)
        )  # [1, num_boxes, 4]
        label = (
            torch.tensor(labels, dtype=torch.long).to(device).unsqueeze(0)
        )  # [1, num_boxes]
        padding_mask = (label == -1).to(device)  # Updated padding token check
        mask = (~padding_mask).to(device)

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
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        x, _ = next(iter(data_loader))
        x = x.to(device)

        with torch.no_grad():
            # Generate completion
            completion = sample(
                model,
                x[:, : args.num_context_boxes + 1],  # +1 for BOS token
                steps=dataset.max_length,
                temperature=args.temperature,
                sample=(args.sample_mode == "random"),
                top_k=args.top_k,
            )

            bos = torch.tensor(dataset.vocab_size - 3)
            eos = torch.tensor(dataset.vocab_size - 2)

            trimmed_completion = trim_tokens(completion[0].cpu(), bos, eos)

            layout = trimmed_completion.reshape(-1, 5)
            categories = layout[:, 0] - dataset.size
            boxes = layout[:, 1:].float() / (dataset.size - 1)

            # After processing the completion
            bbox = boxes.unsqueeze(0).to(device)
            label = categories.unsqueeze(0).to(device)
            padding_mask = torch.zeros_like(label, dtype=torch.bool).to(device)

            # # Process generated layout
            # bbox, label, padding_mask, mask = preprocess_layout(
            #     trimmed_completion.cpu().numpy(), device
            # )

            # Extract features for FID
            feat = fidnet.extract_features(bbox, label, padding_mask)
            gen_features.append(feat.cpu().numpy())

            # Compute other metrics
            for k, v in compute_alignment_score(bbox).items():
                alignment_scores[k].append(v)
            for k, v in compute_overlap(bbox).items():
                overlap_scores[k].append(v)

            # Save sample visualizations for first few iterations
            if i < 5:
                os.makedirs(args.output_dir, exist_ok=True)
                dataset.render(trimmed_completion.cpu().numpy()).save(
                    os.path.join(args.output_dir, f"sample_{i}.png")
                )

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
    metrics = evaluate_model(args)


if __name__ == "__main__":
    main()
