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
from utils import sample, set_seed, transfer_to_category, trim_tokens


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


def preprocess_layout_batch(layouts, device):
    """Preprocess batch of layouts with proper formatting
    Args:
        layouts: numpy array of shape (batch_size, seq_len, 5)
        device: torch device
    """
    # Split into category and coordinates
    layouts = torch.tensor(layouts, dtype=torch.float)
    labels = layouts[..., 0].long()  # Get categories
    bboxes = layouts[..., 1:]  # Get coordinates

    # Add batch dimension
    bboxes = bboxes.to(device)
    labels = labels.to(device)

    # Create mask (ignore padding tokens)
    masks = (labels < 5).to(device)  # 5 = bos, 6 = eos, 7 = padding
    padding_masks = ~masks

    # For FID compatibility, we change padding labels to label 0
    labels = torch.where(masks, labels, torch.zeros_like(labels))

    return bboxes, labels, padding_masks, masks


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


def compute_alignment_score_batch(bboxes):
    """Compute alignment metrics for a batch of layouts"""
    bboxes = bboxes.cpu().numpy()
    batch_scores = []

    for bbox in bboxes:  # Iterate over batch dimension
        S = bbox.shape[0]

        xc = bbox[..., 0] + bbox[..., 2] / 2
        yc = bbox[..., 1] + bbox[..., 3] / 2
        xl = bbox[..., 0]
        xr = bbox[..., 0] + bbox[..., 2]
        yt = bbox[..., 1]
        yb = bbox[..., 1] + bbox[..., 3]

        X = (
            np.stack((xl, xc, xr, yt, yc, yb))[:, :, None]
            - np.stack((xl, xc, xr, yt, yc, yb))[:, None, :]
        )
        indices = np.arange(S)
        X[:, indices, indices] = 1.0
        X = np.abs(X)
        X = X.min(axis=0)

        epsilon = 1e-8
        X = np.clip(X, 0.0, 0.95)

        score_ac = -np.log(1 - X).sum()
        score_layoutgan = score_ac / S if S > 0 else 0.0

        batch_scores.append(
            {
                "alignment-ACLayoutGAN": score_ac,
                "alignment-LayoutGAN++": score_layoutgan,
            }
        )

    # Average scores across batch
    return {
        k: np.mean([score[k] for score in batch_scores]) for k in batch_scores[0].keys()
    }


def compute_overlap_batch(bboxes):
    """Compute overlap between layout elements for a batch of layouts"""
    bboxes = bboxes.cpu().numpy()

    batch_scores = []

    for i in range(len(bboxes)):
        bbox = bboxes[i]        # iterating each batch

        if len(bbox) <= 1:
            batch_scores.append({"overlap": 0.0})
            continue

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

        batch_scores.append({"overlap": total_iou / count if count > 0 else 0.0})

    # Average scores across batch
    return {
        k: np.mean([score[k] for score in batch_scores]) for k in batch_scores[0].keys()
    }


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JSONLayout(args.json_path, max_length=args.max_length)

    # Layout generation model
    mconf = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=dataset.max_length+1,
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
    model = torch.nn.DataParallel(model).to(device)
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
    num_real_batches = (
        min(args.num_samples, len(dataset)) + args.batch_size - 1
    ) // args.batch_size

    for batch_idx in tqdm(range(num_real_batches)):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.num_samples, len(dataset))
        batch_size = batch_end - batch_start

        batch_tensors = []
        for i in range(batch_start, batch_end):
            batch_tensors.append(dataset[i][1])

        x = torch.stack(batch_tensors, dim=0).to(device)
        x = transfer_to_category(x)
        label = x[..., 0].long()
        bbox = x[..., 1:]
        padding_mask = label >= 5
        mask = ~padding_mask

        label = torch.where(mask, label, torch.zeros_like(label))

        with torch.no_grad():
            feat = fidnet.extract_features(bbox, label, padding_mask)
            real_features.append(feat.cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)

    sampling_configs = {
        # "random": 1,  # Only BOS token
        "completion_one": 2,  # BOS + 1 box
        "completion_two": 3,  # BOS + 2 boxes
        # "completion_three": 4,  # BOS + 3 boxes
    }

    all_metrics = {}

    for sample_type, num_context_boxes in sampling_configs.items():
        print(f"\nEvaluating {sample_type}...")

        gen_features = []
        alignment_scores = defaultdict(list)
        overlap_scores = defaultdict(list)

        # Create sequential data loader
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        for batch_idx, (x, _, _) in enumerate(tqdm(data_loader)):
            if batch_idx * args.batch_size >= args.num_samples:
                break

            x = x.to(device)

            with torch.no_grad():
                # Generate completion
                completion = sample(
                    model,
                    x[:, :num_context_boxes],  # Include context boxes
                    steps=dataset.max_length,
                )

                # Process completions
                trimmed_sequences = [trim_tokens(comp) for comp in completion]
                max_length = max(seq.size(0) for seq in trimmed_sequences)

                padded_sequences = [pad_sequence_to_length(seq, max_length) 
                              for seq in trimmed_sequences]

                batch_trimmed = torch.stack(padded_sequences)

                bbox, label, padding_mask, mask = preprocess_layout_batch(
                    batch_trimmed.cpu().numpy(), device
                )

                feat = fidnet.extract_features(bbox, label, padding_mask)
                gen_features.append(feat.cpu().numpy())

                alignment_metrics = compute_alignment_score_batch(bbox)
                for k, v in alignment_metrics.items():
                    alignment_scores[k].append(v)

                overlap_metrics = compute_overlap_batch(bbox)
                for k, v in overlap_metrics.items():
                    overlap_scores[k].append(v)

                # Save sample visualizations only for the first few samples
                if batch_idx == 0:
                    save_dir = os.path.join(args.output_dir, sample_type)
                    os.makedirs(save_dir, exist_ok=True)
                    for i in range(min(5, len(completion))):
                        dataset.render(batch_trimmed[i].cpu().numpy()).save(
                            os.path.join(save_dir, f"sample_{i}.png")
                        )

        # Compute metrics for this sampling type
        gen_features = np.concatenate(gen_features, axis=0)

        metrics = {
            "FID": compute_fid_score(real_features, gen_features),
            **{k: np.mean(v) for k, v in alignment_scores.items()},
            **{k: np.mean(v) for k, v in overlap_scores.items()},
        }

        all_metrics[sample_type] = metrics

    # Print and save all results
    print("\nEvaluation Results:")
    for sample_type, metrics in all_metrics.items():
        print(f"\n{sample_type}:")
        for k, v in sorted(metrics.items()):
            print(f"{k}: {v:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "eval_results.npy"), all_metrics)

    return all_metrics


def pad_sequence_to_length(sequence, target_length):
    """Pad a sequence to the target length with padding tokens (7)"""
    current_length = len(sequence)
    if current_length >= target_length:
        return sequence[:target_length]
    
    padding = torch.full((target_length - current_length, sequence.size(1)), 7, 
                        dtype=sequence.dtype, device=sequence.device)
    return torch.cat([sequence, padding], dim=0)



def main():
    args = get_args()
    set_seed(42)
    evaluate_model(args)


if __name__ == "__main__":
    main()
