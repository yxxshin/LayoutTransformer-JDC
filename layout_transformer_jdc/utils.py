import random

import numpy as np
import seaborn as sns
import torch
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    rgb_triples = [[int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)] for x in palette]
    return rgb_triples


# @torch.no_grad()
# def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
#     """
#     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
#     """
#     block_size = (
#         model.module.get_block_size()
#         if hasattr(model, "module")
#         else model.getcond_block_size()
#     )
#     model.eval()
#     for k in range(steps):
#         x_cond = (
#             x if x.size(1) <= block_size else x[:, -block_size:]
#         )  # crop context if needed
#         logits, _ = model(x_cond)
#         # pluck the logits at the final step and scale by temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
#         # apply softmax to convert to probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
#         # append to the sequence and continue
#         x = torch.cat((x, ix), dim=1)
#
#     return x


@torch.no_grad()
def sample(model, x, steps):
    model.eval()
    b, t, dim = x.size()  # [b, t, 12]

    # Track which sequences have finished
    finished_sequences = torch.zeros(b, dtype=torch.bool, device=x.device)

    for i in range(steps - t):
        # Get indices of unfinished sequences
        unfinished_indices = torch.where(~finished_sequences)[0]

        if len(unfinished_indices) == 0:
            break

        # Only process unfinished sequences
        x_unfinished = x[unfinished_indices]
        processed_logits = model(x_unfinished)  # [unfinished_b, t, 5]
        ix = processed_logits[:, -1, :].unsqueeze(1)  # [unfinished_b, 1, 5]
        ix = transfer_to_onehot(ix)  # [unfinished_b, 1, 12]

        # Create a zero tensor for the step's output
        new_step = torch.zeros((b, 1, dim), device=x.device)
        new_step[:, :, -1] = 1  # padding step

        # Handle coordinate validation and update finished status for unfinished sequences
        for idx, orig_idx in enumerate(unfinished_indices):
            if ix[idx, 0, -2] == 1.0:  # EOS token
                finished_sequences[orig_idx] = True

            coords_valid = (ix[idx, :, 0:4] > 0.0) & (ix[idx, :, 0:4] < 1.0)
            if not coords_valid.all():
                finished_sequences[orig_idx] = True
                ix[idx, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)

            new_step[orig_idx] = ix[idx]

        x = torch.cat((x, new_step), dim=1)

    x = transfer_to_category(x)  # [b, steps, 5]
    return x


# def trim_tokens(tokens, bos, eos, pad=None):
#     bos_idx = np.where(tokens == bos)[0]
#     tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
#     eos_idx = np.where(tokens == eos)[0]
#     tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
#     # tokens = tokens[tokens != bos]
#     # tokens = tokens[tokens != eos]
#     if pad is not None:
#         tokens = tokens[tokens != pad]
#     return tokens


def trim_tokens(tokens, bos=5.0, eos=6.0, pad=7.0):
    categories = tokens[:, 0]
    mask = (categories != bos) & (categories != eos) & (categories != pad)
    trimmed_tokens = tokens[mask]
    return trimmed_tokens


def transfer_to_onehot(x):
    # input: [b, t, 5]
    # output: [b, t, 12]

    x_categories = x[..., 0].long()
    x_coords = x[..., 1:5]

    x_onehot = F.one_hot(x_categories, num_classes=8)

    x = torch.cat([x_coords, x_onehot.float()], dim=-1)

    return x


def transfer_to_category(x):
    # input: [b, t, 12]
    # output: [b, t, 5]
    x_coords = x[..., :4]
    x_onehot = x[..., 4:]
    x_categories = torch.argmax(x_onehot, dim=-1).float()
    x_categories = x_categories.unsqueeze(-1)

    x = torch.cat([x_categories, x_coords], dim=-1)

    return x
