"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import math

import torch
import torch.nn as nn
from diffloss import DiffLoss
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768
    input_dim = 12  # x, y, w, h, categories(one hot, 8)
    disc_dim = 8  # categories(one hot, 8)

    # DiffLoss config
    diffloss_d = 3
    diffloss_w = 256
    num_sampling_steps = "100"
    grad_checkpointing = False
    diffusion_batch_mul = 4


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.tok_emb = nn.Linear(config.input_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.disc_dim, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        self.diffloss = DiffLoss(
            target_channels=4,
            z_channels=config.n_embd,
            width=config.diffloss_w,
            depth=config.diffloss_d,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=config.grad_checkpointing,
        )

        self.diffusion_batch_mul = config.diffusion_batch_mul

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, targets=None, mask=None):
        b, t, dim = idx.size()  # dim = x, y, w, h, onehot(8) = 12
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        # if we are given some desired targets also calculate the loss
        ce_loss = None
        diffusion_loss = None

        cont_dim = 4
        disc_dim = 8

        if targets is not None:
            targets_disc = targets[:, :, cont_dim:]    # [batch_size, max_length, disc_dim=8] (one-hot)
            targets_cont = targets[:, :, :cont_dim]    # [batch_size, max_length, cont_dim=4]

            # Discrete parts
            logits = self.head(x)  # [batch_size, max_length, disc_dim]
            logits_flat = logits.reshape(-1, disc_dim)  # [batch_size * max_length, disc_dim]
            targets_disc_flat = targets_disc.reshape(-1, disc_dim)  # [batch_size * max_length, disc_dim]

            mask_flat = mask.reshape(-1)  # [batch_size * max_length]
            logits_masked = logits_flat[mask_flat]
            targets_disc_masked = targets_disc_flat[mask_flat]

            ce_loss = F.cross_entropy(
                logits_masked.view(-1, logits_masked.size(-1)),
                targets_disc_masked.view(-1, targets_disc_masked.size(-1)),
            )

            # Continuous part
            x_flat = x.reshape(-1, x.size(-1))  # [batch_size * max_length, n_embd]
            x_flat = x_flat.repeat(self.diffusion_batch_mul, 1)  # [batch_size * max_length * diffusion_batch_mul, n_embd]
            targets_cont_flat = targets_cont.reshape(-1, targets_cont.size(-1))
            targets_cont_flat = targets_cont_flat.repeat(self.diffusion_batch_mul, 1) # [batch_size * max_length * diffusion_batch_mul, cont_dim=4]

            diffusion_loss = self.diffloss(z=x_flat, target=targets_cont_flat)
            return ce_loss, diffusion_loss
        
        else:
            # Discrete part
            logits = self.head(x)  # [batch_size, max_length, disc_dim]
            probs_disc = F.softmax(logits, dim=-1)  # [b, t, disc_dim]
            _, top_disc = torch.topk(probs_disc, k=1, dim=-1)  # [b, t, 1]

            # Continuous part
            flat_x = x.reshape(-1, x.size(-1))  # [b*t, n_embd]
            sampled_cont = self.diffloss.sample(
                z=flat_x,
                temperature=1.0,
                cfg=1.0,
            )   # [b*t, cont_dim]
            sampled_cont = sampled_cont.reshape(b, t, cont_dim)

            processed_logits = torch.cat(
                [
                    top_disc.float(),  # [b, t, 1]
                    sampled_cont,  # [b, t, 4]
                ],
                dim=-1,
            )

            return processed_logits  # [b, t, 5]
