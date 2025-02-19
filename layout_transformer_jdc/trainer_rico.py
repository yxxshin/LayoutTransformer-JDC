"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import logging
import math
import os

import numpy as np
import torch
import wandb
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils import sample

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader
    loss_weight = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        print("Using wandb")
        wandb.init(project="LayoutTransformer", name=args.exp)
        wandb.config.update(args)

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.pad_token = torch.tensor(
            [0.0] * 4 + [0] * 13 + [0, 0, 1], dtype=torch.float32
        )

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f"checkpoint_{epoch}.pth")
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y, mask) in pbar:

                if epoch == 0 and not is_train:
                    self.fixed_x = x[: min(4, len(x))]
                    self.fixed_y = y[: min(4, len(y))]

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()
                    ce_loss, diffusion_loss = model(x, y, mask)
                    loss = (ce_loss + config.loss_weight * diffusion_loss) / (1 + config.loss_weight)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()
                    self.iters += 1
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.iters < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(self.iters) / float(
                                max(1, config.warmup_iters)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(self.iters - config.warmup_iters) / float(
                                max(1, config.final_iters - config.warmup_iters)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    wandb.log(
                        {
                            "train loss": loss.item(),
                            "ce loss": ce_loss.item(),
                            "diffuion loss": diffusion_loss.item(),
                            "lr": lr,
                            "epoch": epoch + 1,
                        },
                        step=self.iters,
                    )
                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. ce loss {ce_loss.item():.5f}. diff loss {diffusion_loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({"test loss": test_loss}, step=self.iters)
                return test_loss

        best_loss = float("inf")
        for epoch in range(config.max_epochs):
            run_epoch("train")
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch("test")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_dir is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()
            if epoch % 50 == 0:
                self.save_checkpoint(epoch)

            # sample from the model
            if (
                self.config.samples_dir is not None
                and (epoch + 1) % self.config.sample_every == 0
            ):
                # import ipdb; ipdb.set_trace()
                # inputs
                # layouts = self.fixed_x.detach().cpu().numpy()
                fixed_x_categories = self.fixed_x[:, :, 4:].argmax(dim=2, keepdim=True)
                fixed_x_coords = self.fixed_x[:, :, :4]
                layouts = (
                    torch.cat((fixed_x_categories, fixed_x_coords), dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                input_layouts = [
                    self.train_dataset.render(layout) for layout in layouts
                ]

                # reconstruction
                x_cond = self.fixed_x.to(self.device)
                processed_logits = model(
                    x_cond
                )  # Already passed diffloss and softmax/topk

                layouts = processed_logits.detach().cpu().numpy()

                recon_layouts = [
                    self.train_dataset.render(layout) for layout in layouts
                ]

                wandb.log(
                    {
                        "input_layouts": [
                            wandb.Image(pil, caption=f"input_{epoch:02d}_{i:02d}.png")
                            for i, pil in enumerate(input_layouts)
                        ],
                        "recon_layouts": [
                            wandb.Image(pil, caption=f"recon_{epoch:02d}_{i:02d}.png")
                            for i, pil in enumerate(recon_layouts)
                        ],
                    },
                    step=self.iters,
                )
