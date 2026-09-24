#!/usr/bin/env python
"""
Train the minimal SpeechLLM.
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer, get_cosine_schedule_with_warmup

from data import AudioTextDataset, Collator, manifest_name, split_manifest_entries
from modeling import SpeechLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s")
logger = logging.getLogger(__name__)


# -- config --

@dataclass
class Config:
    # model
    llm_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    encoder_id: str = "openai/whisper-large-v3"
    n_downsample: int = 2
    adapter_hidden_dim: int = 0
    freeze_encoder: bool = True
    dtype: str = "bfloat16"  # "float16" on pre-Ampere (T4)
    gradient_checkpointing: bool = False

    # LoRA on the LLM (the connector always trains in full, the encoder only
    # when freeze_encoder is false)
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # tokens
    audio_locator: str = "<|AUDIO|>"
    placeholder_token: str = "<|vision_pad|>"  # any reserved token of your LLM

    # data
    # entries: "path.jsonl" (ratio 1.0) or {path: ..., ratio: 0.3}; ratios apply to training
    # only, and a fractional ratio draws a fresh random subset every epoch
    train_manifest: List[Any] = field(default_factory=lambda: [
        "data/Librispeech_self-generation/train_audiobook_respond.jsonl"])
    # always full; each gets its own val/<name>/loss curve
    val_manifest: List[str] = field(default_factory=lambda: [
        "data/Librispeech-dev-test/dev-clean_asr.jsonl",
        "data/Librispeech-dev-test/dev-clean_gender.jsonl"])
    audio_root: str = ""  # prefixed to each audio_filepath; "" = paths are relative to cwd
    batch_size: int = 10  # per GPU
    num_workers: int = 8
    max_seq_length: int = 1024
    max_audio_seconds: float = 30.0

    # optim
    lr: float = 1e-4
    encoder_lr: Optional[float] = None  # None = same as lr; ignored when freeze_encoder
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 500

    # trainer
    exp_dir: str = "exp/selfgen"
    max_epochs: int = 3
    max_steps: int = -1  # takes precedence over max_epochs
    limit_train_batches: float = 1.0  # <1.0 = fraction, >=1 = number of batches
    limit_val_batches: float = 1.0
    devices: int = 8
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: float = 0.25
    log_every_n_steps: int = 10
    save_every_n_steps: int = 1000
    keep_best_n_checkpoints: int = 3  # lowest `checkpoint_monitor`, saved after each validation
    # which validation loss picks the best checkpoints: "val/loss" is the mean over all
    # val sets; "val/<manifest name>/loss" is one task, e.g. "val/dev-clean_asr/loss"
    checkpoint_monitor: str = "val/dev-clean_asr/loss"
    wandb_project: str = "desta-tutorial"
    wandb_run_name: str = ""
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        known = {f.name: f for f in fields(cls)}
        unknown = set(values) - set(known)
        assert not unknown, f"{path}: unknown config keys {sorted(unknown)}"
        for key, value in values.items():
            # yaml has no tuples; keep tuple-typed fields as tuples
            if isinstance(value, list) and str(known[key].type).startswith("typing.Tuple"):
                values[key] = tuple(value)
        return cls(**values)


# -- lightning --

class PeakMemory(pl.Callback):
    """Log peak allocated VRAM. Freezing the backbones saves optimizer state, not
    activations, so this is the number that actually decides your batch size."""

    def on_train_batch_end(self, trainer, pl_module, *args):
        if torch.cuda.is_available() and trainer.global_step % 20 == 0:
            pl_module.log("mem/peak_gb", torch.cuda.max_memory_allocated() / 2 ** 30,
                          prog_bar=True, rank_zero_only=True)


class Throughput(pl.Callback):
    """Every `every` steps, log samples/s (all GPUs) and the time one epoch will take."""

    def __init__(self, every: int = 50):
        self.every = every
        self.start = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step == 5:  # skip warm-up / compilation / first-batch noise
            self.start, self.start_step = time.time(), step
        elif self.start and step > self.start_step and step % self.every == 0:
            steps_per_s = (step - self.start_step) / (time.time() - self.start)
            samples_per_s = steps_per_s * pl_module.cfg.batch_size * trainer.world_size \
                * trainer.accumulate_grad_batches
            steps_per_epoch = trainer.num_training_batches // trainer.accumulate_grad_batches
            logger.info("step %d: %.2f steps/s, %.1f samples/s, one epoch (%d steps) = %.2f h",
                        step, steps_per_s, samples_per_s, steps_per_epoch,
                        steps_per_epoch / steps_per_s / 3600)
            pl_module.log("perf/samples_per_s", samples_per_s, rank_zero_only=True)


class SpeechLLMModule(pl.LightningModule):
    """Next-token prediction on the answer span. That is the whole objective."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(asdict(cfg))
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # so the chat template's <|AUDIO|> survives tokenize() as one token; it is
        # always expanded into placeholders before ids are produced, so this id
        # never reaches the (un-resized) LLM embedding table
        self.tokenizer.add_tokens([cfg.audio_locator])
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.encoder_id)

        self.model = SpeechLLM(
            llm_id=cfg.llm_id,
            encoder_id=cfg.encoder_id,
            n_downsample=cfg.n_downsample,
            adapter_hidden_dim=cfg.adapter_hidden_dim,
            dtype=getattr(torch, cfg.dtype),
            gradient_checkpointing=cfg.gradient_checkpointing,
            freeze_encoder=cfg.freeze_encoder,
            use_lora=True,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=cfg.lora_target_modules,
        )

    # -- steps --
    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train/loss", loss, prog_bar=True, sync_dist=True,
                 batch_size=batch["input_ids"].size(0))
        self.log("train/ppl", torch.exp(loss), prog_bar=True, sync_dist=True,
                 batch_size=batch["input_ids"].size(0))
        return loss

    # one val loader per manifest, so each task gets its own curve: val/<name>/loss
    def _val_names(self):
        return [manifest_name(m) for m in self.cfg.val_manifest]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.model(**batch).loss
        self.log(f"val/{self._val_names()[dataloader_idx]}/loss", loss, sync_dist=True,
                 add_dataloader_idx=False, batch_size=batch["input_ids"].size(0))
        return loss

    def on_validation_epoch_end(self):
        # val/loss = unweighted mean over the val sets; this is what checkpoints monitor
        losses = [self.trainer.callback_metrics[f"val/{name}/loss"] for name in self._val_names()
                  if f"val/{name}/loss" in self.trainer.callback_metrics]
        if losses:
            self.log("val/loss", torch.stack(losses).mean(), prog_bar=True, sync_dist=True)

    # -- data --
    def _dataloader(self, dataset, shuffle):
        collator = Collator(
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            audio_locator=self.cfg.audio_locator,
            placeholder_token=self.cfg.placeholder_token,
            max_seq_length=self.cfg.max_seq_length,
            max_audio_seconds=self.cfg.max_audio_seconds,
            n_downsample=self.cfg.n_downsample,
        )
        return DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collator, drop_last=shuffle,
        )

    def train_dataloader(self):
        if not self.cfg.train_manifest:
            return None
        # built once (indexing is the slow part); with fractional ratios Lightning asks
        # for a new dataloader every epoch and we redraw the subset for that epoch
        if not hasattr(self, "_train_dataset"):
            paths, ratios = split_manifest_entries(self.cfg.train_manifest)
            self._train_dataset = AudioTextDataset(paths, self.cfg.audio_root, ratios, seed=self.cfg.seed)
        # also on the first call after --resume, which may start at epoch > 0
        if self._train_dataset.needs_resampling and self.current_epoch > 0:
            self._train_dataset.resample(self.current_epoch)
            logger.info("epoch %d: resampled %d training rows",
                        self.current_epoch, len(self._train_dataset))
        return self._dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        return [self._dataloader(AudioTextDataset([m], self.cfg.audio_root), shuffle=False)
                for m in self.cfg.val_manifest]

    # -- optim --
    def configure_optimizers(self):
        """Two groups: the pretrained encoder (`encoder_lr`) and everything else that
        trains -- connector + LoRA (`lr`). The cosine schedule scales both."""
        encoder_lr = self.cfg.lr if self.cfg.encoder_lr is None else self.cfg.encoder_lr
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        encoder = [p for n, p in named if n.startswith("encoder.")]
        other = [p for n, p in named if not n.startswith("encoder.")]
        groups = [{"params": other, "lr": self.cfg.lr, "name": "connector+lora"}]
        if encoder:
            groups.append({"params": encoder, "lr": encoder_lr, "name": "encoder"})
        for g in groups:
            logger.info("optim group %-15s lr=%.1e  %.1fM params", g["name"], g["lr"],
                        sum(p.numel() for p in g["params"]) / 1e6)
        optimizer = torch.optim.AdamW(
            groups, lr=self.cfg.lr, betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # -- checkpoints --
    # Save the trainable parameters only: a few tens of MB per checkpoint instead of ~11 GB
    # of frozen weights we already have on disk. The matching strict=False on
    # load is what lets `--resume` work against those partial checkpoints.
    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        keep = {f"model.{n}" for n, p in self.model.named_parameters() if p.requires_grad}
        return {k: v for k, v in full.items() if k in keep}

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return super().load_state_dict(state_dict, strict=False)


# -- main --

def parse_args() -> Tuple[Config, Optional[str]]:  # (config, resume ckpt)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="yaml file whose keys are `Config` fields")
    # no defaults below: an unset flag must not overwrite the yaml
    parser.add_argument("--train-manifest", nargs="+")
    parser.add_argument("--val-manifest", nargs="+")
    parser.add_argument("--audio-root")
    parser.add_argument("--exp-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-train-batches", type=float)
    parser.add_argument("--limit-val-batches", type=float)
    parser.add_argument("--val-check-interval", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-audio-seconds", type=float)
    parser.add_argument("--adapter-hidden-dim", type=int)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--encoder-lr", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-target-modules", nargs="+")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--resume", default=None, help="path to a .ckpt to resume from")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    for key, value in vars(args).items():
        # `is None` rather than a truthiness test: --max-steps 0 and --devices 0
        # are meaningful, and `0 == False` would otherwise swallow them
        if key in ("config", "resume") or value is None or value is False:
            continue
        setattr(cfg, key, tuple(value) if key == "lora_target_modules" else value)
    return cfg, args.resume


def main():
    cfg, resume = parse_args()
    pl.seed_everything(cfg.seed)
    logger.info("config: %s", cfg)

    # Single stage: encoder, connector and LoRA all train from step 0. Do not flip requires_grad mid-run --
    # DDP fixes its set of reduced parameters when it wraps the model.
    # fail now, not after the first validation, if the monitored metric will never exist
    split_manifest_entries(cfg.train_manifest)  # validate entries early
    assert all(isinstance(m, str) for m in cfg.val_manifest), "val_manifest takes plain paths (no ratio)"
    val_names = [manifest_name(m) for m in cfg.val_manifest]
    valid = ["val/loss"] + [f"val/{n}/loss" for n in val_names]
    assert cfg.checkpoint_monitor in valid, \
        f"checkpoint_monitor {cfg.checkpoint_monitor!r} is not logged; choose from {valid}"

    module = SpeechLLMModule(cfg)

    logger_ = True  # TensorBoard under exp_dir
    if cfg.wandb_project:
        logger_ = WandbLogger(project=cfg.wandb_project,
                              name=cfg.wandb_run_name or os.path.basename(cfg.exp_dir),
                              save_dir=cfg.exp_dir, config=asdict(cfg))

    trainer = pl.Trainer(
        logger=logger_,
        default_root_dir=cfg.exp_dir,
        accelerator="auto",
        devices=cfg.devices,
        num_nodes=cfg.num_nodes,
        strategy="ddp" if (cfg.devices > 1 or cfg.num_nodes > 1) else "auto",
        precision={"bfloat16": "bf16-mixed",
                   "float16": "16-mixed",
                   "float32": "32-true"}[cfg.dtype],
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        limit_train_batches=cfg.limit_train_batches,
        # a fractional mixing ratio redraws its subset each epoch (see train_dataloader)
        reload_dataloaders_every_n_epochs=int(any(
            r != int(r) for r in split_manifest_entries(cfg.train_manifest)[1])),
        limit_val_batches=cfg.limit_val_batches,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.gradient_clip_val,
        val_check_interval=cfg.val_check_interval if cfg.val_manifest else 1.0,
        num_sanity_val_steps=0,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=[
            # best checkpoints by `checkpoint_monitor`, written right after each validation
            ModelCheckpoint(dirpath=f"{cfg.exp_dir}/checkpoints",
                            filename="best-epoch{epoch}-step{step}-val{" + cfg.checkpoint_monitor + ":.4f}",
                            auto_insert_metric_name=False, monitor=cfg.checkpoint_monitor, mode="min",
                            save_top_k=cfg.keep_best_n_checkpoints),
            # rolling checkpoint for --resume: only the newest is kept
            ModelCheckpoint(dirpath=f"{cfg.exp_dir}/checkpoints", filename="latest-step{step}",
                            auto_insert_metric_name=False, save_top_k=1,
                            every_n_train_steps=cfg.save_every_n_steps),
            LearningRateMonitor(logging_interval="step"),
            PeakMemory(),
            Throughput(),
        ],
    )
    trainer.fit(module, ckpt_path=resume)


if __name__ == "__main__":
    main()
