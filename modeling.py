"""
A minimal SpeechLLM: Whisper encoder + connector + LLM with LoRA.

    audio ──► Whisper encoder (trained or frozen) ──► connector (trained) ──┐
                                                                            ├──► LLM + LoRA ──► text
    text  ────────────────────────────────────────► embedding table ────────┘
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, WhisperModel

logger = logging.getLogger(__name__)


class ConcatMLPConnector(nn.Module):
    """Whisper hidden states → LLM-width embeddings by frame concatenation.
    """

    def __init__(self, d_encoder: int, d_llm: int, n_downsample: int = 2,
                 hidden_dim: int = 0):
        super().__init__()
        self.n_downsample = n_downsample
        hidden_dim = hidden_dim or d_llm
        self.proj = nn.Sequential(
            nn.LayerNorm(d_encoder * n_downsample),
            nn.Linear(d_encoder * n_downsample, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_llm),
        )

    def forward(self, hidden_states):                          # (B, T, d_encoder)
        B, T, D = hidden_states.shape
        k = self.n_downsample
        pad = (-T) % k
        if pad:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad))
        stacked = hidden_states.reshape(B, (T + pad) // k, D * k)
        return self.proj(stacked)                              # (B, ceil(T/k), d_llm)


class SpeechLLM(nn.Module):
    def __init__(self, llm_id: str, encoder_id: str,
                 n_downsample: int = 2, dtype=torch.bfloat16,
                 adapter_hidden_dim: int = 0,
                 gradient_checkpointing: bool = False, cache_dir=None,
                 freeze_encoder: bool = True, use_lora: bool = False,
                 lora_rank: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05,
                 lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=dtype, cache_dir=cache_dir)
        # We only ever need the encoder; loading WhisperModel and keeping .encoder
        # lets the decoder weights fall out of scope immediately.
        self.encoder = WhisperModel.from_pretrained(
            encoder_id, torch_dtype=dtype, cache_dir=cache_dir).encoder

        self.connector = ConcatMLPConnector(
            d_encoder=self.encoder.config.d_model,
            d_llm=self.llm.config.hidden_size,
            n_downsample=n_downsample,
            hidden_dim=adapter_hidden_dim,
        )
        logger.info("connector: %.1fM params",
                    sum(p.numel() for p in self.connector.parameters()) / 1e6)

        for param in self.llm.parameters():
            param.requires_grad = False

        self.use_lora = use_lora
        if use_lora:
            from peft import LoraConfig ,get_peft_model
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules), bias="none",
                task_type="CAUSAL_LM",
            )

            self.llm = get_peft_model(self.llm, lora_config).base_model.model
            n_lora = sum(p.numel() for n, p in self.llm.named_parameters() if "lora_" in n)
            assert n_lora > 0, f"LoRA matched no modules in {lora_target_modules}"
            logger.info("LoRA r=%d on %s: %.1fM adapter params",
                        lora_rank, list(lora_target_modules), n_lora / 1e6)

        self.freeze_encoder = freeze_encoder
        for param in self.encoder.parameters():
            param.requires_grad = not freeze_encoder

        if not freeze_encoder:
            self.encoder.float()

        if gradient_checkpointing:
            ckpt_kwargs = {"use_reentrant": False}
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=ckpt_kwargs)
            if not freeze_encoder:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=ckpt_kwargs)

        dtypes = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                part = name.split(".")[0]
                dtypes.setdefault(part, set()).add(str(param.dtype).replace("torch.", ""))
        logger.info("trainable dtypes: %s", {k: sorted(v) for k, v in sorted(dtypes.items())})

        by_part = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                by_part[name.split(".")[0]] = by_part.get(name.split(".")[0], 0) + param.numel()
        trainable = sum(by_part.values())
        total = sum(p.numel() for p in self.parameters())
        logger.info("trainable %.1fM / %.1fM total (%.2f%%) -- %s",
                    trainable / 1e6, total / 1e6, 100 * trainable / total,
                    {k: f"{v/1e6:.1f}M" for k, v in sorted(by_part.items())})

    def encode_audio(self, input_features):
        """mel features → speech embeddings in the LLM's width.
        """
        input_features = input_features.to(self.encoder.conv1.weight.dtype)
        return self.connector(self.encoder(input_features).last_hidden_state)

    def build_inputs_embeds(self, input_ids, speech_features, audio_lengths, start_positions):
        """Overwrite each audio's placeholder embeddings with its speech features."""
        embed = self.llm.get_input_embeddings()
        inputs_embeds = embed(input_ids).clone()

        for k, (row, start) in enumerate(start_positions):
            segment = speech_features[k, :audio_lengths[k]]
            end = start + segment.size(0)
            assert end <= inputs_embeds.size(1), (
                f"audio {k} needs slots [{start}:{end}] but the sequence is "
                f"{inputs_embeds.size(1)} long")
            # outside autocast (inference) the float32 connector output must be
            # cast to the LLM's embedding dtype before the in-place write
            inputs_embeds[row, start:end] = segment.to(inputs_embeds.dtype)

        return inputs_embeds

    def forward(self, input_ids, attention_mask, input_features, audio_lengths,
                start_positions, labels=None):
        speech_features = self.encode_audio(input_features)
        inputs_embeds = self.build_inputs_embeds(
            input_ids, speech_features, audio_lengths, start_positions)
        return self.llm(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels)

    @torch.no_grad()
    def generate(self, batch, tokenizer, **generation_kwargs):
        """Greedy/sampled decoding from the same batch dict used for training.

        Note we pass `inputs_embeds`, so `generate` returns only the newly
        generated ids -- there is no prompt prefix to strip.
        """
        speech_features = self.encode_audio(batch["input_features"])
        inputs_embeds = self.build_inputs_embeds(
            batch["input_ids"], speech_features, batch["audio_lengths"],
            batch["start_positions"])
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

    def trainable_state_dict(self):
        """Only the connector -- a few tens of MB instead of the full ~11 GB."""
        return {name: param.detach().clone()
                for name, param in self.named_parameters() if param.requires_grad}
