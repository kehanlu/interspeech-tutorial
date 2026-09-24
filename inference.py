#!/usr/bin/env python
"""
Inference for the minimal SpeechLLM -- the tutorial's demo entry point.

    # one file
    python inference.py --ckpt exp/run/checkpoints/last.ckpt --audio sample.flac

    # score a manifest: prints reference vs hypothesis side by side
    python inference.py --ckpt exp/run/checkpoints/last.ckpt \
        --manifest data/Librispeech-dev-test/test-clean_asr.jsonl \
        --audio-root /path/to/data --limit 20

"""

import argparse
import json
import logging
import os

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

from data import Collator
from modeling import SpeechLLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "<audio><|AUDIO|></audio>\n\nTranscribe the speech into text"


class SpeechLLMForInference:
    """Checkpoint in, text out.

    `generate()` takes chat-style messages so the demo reads like a chat call:

        [{"role": "user",
          "content": "<audio><|AUDIO|></audio>\\n\\nTranscribe the speech into text",
          "audios": [{"audio": "sample.flac"}]}]

    Pass a list of those to batch several utterances in one forward pass.
    """

    def __init__(self, model, tokenizer, collator, device, dtype):
        self.model, self.tokenizer = model, tokenizer
        self.collator, self.device, self.dtype = collator, device, dtype

    # -- loading --
    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        dtype = dtype or getattr(torch, hp["dtype"])
        logger.info("checkpoint %s (epoch %s, step %s)",
                    os.path.basename(ckpt_path), ckpt.get("epoch"), ckpt.get("global_step"))
        # n_downsample used to be n_downsample_layers, an exponent: 1 meant 2 frames
        # per token. Map the old key so earlier checkpoints still load.
        n_downsample = hp.get("n_downsample") or 2 ** hp["n_downsample_layers"]
        logger.info("  llm=%s encoder=%s n_downsample=%d frozen_encoder=%s",
                    hp["llm_id"], hp["encoder_id"],
                    n_downsample, hp.get("freeze_encoder", False))

        # rebuilt exactly as SpeechLLMModule.__init__ does, or the ids shift
        tokenizer = AutoTokenizer.from_pretrained(hp["llm_id"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.add_tokens([hp["audio_locator"]])
        feature_extractor = AutoFeatureExtractor.from_pretrained(hp["encoder_id"])

        model = SpeechLLM(
            llm_id=hp["llm_id"], encoder_id=hp["encoder_id"],
            n_downsample=n_downsample, dtype=dtype,
            # .get(): older checkpoints predate these options
            adapter_hidden_dim=hp.get("adapter_hidden_dim", 0),
            freeze_encoder=hp.get("freeze_encoder", False), use_lora=True,
            lora_rank=hp["lora_rank"], lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
            lora_target_modules=hp["lora_target_modules"],
        )
        cls._load_trainable(model, ckpt["state_dict"])
        model.to(device).eval()

        collator = Collator(
            tokenizer=tokenizer, feature_extractor=feature_extractor,
            audio_locator=hp["audio_locator"], placeholder_token=hp["placeholder_token"],
            max_seq_length=hp["max_seq_length"], max_audio_seconds=hp["max_audio_seconds"],
            n_downsample=n_downsample,
            for_generation=True,
        )
        return cls(model, tokenizer, collator, device, dtype)

    @staticmethod
    def _load_trainable(model, state_dict):
        """Load the trainable-only checkpoint, and prove every tensor landed."""
        state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        missing, unexpected = expected - set(state), set(state) - expected
        assert not unexpected, (
            f"{len(unexpected)} tensors in the checkpoint match nothing in the model, "
            f"e.g. {sorted(unexpected)[:3]} -- the architecture does not match")
        assert not missing, (
            f"{len(missing)} trainable tensors were not in the checkpoint, "
            f"e.g. {sorted(missing)[:3]} -- they would stay randomly initialised")
        result = model.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        logger.info("  restored %d trainable tensors (%.1fM params)", len(state),
                    sum(v.numel() for v in state.values()) / 1e6)

    # -- generation --
    def _to_rows(self, conversations):
        """Chat-style messages -> the row dicts `Collator` consumes."""
        rows = []
        for conv in conversations:
            audios = []
            for message in conv:
                for audio in message.get("audios", []):
                    path = audio["audio"] if isinstance(audio, dict) else audio
                    assert os.path.exists(path), f"no such audio: {path}"
                    audios.append({"audio_filepath": path})
            n_locators = sum(m["content"].count(self.collator.audio_locator) for m in conv)
            assert n_locators == len(audios), (
                f"{len(audios)} audios but {n_locators} {self.collator.audio_locator} "
                "in the conversation")
            # strip `audios` before the chat template ever sees it
            rows.append({"messages": [{"role": m["role"], "content": m["content"]} for m in conv],
                         "audios": audios, "target": ""})
        return rows

    @torch.no_grad()
    def generate(self, conversations, max_new_tokens=200, do_sample=False, **generation_kwargs):
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]  # a single conversation
        batch = self.collator(self._to_rows(conversations))

        batch["input_ids"] = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        # SpeechLLM.encode_audio casts features to the encoder's own dtype
        batch["input_features"] = batch["input_features"].to(self.device)

        # The trainable tensors (connector, LoRA) stay in float32 while the frozen encoder and
        # LLM run in `dtype`, exactly as in training -- so the forward pass has to happen under
        # autocast, or layer_norm sees a float32 weight and a float16 activation and raises
        # "expected scalar type Half but found Float".
        # passing inputs_embeds means `generate` returns only the new tokens --
        # there is no prompt prefix to slice off
        with torch.autocast(self.device.split(":")[0], dtype=self.dtype,
                            enabled=not self.device.startswith("cpu")):
            generated = self.model.generate(
                batch, self.tokenizer, max_new_tokens=max_new_tokens,
                do_sample=do_sample, **generation_kwargs)
        return [t.strip() for t in
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)]


# -- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio", nargs="+", help="one or more audio files")
    ap.add_argument("--manifest", help="jsonl to run over instead of --audio")
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--limit", type=int, default=10, help="rows to take from --manifest")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    assert args.audio or args.manifest, "give --audio or --manifest"

    pipe = SpeechLLMForInference.from_checkpoint(args.ckpt, device=args.device)

    if args.audio:
        items = [(os.path.join(args.audio_root, p), None) for p in args.audio]
    else:
        items = []
        with open(args.manifest) as f:
            for line in f:
                if len(items) >= args.limit:
                    break
                row = json.loads(line)
                items.append((os.path.join(args.audio_root,
                                           row["audios"][0]["audio_filepath"]),
                              row.get("response") or row.get("target")))

    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        convs = [[{"role": "user", "content": args.prompt,
                   "audios": [{"audio": path}]}] for path, _ in chunk]
        for (path, reference), hypothesis in zip(chunk, pipe.generate(
                convs, max_new_tokens=args.max_new_tokens)):
            print(f"\n--- {os.path.basename(path)}")
            if reference is not None:
                print(f"  ref: {reference}")
            print(f"  hyp: {hypothesis}")


if __name__ == "__main__":
    main()
