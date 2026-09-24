"""
Data pipeline for the minimal SpeechLLM.

It sees a plain text token sequence in which every ``<|AUDIO|>`` has been replaced
by exactly as many placeholder tokens as that audio will occupy after encoding.
The model later overwrites those placeholders' *embeddings* with real speech
features (see modeling.py). Nothing else about the LLM changes.

Everything that turns "a manifest row" into "model inputs" happens in `Collator`.
The `Dataset` is a dumb jsonl reader. Keeping it in one place matters here: the
number of placeholders depends on the audio's true duration, which we only know
once the waveform is loaded -- so the expansion has to happen next to the audio.

Manifest format (one JSON object per line)::

    {
      "audios": [
        {"audio_filepath": "ESC50/1-103999-A-30.wav"}
      ],
      "messages": [
        {"role": "user", "content": "What do you hear? <|AUDIO|>"}
      ],
      "target": "A wooden door being knocked on."
    }

`audio_filepath` is relative to `audio_root`. `target` may also be called
`response` (both appear in the DeSTA3 manifests). Extra keys are ignored.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import librosa
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# How many tokens will one audio occupy?
#
# Whisper's feature extractor always pads to 30 s, so we cannot use the padded
# length -- a 5 s clip would reserve 25 s worth of slots. We ask the feature
# extractor for an attention mask over the mel frames instead, and push the true
# length through the same downsampling arithmetic the model uses.

def whisper_frames(mel_len: int) -> int:
    """Whisper's encoder conv2 has stride 2. 3000 mel frames -> 1500 positions."""
    return (mel_len - 1) // 2 + 1


def audio_token_count(mel_len: int, n_downsample: int) -> int:
    """mel frames -> number of LLM token slots this audio needs.

    The connector concatenates every `n_downsample` adjacent frames and right-pads to
    a multiple of that, so the count is ceil(frames / n_downsample), rounding up.
    """
    return -(-whisper_frames(mel_len) // n_downsample)


def expand_audio_locator(tokens: List[str], audio_locator: str,
                         placeholder_token: str, slot_sizes: List[int]):
    """Replace each `audio_locator` with `slot_sizes[i]` placeholder tokens.

    ``["a", "<|AUDIO|>", "b"]`` with ``slot_sizes=[3]`` becomes
    ``["a", "<|pad|>", "<|pad|>", "<|pad|>", "b"]`` and ``start_positions=[1]``.
    """
    out: List[str] = []
    starts: List[int] = []
    sizes = iter(slot_sizes)
    for token in tokens:
        if token == audio_locator:
            starts.append(len(out))
            out.extend([placeholder_token] * next(sizes))
        else:
            out.append(token)
    return out, starts


def resolve_manifest(path: str) -> str:
    """Accept a local path or a huggingface.co/datasets/... resolve URL."""
    prefix = "https://huggingface.co/datasets/"
    if not path.startswith(prefix):
        return path
    from huggingface_hub import hf_hub_download

    parts = path[len(prefix):].split("/")
    repo_id, marker, revision = "/".join(parts[:2]), parts[2], parts[3]
    assert marker == "resolve", f"unsupported HF url: {path}"
    return hf_hub_download(repo_id=repo_id, filename="/".join(parts[4:]),
                           revision=revision, repo_type="dataset")


def resolve_audio_filepath(path: str) -> str:
    if os.path.exists(path):
        return path
    wav = os.path.splitext(path)[0] + ".wav"
    if os.path.exists(wav):
        return wav
    raise FileNotFoundError(path)


def split_manifest_entries(entries):
    """Manifest entries are a path (ratio 1.0) or {"path": ..., "ratio": ...}.

    Returns (paths, ratios).
    """
    paths, ratios = [], []
    for entry in entries:
        if isinstance(entry, str):
            path, ratio = entry, 1.0
        else:
            unknown = set(entry) - {"path", "ratio"}
            assert "path" in entry and not unknown, f"bad manifest entry {entry!r}"
            path, ratio = entry["path"], float(entry.get("ratio", 1.0))
        assert ratio >= 0, f"{path}: ratio must be >= 0, got {ratio}"
        paths.append(path)
        ratios.append(ratio)
    return paths, ratios


def manifest_name(path: str) -> str:
    return os.path.basename(path).removesuffix(".jsonl")


class AudioTextDataset(Dataset):
    """Lazy jsonl reader with per-manifest mixing ratios.
    """

    def __init__(self, manifest_filepaths, audio_root: str, ratios=None, seed: int = 0):
        self.audio_root = audio_root
        self.paths = [resolve_manifest(p) for p in manifest_filepaths]
        self.ratios = list(ratios) if ratios is not None else [1.0] * len(self.paths)
        assert len(self.ratios) == len(self.paths)
        self.seed = seed
        self.rows = []  # per manifest: [(file_idx, byte_offset), ...]
        for file_idx, path in enumerate(self.paths):
            rows = []
            with open(path, "rb") as f:
                offset = 0
                for line in f:
                    if line.strip():
                        rows.append((file_idx, offset))
                    offset += len(line)
            self.rows.append(rows)
        self._handles = {}  # opened lazily, per dataloader worker
        self.resample(epoch=0, log=True)

    @property
    def needs_resampling(self) -> bool:
        """True when some ratio has a fractional part, i.e. a random subset is drawn."""
        return any(r != math.floor(r) for r in self.ratios)

    def resample(self, epoch: int, log: bool = False):
        self.index = []  # (file_idx, byte_offset)
        for file_idx, (rows, ratio) in enumerate(zip(self.rows, self.ratios)):
            whole = math.floor(ratio)
            n_extra = round((ratio - whole) * len(rows))
            # str seed: hashed with sha512, identical across processes and runs
            rng = random.Random(f"{self.seed}-{epoch}-{file_idx}")
            self.index += rows * whole + rng.sample(rows, n_extra)
            if log:
                logger.info("  %-32s ratio %.2f: %d of %d rows", manifest_name(self.paths[file_idx]),
                            ratio, whole * len(rows) + n_extra, len(rows))
        if log:
            logger.info("loaded %d rows from %d manifest(s)", len(self.index), len(self.paths))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        file_idx, offset = self.index[i]
        handle = self._handles.get(file_idx)
        if handle is None:
            handle = self._handles[file_idx] = open(self.paths[file_idx], "rb")
        handle.seek(offset)
        row = json.loads(handle.readline())

        # DeSTA3 manifests use either key; `response` wins when both are present.
        target = row.get("response") or row.get("target")
        assert target, f"row {i} has neither `response` nor `target`"

        audios = [{"audio_filepath": resolve_audio_filepath(
                       os.path.join(self.audio_root, audio["audio_filepath"]))}
                  for audio in row["audios"]]
        return {"messages": row["messages"], "target": target, "audios": audios}


@dataclass
class Collator:
    """Turns a list of rows into the tensors `SpeechLLM.forward` expects.

    Padding is on the **left** so that every sequence's answer ends at the same
    index -- that keeps `generate()` simple. It also means every position we
    record has to be shifted by that sequence's pad length.
    """

    tokenizer: Any
    feature_extractor: Any
    audio_locator: str = "<|AUDIO|>"
    placeholder_token: str = "<|vision_pad|>"
    max_seq_length: int = 1024
    max_audio_seconds: float = 30.0
    n_downsample: int = 2
    # inference: build the prompt only, with no answer appended and no labels.
    # Same slot arithmetic as training -- that is the whole point of reusing this
    # class rather than rebuilding the expansion in the demo script.
    for_generation: bool = False

    def __post_init__(self):
        assert self.tokenizer.padding_side == "left"
        assert self.placeholder_token in self.tokenizer.get_vocab(), (
            f"placeholder_token {self.placeholder_token!r} is not in the tokenizer "
            "vocabulary; pick a reserved/unused token of your LLM")
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        self.pad_id = self.tokenizer.pad_token_id

    def __call__(self, batch):
        # 1. audio -> mel features, and the true (unpadded) length of each
        waveforms = []
        for row in batch:
            for audio in row["audios"]:
                wav, _ = librosa.load(audio["audio_filepath"], sr=SAMPLE_RATE, mono=True)
                waveforms.append(wav[:int(self.max_audio_seconds * SAMPLE_RATE)])

        features = self.feature_extractor(
            waveforms, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True,
        )
        mel_lengths = features["attention_mask"].sum(-1).tolist()
        audio_lengths = [audio_token_count(n, self.n_downsample) for n in mel_lengths]

        # 2. text -> token ids, with <|AUDIO|> expanded to placeholders
        context_ids, target_ids = [], []
        start_positions, audio_index = [], 0

        for row in batch:
            n_audios = len(row["audios"])
            # each audio reserves one slot per speech feature frame
            slot_sizes = audio_lengths[audio_index:audio_index + n_audios]

            prompt = self.tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = self.tokenizer.tokenize(prompt)
            assert tokens.count(self.audio_locator) == n_audios, (
                f"{n_audios} audios but {tokens.count(self.audio_locator)} "
                f"{self.audio_locator} in the prompt")

            tokens, starts = expand_audio_locator(
                tokens, self.audio_locator, self.placeholder_token, slot_sizes)

            # straight to ids -- no convert_tokens_to_string round-trip, so the
            # placeholder can never be re-tokenized into something else
            ctx = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.for_generation:
                tgt = []  # nothing to condition on; the model writes it
            else:
                tgt = self.tokenizer.encode(row["target"], add_special_tokens=False)
                tgt = tgt + [self.tokenizer.eos_token_id]

            assert len(ctx) < self.max_seq_length, (
                f"prompt alone is {len(ctx)} tokens (max_seq_length="
                f"{self.max_seq_length}); shorten the audio or raise the limit")
            tgt = tgt[:self.max_seq_length - len(ctx)]  # only ever truncate the answer

            context_ids.append(ctx)
            target_ids.append(tgt)
            start_positions.append(starts)
            audio_index += n_audios

        # 3. left-pad, build labels, shift the recorded positions
        width = max(len(c) + len(t) for c, t in zip(context_ids, target_ids))
        input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        shifted_starts = []

        for i, (ctx, tgt) in enumerate(zip(context_ids, target_ids)):
            pad = width - len(ctx) - len(tgt)
            input_ids[i, pad:] = torch.tensor(ctx + tgt, dtype=torch.long)
            attention_mask[i, pad:] = 1
            labels[i, pad + len(ctx):] = torch.tensor(tgt, dtype=torch.long)  # answer only
            for start in start_positions[i]:
                shifted_starts.append((i, start + pad))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": features["input_features"],
            "audio_lengths": audio_lengths,  # per audio, after downsampling
            "start_positions": shifted_starts,  # per audio, (row, first placeholder)
        }
