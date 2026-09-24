#!/usr/bin/env python
"""
Self-generation on LibriSpeech with vLLM (offline).

For each utterance the LLM sees only text: the training prompt with the audio
placeholder `<|AUDIO|>` replaced by a metadata string (transcription + speaker gender).
Both come straight from the corpus -- the `*.trans.txt` files and `SPEAKERS.TXT` --
so no other manifest has to be built first.

    generation input:  <audio>{transcription} (Gender: {gender})</audio>\n\nResponse as you are a conversation partner
    training input:    <audio><|AUDIO|></audio>\n\nResponse as you are a conversation partner

Its response becomes the training target, paired with the audio, so the speech model
learns to reproduce what the text LLM said given the same (heard) content.

    python example/self-generation/generate_vllm.py \
        --audio-root /path/to/data \
        --output data/Librispeech_self-generation/train_conversation.jsonl

The output rows follow the manifest format read by data.py:
    {"audios": [...], "messages": [{"role": "user", "content": "<audio><|AUDIO|></audio>\n\n<prompt>"}],
     "target": <LLM response>, "metadata": <metadata string>}

`audio_filepath` is relative to `--audio-root` (the directory holding LibriSpeech/), the
same as the manifests written by data/prepare_librispeech.py, so train with the same
`--audio-root`.
"""

import argparse
import glob
import json
import os

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # same LLM backbone as configs/*.yaml
PROMPT = "Respond directly as you are a natural conversation partner"
METADATA_TEMPLATE = "{transcription} (Gender: {gender})"
AUDIO_LOCATOR = "<|AUDIO|>"
USER_TEMPLATE = "<audio>" + AUDIO_LOCATOR + "</audio>\n\n{prompt}"  # same layout as the SFT manifests
TRAIN_SUBSETS = ["train-clean-100", "train-clean-360", "train-other-500"]
GENDER = {"F": "female", "M": "male"}


def load_librispeech(audio_root, subsets=TRAIN_SUBSETS):
    """One sample per utterance: transcript from *.trans.txt, gender from SPEAKERS.TXT."""
    root = os.path.join(audio_root, "LibriSpeech")
    genders = {}
    with open(os.path.join(root, "SPEAKERS.TXT")) as f:  # ID | SEX | SUBSET | MINUTES | NAME
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            speaker_id, sex = [field.strip() for field in line.split("|")[:2]]
            genders[speaker_id] = GENDER[sex]

    samples = []
    for subset in subsets:
        for trans in sorted(glob.glob(os.path.join(root, subset, "*", "*", "*.trans.txt"))):
            with open(trans) as f:
                for line in f:
                    utt_id, text = line.strip().split(" ", 1)
                    speaker_id, chapter_id, _ = utt_id.split("-")
                    path = os.path.join(os.path.basename(root), subset, speaker_id, chapter_id, f"{utt_id}.flac")
                    samples.append({"audios": [{"audio_filepath": path}],
                                    "transcription": text.lower(), "gender": genders[speaker_id]})
    assert samples, f"no utterances found for {subsets} under {root}"
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", required=True, help="the directory holding the unpacked LibriSpeech/")
    ap.add_argument("--subsets", nargs="+", default=TRAIN_SUBSETS, help="LibriSpeech subsets to generate for")
    ap.add_argument("--output", default="data/Librispeech_self-generation/train_conversation.jsonl")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--limit", type=int, default=None, help="only the first N utterances (quick check)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    # throughput: one 4B model on one H200 -> push concurrency until the KV cache is the limit
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument("--max-model-len", type=int, default=2048, help="prompt (~150 tokens) + --max-tokens")
    ap.add_argument("--max-num-seqs", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="rows per llm.chat call; each chunk is appended to --output so a killed job can resume")
    args = ap.parse_args()

    samples = load_librispeech(args.audio_root, args.subsets)[:args.limit]
    n_done = sum(1 for _ in open(args.output)) if os.path.exists(args.output) else 0
    print(f"{len(samples)} utterances, {n_done} already in {args.output}")

    user_content = USER_TEMPLATE.format(prompt=args.prompt)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              tensor_parallel_size=args.tensor_parallel_size, seed=args.seed)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_tokens, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_truncated = 0
    for start in range(n_done, len(samples), args.chunk_size):
        chunk = samples[start:start + args.chunk_size]
        conversations = []
        for s in chunk:
            s["metadata"] = METADATA_TEMPLATE.format(transcription=s["transcription"], gender=s["gender"])
            conversations.append([{"role": "user", "content": user_content.replace(AUDIO_LOCATOR, s["metadata"])}])
        outputs = llm.chat(conversations, sampling_params)  # batched; applies the chat template

        with open(args.output, "a") as f:
            for s, out in zip(chunk, outputs):
                completion = out.outputs[0]
                n_truncated += completion.finish_reason == "length"
                f.write(json.dumps({
                    "audios": s["audios"],
                    "messages": [{"role": "user", "content": user_content}],
                    "target": completion.text.strip(),
                    "metadata": s["metadata"],
                }, ensure_ascii=False) + "\n")
        print(f"rows {start + len(chunk)}/{len(samples)} written ({n_truncated} hit --max-tokens so far)", flush=True)

        if start == n_done:
            for s, out in list(zip(chunk, outputs))[:3]:
                print(f"\n### Input: {user_content.replace(AUDIO_LOCATOR, s['metadata'])}\n### Response: {out.outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
