#!/usr/bin/env python
"""
ASR evaluation (WER) on a LibriSpeech `*_asr.jsonl` manifest.

    python example/evaluate/evaluate_asr.py --ckpt <ckpt> \
        --manifest data/Librispeech-dev-test/test-clean_asr.jsonl \
        --audio-root /path/to/data --prompt answer_fmt

The manifest supplies the audio and the reference transcript; `--prompt` replaces the
prompt in every row, so one manifest covers both prompts:

    default     Transcribe the speech into text
    answer_fmt  Transcribe the speech word for word. Output only the transcription,
                with no explanation, in this format:
                Answer: "<transcription>"

`default` is what the task-specific model was trained on. A model trained only on
self-generated replies answers it with an essay about the passage, which is what
`answer_fmt` is for -- it asks for a field the clean-up below can find.

Hypotheses are always scored twice: as generated, and after the rule-based clean-up
in `postprocess()`. The clean-up only helps a model that wraps its transcript in
commentary; on a model that already answers with a bare transcript it costs a little.

WER follows the Open ASR Leaderboard (github.com/huggingface/open_asr_leaderboard):
reference and hypothesis both go through Whisper's `EnglishTextNormalizer`, rows whose
normalized reference is empty are dropped, and WER is corpus-level over the rest.
Clips longer than `--max-audio-seconds` are cut to 30 s by the collator while their
reference still covers the whole utterance; `wer_excluding_long_clips` shows the effect.
"""

import argparse
import collections
import functools
import json
import logging
import os
import re
import sys
import time

import jiwer
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from inference import SpeechLLMForInference  # noqa: E402

logger = logging.getLogger(__name__)

ENCODER_ID = "openai/whisper-large-v3"
MAX_NEW_TOKENS = 256

PROMPTS = {
    # the prompt in the task-specific SFT manifests
    "default": "Transcribe the speech into text",
    # ask for the transcript in a field the clean-up below can find
    "answer_fmt": "Transcribe the speech word for word. Output only the transcription, "
                  'with no explanation, in this format:\nAnswer: "<transcription>"',
}


# -- clean-up --
#
# A model trained on self-generated replies sometimes answers an ASR prompt as if it
# had read the metadata text it was trained on:
#
#     The audio contains the phrase: "Put in the oven and brown to a golden color."
#     not exactly returned calico (Gender: male)
#     I cannot transcribe audio content as no audio file has been provided. ...
#
# These rules look only at the hypothesis, never at the reference.

GENDER_TAG = re.compile(r"\s*\((?:gender\s*:?\s*)?(?:fe)?male\)", re.I)
AUDIO_TAG = re.compile(r"<\s*audio[^>]*>(.*?)<\s*/\s*audio\s*>", re.I | re.S)
ANSWER = re.compile(r"answer\s*:", re.I)
QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")
EXPLANATION = re.compile(
    r"^\s*(the (audio|provided|input|query|transcription)\b|the text (you|provided)\b|"
    r"i'?m sorry|i (cannot|can'?t|can not|don'?t|do not|am unable)|unfortunately|there is no|"
    r"no (speech|audio|transcription)|note:|sorry)"
    r"|<\s*/?\s*audio\b", re.I)
META_WORD = re.compile(r"transcri|audio|content|speech|provide|assist|input|request|tag", re.I)


def postprocess(hyp):
    """hypothesis -> (cleaned hypothesis, which rule fired)."""
    text = hyp
    # 1. an `Answer:` field: keep what follows it, up to the first blank line
    found_answer = ANSWER.search(text)
    if found_answer:
        text = ANSWER.split(text)[-1].strip().split("\n\n")[0]
        text = text.strip().strip("`\"“”'").strip()
    # 2. the transcript wrapped in the <audio>…</audio> format it was trained on
    inner = AUDIO_TAG.search(text)
    found_audio = bool(inner) and len(inner.group(1).strip()) >= 0.5 * len(
        GENDER_TAG.sub("", text).strip())
    if found_audio:
        text = inner.group(1)
    # 3. speaker tags and markdown
    text = GENDER_TAG.sub("", text).replace("**", "").strip()
    # 4. output that reads like an explanation *about* the task: keep the longest quoted
    #    span if there is one, else nothing. Both conditions matter -- "i don't anticipate"
    #    is a real transcript, not a refusal.
    if not (EXPLANATION.search(text) and META_WORD.search(text)):
        return text, "audio_tag" if found_audio else ("answer" if found_answer else "kept")
    quotes = [GENDER_TAG.sub("", q).strip() for q in QUOTED.findall(text)]
    quotes = [q for q in quotes if re.search(r"[A-Za-z]", q)
              and not (EXPLANATION.search(q) and META_WORD.search(q))]
    if quotes:
        return max(quotes, key=len), "quote"
    return "", "emptied"


# -- scoring --

@functools.lru_cache(maxsize=1)
def openasr_normalizer():
    """Whisper's EnglishTextNormalizer plus its English spelling mapping."""
    from transformers import WhisperTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    mapping = WhisperTokenizer.from_pretrained(ENCODER_ID).english_spelling_normalizer
    assert mapping, f"{ENCODER_ID} tokenizer has no english spelling mapping"
    return EnglishTextNormalizer(mapping)


def wer_of(rows, key, max_audio_seconds):
    norm = openasr_normalizer()
    refs = [norm(r["reference"]) for r in rows]
    hyps = [norm(r[key]) for r in rows]
    keep = [i for i, ref in enumerate(refs) if ref.strip()]   # leaderboard drops empty refs
    short = [i for i in keep if rows[i]["duration"] <= max_audio_seconds]
    score = lambda idx: round(100 * jiwer.wer([refs[i] for i in idx], [hyps[i] for i in idx]), 2)
    return {"n_scored": len(keep), "wer": score(keep),
            "wer_excluding_long_clips": score(short),
            "empty_hypotheses": sum(not hyps[i].strip() for i in keep)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="a *_asr.jsonl manifest")
    ap.add_argument("--prompt", default="default", choices=list(PROMPTS))
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--out-dir", help="default: results/<ckpt>_<prompt>/")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, help="first N rows (quick check)")
    ap.add_argument("--max-audio-seconds", type=float, default=30.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stem = os.path.basename(args.manifest).removesuffix(".jsonl")
    out_dir = args.out_dir or os.path.join(
        ROOT, "results", f"{os.path.basename(args.ckpt).removesuffix('.ckpt')}_{args.prompt}")

    # -- load the manifest, swapping in the chosen prompt ----------------------
    prompt = PROMPTS[args.prompt]
    rows = []
    with open(args.manifest) as f:
        for line in f:
            if args.limit and len(rows) >= args.limit:
                break
            row = json.loads(line)
            path = os.path.join(args.audio_root, row["audios"][0]["audio_filepath"])
            rows.append({"audio_filepath": row["audios"][0]["audio_filepath"], "path": path,
                         "duration": sf.info(path).frames / sf.info(path).samplerate,
                         "reference": row.get("response") or row.get("target")})

    # -- generate -------------------------------------------------------------
    pipe = SpeechLLMForInference.from_checkpoint(args.ckpt)
    logger.info("%s: %d rows, prompt %r", stem, len(rows), prompt)
    # longest first: similar lengths share a batch (less padding), and an OOM shows up at once
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["duration"])
    start = time.time()
    for n, b in enumerate(range(0, len(order), args.batch_size)):
        idx = order[b:b + args.batch_size]
        convs = [[{"role": "user", "content": f"<audio><|AUDIO|></audio>\n\n{prompt}",
                   "audios": [{"audio": rows[i]["path"]}]}] for i in idx]
        hyps = pipe.generate(convs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        for i, hyp in zip(idx, hyps):
            rows[i]["hypothesis_raw"] = hyp
            rows[i]["hypothesis"], rows[i]["postprocess"] = postprocess(hyp)
        if n % 10 == 0:
            done = min(b + args.batch_size, len(order))
            logger.info("%s: %d/%d  %.1f utt/s", stem, done, len(order), done / (time.time() - start))

    # -- score, as generated and after the clean-up ---------------------------
    summary = {"manifest": args.manifest, "checkpoint": args.ckpt, "prompt": args.prompt,
               "prompt_text": prompt, "n": len(rows),
               "n_over_max_audio_seconds": sum(r["duration"] > args.max_audio_seconds for r in rows),
               "seconds": round(time.time() - start, 1),
               "raw": wer_of(rows, "hypothesis_raw", args.max_audio_seconds),
               "postprocessed": wer_of(rows, "hypothesis", args.max_audio_seconds),
               "postprocess_actions": dict(collections.Counter(r["postprocess"] for r in rows))}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "path"},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"{stem}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{stem}  --prompt {args.prompt}  ({len(rows)} utterances)")
    print(f"  WER as generated      {summary['raw']['wer']:6.2f}")
    print(f"  WER after clean-up    {summary['postprocessed']['wer']:6.2f}"
          f"   {summary['postprocess_actions']}")
    print(f"\nresults: {out_dir}")


if __name__ == "__main__":
    main()
