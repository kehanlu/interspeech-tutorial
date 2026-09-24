#!/usr/bin/env python
"""
Speaker-gender evaluation (accuracy) on a LibriSpeech `*_gender.jsonl` manifest.

    python example/evaluate/evaluate_gender.py --ckpt <ckpt> \
        --manifest data/Librispeech-dev-test/test-clean_gender.jsonl \
        --audio-root /path/to/data --prompt mcq

Same shape as `evaluate_asr.py`: the manifest supplies the audio and the reference
label, `--prompt` replaces the prompt in every row:

    default  Identify the gender of speaker
    mcq      The audio is a passage read aloud from a book. Is the speaker male or
             female? Answer with one word.

`default` is what the task-specific model was trained on. A model trained only on
self-generated replies answers it with a sentence about the audio, and the label has
to be dug out of the text -- often it is not there at all. `mcq` puts the two options
in the question and pins the answer down to one word.

A prediction that contains neither "male" nor "female" is counted as `unknown`, and
counted as wrong.
"""

import argparse
import collections
import json
import logging
import os
import re
import sys
import time

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from inference import SpeechLLMForInference  # noqa: E402

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 8

PROMPTS = {
    # the prompt in the task-specific SFT manifests
    "default": "Identify the gender of speaker",
    # the same question as a two-way multiple choice, with the answer length pinned down
    "mcq": "The audio is a passage read aloud from a book. "
           "Is the speaker male or female? Answer with one word.",
}


def to_label(text: str) -> str:
    """Pull a gender label out of whatever the model said."""
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split()
    for label in ("female", "male"):   # "female" first: it contains "male"
        if any(w == label or w.startswith(label) for w in words):
            return label
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="a *_gender.jsonl manifest")
    ap.add_argument("--prompt", default="default", choices=list(PROMPTS))
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--out-dir", help="default: results/<ckpt>_<prompt>/")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, help="first N rows (quick check)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stem = os.path.basename(args.manifest).removesuffix(".jsonl")
    out_dir = args.out_dir or os.path.join(
        ROOT, "results", f"{os.path.basename(args.ckpt).removesuffix('.ckpt')}_{args.prompt}")

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

    pipe = SpeechLLMForInference.from_checkpoint(args.ckpt)
    logger.info("%s: %d rows, prompt %r", stem, len(rows), prompt)
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["duration"])
    start = time.time()
    for n, b in enumerate(range(0, len(order), args.batch_size)):
        idx = order[b:b + args.batch_size]
        convs = [[{"role": "user", "content": f"<audio><|AUDIO|></audio>\n\n{prompt}",
                   "audios": [{"audio": rows[i]["path"]}]}] for i in idx]
        hyps = pipe.generate(convs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        for i, hyp in zip(idx, hyps):
            rows[i]["hypothesis"] = hyp
            rows[i]["prediction"] = to_label(hyp)
        if n % 10 == 0:
            done = min(b + args.batch_size, len(order))
            logger.info("%s: %d/%d  %.1f utt/s", stem, done, len(order), done / (time.time() - start))

    correct = sum(r["prediction"] == r["reference"] for r in rows)
    predictions = collections.Counter(r["prediction"] for r in rows)
    summary = {"manifest": args.manifest, "checkpoint": args.ckpt, "prompt": args.prompt,
               "prompt_text": prompt, "n": len(rows),
               "seconds": round(time.time() - start, 1),
               "accuracy": round(100 * correct / len(rows), 2),
               "unknown_predictions": predictions["unknown"],
               "predictions": dict(predictions)}

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "path"},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, f"{stem}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{stem}  --prompt {args.prompt}  ({len(rows)} utterances)")
    print(f"  accuracy   {summary['accuracy']:6.2f}"
          f"   ({summary['unknown_predictions']} unknown)")
    print(f"\nresults: {out_dir}")


if __name__ == "__main__":
    main()
