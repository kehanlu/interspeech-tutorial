#!/usr/bin/env python
"""
Build task-specific LibriSpeech manifests (ASR + speaker gender) from the raw corpus.

    python data/prepare_librispeech.py --audio-root /path/to/data     # holds LibriSpeech/

Writes, one row per utterance (the same audio appears in both tasks):

    data/Librispeech_task-specific_SFT/
        train_asr.jsonl       train-clean-100 + train-clean-360 + train-other-500
        train_gender.jsonl
    data/Librispeech-dev-test/
        {dev-clean,dev-other,test-clean,test-other}_{asr,gender}.jsonl

    asr     "Transcribe the speech into text"  -> lowercase transcript
    gender  "Identify the gender of speaker"   -> "female" / "male"

`audio_filepath` is relative to `--audio-root`, so train with the same
`--audio-root /path/to/data`.
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# output file prefix -> (output dir, LibriSpeech subsets merged into it)
SPLITS = {
    "train": ("Librispeech_task-specific_SFT",
              ["train-clean-100", "train-clean-360", "train-other-500"]),
    "dev-clean": ("Librispeech-dev-test", ["dev-clean"]),
    "dev-other": ("Librispeech-dev-test", ["dev-other"]),
    "test-clean": ("Librispeech-dev-test", ["test-clean"]),
    "test-other": ("Librispeech-dev-test", ["test-other"]),
}
TASKS = {
    "asr": "Transcribe the speech into text",
    "gender": "Identify the gender of speaker",
}
GENDER = {"F": "female", "M": "male"}


def read_speakers(root):
    """SPEAKERS.TXT:  ID | SEX | SUBSET | MINUTES | NAME"""
    speakers = {}
    with open(os.path.join(root, "SPEAKERS.TXT")) as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            speaker_id, sex = [field.strip() for field in line.split("|")[:2]]
            speakers[speaker_id] = GENDER[sex]
    return speakers


def read_utterances(root, subset):
    """Yield (speaker_id, relative flac path, transcript) from every *.trans.txt."""
    pattern = os.path.join(root, subset, "*", "*", "*.trans.txt")
    for trans in sorted(glob.glob(pattern)):
        with open(trans) as f:
            for line in f:
                utt_id, text = line.strip().split(" ", 1)
                speaker_id, chapter_id, _ = utt_id.split("-")
                path = os.path.join(os.path.basename(root), subset,
                                    speaker_id, chapter_id, f"{utt_id}.flac")
                yield speaker_id, path, text.lower()


def row(path, prompt, target):
    return {"audios": [{"audio_filepath": path}],
            "messages": [{"role": "user", "content": f"<audio><|AUDIO|></audio>\n\n{prompt}"}],
            "target": target}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", required=True,
                    help="the directory holding the unpacked LibriSpeech/")
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument("--out-root", default=HERE)
    args = ap.parse_args()

    root = os.path.join(args.audio_root, "LibriSpeech")
    speakers = read_speakers(root)

    for split in args.splits:
        out_dir, subsets = SPLITS[split]
        out_dir = os.path.join(args.out_root, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        paths = {task: os.path.join(out_dir, f"{split}_{task}.jsonl") for task in TASKS}
        outputs = {task: open(path, "w") for task, path in paths.items()}
        counts = {"female": 0, "male": 0}
        for subset in subsets:
            for speaker_id, path, text in read_utterances(root, subset):
                gender = speakers[speaker_id]
                outputs["asr"].write(json.dumps(row(path, TASKS["asr"], text)) + "\n")
                outputs["gender"].write(json.dumps(row(path, TASKS["gender"], gender)) + "\n")
                counts[gender] += 1
        for f in outputs.values():
            f.close()

        total = sum(counts.values())
        assert total, f"no utterances found for {split} under {root}"
        print(f"{split:11s} {total:7d} rows  (female {counts['female']}, male {counts['male']})"
              f"  -> {os.path.relpath(out_dir, args.out_root)}/{split}_{{asr,gender}}.jsonl")


if __name__ == "__main__":
    main()
