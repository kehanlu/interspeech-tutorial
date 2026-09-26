# Interspeech 2026 Tutorial

## Hands-on Session: Training a Speech-aware LLM without Forgetting

 Try your own prompt on our fine-tuned checkpoints.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kehanlu/interspeech-tutorial/blob/main/colab_demo.ipynb) [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://github.com/kehanlu/interspeech-tutorial/blob/main/colab_demo.ipynb)


```
audio ──► Whisper-large-v3 encoder ──► adapter ──┐
                                                 ├──► Qwen3-4B-Instruct-2507 (+ LoRA) ──► text
text  ──────────────────────────► embedding table┘
```


## Table of contents



Steps below, in order:

1. [Download LibriSpeech](#1-download-librispeech)
2. [Prepare the training data](#2-prepare-the-training-data): task-specific SFT, and self-generation
3. [Train](#3-train): one run per kind of training data
4. [Inference and evaluation](#4-inference-and-evaluation): download the trained checkpoints, score them, and try them on single files

To only try the trained models, skip to [section 4](#4-inference-and-evaluation).

## 1. Download LibriSpeech

Everything here uses [LibriSpeech](https://www.openslr.org/12) (audio + labels). Pick a
directory to hold it (this is the `audio_root` the configs refer to) and unpack the
archives there:

```bash
export AUDIO_ROOT=/path/to/data           # the parent of LibriSpeech/
mkdir -p "$AUDIO_ROOT" && cd "$AUDIO_ROOT"

for part in train-clean-100 train-clean-360 train-other-500 \
            dev-clean dev-other test-clean test-other; do
    wget "https://www.openslr.org/resources/12/${part}.tar.gz"
    tar xzf "${part}.tar.gz"             # all of them unpack into ./LibriSpeech/
done
```

Roughly 60 GB in total (train-other-500 alone is 30 GB). To try the pipeline without
that, `dev-clean` and `test-clean` are ~700 MB together and are enough for everything
except training: build their manifests with `--splits dev-clean test-clean` in step 2-1.
The `train` split of step 2-1 always merges all three train subsets; to train on
`train-clean-100` alone, edit `SPLITS` in `data/prepare_librispeech.py` (step 2-2
takes `--subsets train-clean-100` instead).

The result must look like this:

```
$AUDIO_ROOT/
└── LibriSpeech/
    ├── SPEAKERS.TXT
    ├── train-clean-100/  train-clean-360/  train-other-500/
    ├── dev-clean/  dev-other/
    └── test-clean/  test-other/
```

Every manifest stores `audio_filepath` relative to `$AUDIO_ROOT`
(`LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac`), so the manifests stay
valid if you move the corpus; you only change `audio_root`.

## 2. Prepare the training data

Two kinds of training data. Both read only the corpus itself: the `*.trans.txt` files
and `SPEAKERS.TXT`. Both steps write into `data/`:

```
data/
├── prepare_librispeech.py
├── Librispeech_task-specific_SFT/          # step 2-1
│   ├── train_asr.jsonl                     # 281,241 rows
│   └── train_gender.jsonl                  # 281,241 rows
├── Librispeech-dev-test/                   # step 2-1
│   ├── dev-clean_asr.jsonl                 # 2,703 rows
│   ├── dev-clean_gender.jsonl
│   ├── dev-other_{asr,gender}.jsonl
│   ├── test-clean_{asr,gender}.jsonl       # 2,620 rows
│   └── test-other_{asr,gender}.jsonl
└── Librispeech_self-generation/            # step 2-2
    └── train_audiobook_respond.jsonl       # 281,241 rows
```

Every one of those files is JSON Lines in the same format, one utterance per row:

```json
{"audios": [{"audio_filepath": "LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac"}],
 "messages": [{"role": "user", "content": "<audio><|AUDIO|></audio>\n\nIdentify the gender of speaker"}],
 "target": "male"}
```

`audio_filepath` is relative to `audio_root`, `<|AUDIO|>` is where the speech features
are spliced in, and `target` is what the model is trained to produce. The datasets
differ only in the prompt and the target. The examples below all use the same
utterance, `103-1240-0000`.

### 2-1. Task-specific SFT (ASR + speaker gender)

Two supervised tasks read straight off the corpus: transcribe the speech, and name
the speaker's gender.

```bash
python data/prepare_librispeech.py --audio-root "$AUDIO_ROOT"
```

Takes well under a minute: it only reads the `.trans.txt` files and `SPEAKERS.TXT`,
never the audio. Use `--splits dev-clean test-clean` to build a subset.

**`train_asr.jsonl`**, transcribe:

```json
{"audios": [{"audio_filepath": "LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"}],
 "messages": [{"role": "user", "content": "<audio><|AUDIO|></audio>\n\nTranscribe the speech into text"}],
 "target": "chapter one missus rachel lynde is surprised missus rachel lynde lived just where the avonlea main road dipped down into a little hollow fringed with alders and ladies eardrops and traversed by a brook"}
```

**`train_gender.jsonl`**, same audio, different question:

```json
{"audios": [{"audio_filepath": "LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"}],
 "messages": [{"role": "user", "content": "<audio><|AUDIO|></audio>\n\nIdentify the gender of speaker"}],
 "target": "female"}
```

The dev / test files have exactly this shape too. Together they are all
[configs/asr_gender.yaml](configs/asr_gender.yaml) needs. They are also the validation
sets of [configs/selfgen.yaml](configs/selfgen.yaml) and the evaluation sets of section
4, so build at least `dev-clean` and `test-clean` even if you only train on
self-generated data.

### 2-2. Self-generation (vLLM)

The idea of self-generation: instead of writing targets by hand, let the **text** LLM
write them. It is shown a text description of what is in the audio, and its reply
becomes the training target for the speech model. The speech model is therefore
trained towards what its own LLM backbone would have said, rather than towards a style
the backbone has to be dragged into.

```
generation input (text LLM):  <audio>{transcription} (Gender: {gender})</audio>\n\n{prompt}
training input (speech LLM):  <audio><|AUDIO|></audio>\n\n{prompt}
target:                       the text LLM's reply
```

Needs vLLM and one GPU:

```bash
python example/self-generation/generate_vllm.py \
    --audio-root "$AUDIO_ROOT" \
    --output data/Librispeech_self-generation/train_audiobook_respond.jsonl \
    --temperature 0 \
    --prompt "The audio is a passage read aloud from a book. Respond directly as a natural conversation partner. Do not mention the audio, the transcription, or the speaker attributes."
```

`--temperature 0` is greedy decoding. Qwen3-4B-Instruct-2507, 512 max tokens, ~25
minutes for all 281,241 rows on one GPU; add `--limit 50` for a quick check first. On
slurm, [generate_slurm.sh](example/self-generation/generate_slurm.sh) takes the same
arguments.

**`train_audiobook_respond.jsonl`**, same audio again, target written by the text LLM:

```json
{"audios": [{"audio_filepath": "LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac"}],
 "messages": [{"role": "user", "content": "<audio><|AUDIO|></audio>\n\nThe audio is a passage read aloud from a book. Respond directly as a natural conversation partner. Do not mention the audio, the transcription, or the speaker attributes."}],
 "target": "Oh, Missus Rachel Lynde—such a vivid character, isn’t she? I love how she lives in that little hollow by the brook, surrounded by alders and ladies' eardrops. There’s something so peaceful about that setting, like a quiet corner of the world where life moves at its own pace. And the brook—it must be full of little sounds, rustling leaves and the gentle flow of water. Makes you want to sit and just listen.",
 "metadata": "chapter one missus rachel lynde is surprised missus rachel lynde lived just where the avonlea main road dipped down into a little hollow fringed with alders and ladies eardrops and traversed by a brook (Gender: female)"}
```

## 3. Train

One stage, next-token prediction on the target span only (the prompt is masked out).
The Whisper encoder and the LLM stay frozen; only the adapter and LoRA train:

| part | trainable |
|---|---|
| Whisper-large-v3 encoder | frozen |
| adapter (concat 2 frames + 2-layer MLP) | 13.1M |
| Qwen3-4B-Instruct-2507 | frozen except LoRA r=32 on q/k/v/o: 23.6M |

36.7M of 4.7B parameters. Both runs below use exactly this model and recipe; the only
difference between their configs is the training data.

Checkpoints go to `exp/<config>/checkpoints/` (under slurm,
`exp/<config>/<job id>/checkpoints/`).

### 3-1. Task-specific SFT

Trains on `train_asr.jsonl` + `train_gender.jsonl` from step 2-1
([configs/asr_gender.yaml](configs/asr_gender.yaml)).

```bash
# python (8 GPUs on this machine)
python train.py --config configs/asr_gender.yaml --audio-root "$AUDIO_ROOT"

# slurm (1 node, 8 GPUs)
sbatch run_slurm.sh configs/asr_gender.yaml --audio-root "$AUDIO_ROOT"
```

100% of `train_asr` + a fresh random 30% of `train_gender` each epoch (365,613 rows
per epoch). The gender data is downweighted on purpose: training on all of it
overfits gender by epoch 1 and costs about 4% relative ASR.

Reference run: 8×H200, 2 h 02 min, 13,710 steps at 1.89 steps/s, peak 73 GB.

### 3-2. Self-generation

Trains on `train_audiobook_respond.jsonl` from step 2-2, nothing else
([configs/selfgen.yaml](configs/selfgen.yaml)).

```bash
# python (8 GPUs on this machine)
python train.py --config configs/selfgen.yaml --audio-root "$AUDIO_ROOT"

# slurm (1 node, 8 GPUs)
sbatch run_slurm.sh configs/selfgen.yaml --audio-root "$AUDIO_ROOT"
```

**ASR and gender recognition are never trained here.** The model only ever sees
free-form conversational replies; we rely on the instruction-following ability of
Qwen3-4B-Instruct to perform the downstream tasks.

Reference run: 8×H200, 1 h 47 min, 10,541 steps. Train loss ~2.2 → 0.31. The
dev-clean ASR validation loss is lowest at the end of epoch 0 and rises afterwards.
This is expected, since it measures a task this run never trains on.

## 4. Inference and evaluation

### Download the checkpoints

Both checkpoints from section 3 are on the Hugging Face Hub at
[kehanlu/interspeech-tutorial](https://huggingface.co/kehanlu/interspeech-tutorial)
(public, no login needed), together with a few LibriSpeech dev-clean clips. That is
enough to try the models without doing steps 1 to 3:

| folder | trained in | size |
|---|---|---|
| `asr_gender/model.ckpt` | 3-1 task-specific SFT | ~147 MB |
| `selfgen/model.ckpt` | 3-2 self-generation | ~147 MB |
| `samples/` | n/a | 4 dev-clean clips + `samples.json` (transcripts) |

```bash
hf download kehanlu/interspeech-tutorial --local-dir checkpoints
```

or from python:

```python
from huggingface_hub import snapshot_download
snapshot_download("kehanlu/interspeech-tutorial", local_dir="checkpoints")
```

Both give `checkpoints/asr_gender/model.ckpt`, `checkpoints/selfgen/model.ckpt` and
`checkpoints/samples/`. Only the adapter and LoRA weights are stored; the frozen Whisper
and Qwen3 weights are downloaded from their own repos the first time a checkpoint loads.

Everything below works the same with your own checkpoints from section 3: pass
`exp/<config>/checkpoints/<name>.ckpt` instead.

### Evaluate

One script per task, over a manifest from step 2-1 (so it needs the LibriSpeech audio
from step 1 and at least `--splits test-clean` from step 2-1). Each script takes a
`--prompt` chosen from the two written into it and puts it into every row, so one
manifest covers both prompts:

| task | `--prompt` | text |
|---|---|---|
| ASR | `default` | `Transcribe the speech into text` |
| ASR | `answer_fmt` | `Transcribe the speech word for word. Output only the transcription, with no explanation, in this format:`<br>`Answer: "<transcription>"` |
| gender | `default` | `Identify the gender of speaker` |
| gender | `mcq` | `The audio is a passage read aloud from a book. Is the speaker male or female? Answer with one word.` |

`default` is what the task-specific model was trained on; the other two ask for a
specific output format.

```bash
# task-specific SFT, with the prompts it was trained on
python example/evaluate/evaluate_asr.py --ckpt checkpoints/asr_gender/model.ckpt \
    --manifest data/Librispeech-dev-test/test-clean_asr.jsonl \
    --audio-root "$AUDIO_ROOT" --prompt default --out-dir results/asr_gender_default

python example/evaluate/evaluate_gender.py --ckpt checkpoints/asr_gender/model.ckpt \
    --manifest data/Librispeech-dev-test/test-clean_gender.jsonl \
    --audio-root "$AUDIO_ROOT" --prompt default --out-dir results/asr_gender_default

# self-generation, with the format-asking prompts
python example/evaluate/evaluate_asr.py --ckpt checkpoints/selfgen/model.ckpt \
    --manifest data/Librispeech-dev-test/test-clean_asr.jsonl \
    --audio-root "$AUDIO_ROOT" --prompt answer_fmt --out-dir results/selfgen_answer_fmt

python example/evaluate/evaluate_gender.py --ckpt checkpoints/selfgen/model.ckpt \
    --manifest data/Librispeech-dev-test/test-clean_gender.jsonl \
    --audio-root "$AUDIO_ROOT" --prompt mcq --out-dir results/selfgen_mcq
```

`evaluate_asr.py` reports WER twice: as generated, and after a rule-based clean-up
that strips the commentary a self-generation model tends to wrap its transcript in.
Each run writes per-utterance outputs and a summary to `--out-dir`; add `--limit 50`
for a quick check.

### Results

test-clean, 2,620 utterances. ASR is WER, lower is better; gender is accuracy.

| model | `--prompt` | ASR as generated | ASR after clean-up | gender |
|---|---|---|---|---|
| whisper-large-v3 (ASR reference) | n/a | 1.89 | n/a | n/a |
| **3-1** task-specific SFT | `default` | **1.81** | 1.81 | **98.85** |
| **3-2** self-generation | `default` | 52.35 | 16.54 | 81.87 |
| **3-2** self-generation | `answer_fmt` / `mcq` | 8.94 | **3.93** | **98.24** |

The task-specific model matches Whisper on ASR (1.81 vs 1.89) while also doing a
second task, with 36.7M trainable parameters and a frozen encoder. The clean-up is a
no-op for it: all 2,620 hypotheses pass through untouched, because it already answers
with a bare transcript.

The self-generation model is the interesting row. It was **never trained on ASR or
gender**. Given `default`, the prompt the other model was trained on, it answers with
an essay about the passage rather than a transcript, which is why it scores 52 WER;
for gender it writes a sentence, and 453 of 2,620 answers contain neither "male" nor
"female" at all. The clean-up pulls a transcript out of some of that (262 rows
recovered from a quoted span, 89 emptied because there was no transcript to find), but
no set of rules rescues a wrong output format.

Asking for the format is what works, and it costs nothing but a longer prompt.
`answer_fmt` gives the clean-up something to key on: 2,550 of 2,620 outputs put the
transcript after `Answer:`, and another 56 wrap it in the `<audio>…</audio>` tag the
model saw during self-generation. Both are recoverable; free-form commentary is not.
For gender, `mcq` takes 81.87 → **98.24**, with zero unparseable answers.

So a model trained only on self-generated conversational replies still carries the
speech understanding needed for both tasks. It just has to be asked in a way that
leaves room for a short answer.

### Per-file inference in python

For a notebook, or your own audio: load a checkpoint once with
`SpeechLLMForInference` from [inference.py](inference.py), then ask about any file.
Run from the repo root (or put it on `sys.path`); only the `checkpoints/` download is
needed, not LibriSpeech.

```python
import json
import torch
from inference import SpeechLLMForInference

pipe = SpeechLLMForInference.from_checkpoint("checkpoints/selfgen/model.ckpt")  # or asr_gender
# on a GPU without bfloat16 (e.g. a Colab T4): from_checkpoint(..., dtype=torch.float16)

def ask(prompt, audio, max_new_tokens=200):
    # <|AUDIO|> is where the speech features go; the rest is an ordinary user turn
    conversation = [{"role": "user",
                     "content": f"<audio><|AUDIO|></audio>\n\n{prompt}",
                     "audios": [{"audio": audio}]}]
    return pipe.generate(conversation, max_new_tokens=max_new_tokens)[0]

audio = "checkpoints/samples/1272-128104-0000.flac"
print(json.load(open("checkpoints/samples/samples.json"))[0]["transcription"])  # reference
```

The two task prompts:

```python
ask("Transcribe the speech into text", audio)
ask("The audio is a passage read aloud from a book. Is the speaker male or female? "
    "Answer with one word.", audio, max_new_tokens=8)
```

And anything else. The model is an LLM, so the question is free-form:

```python
ask("What is the speaker talking about? Answer in one sentence.", audio)
ask("The audio is a passage read aloud from a book. Respond directly as a natural "
    "conversation partner. Do not mention the audio, the transcription, or the "
    "speaker attributes.", audio)   # the self-generation prompt
```

Try the same calls with `asr_gender`: it answers the two task prompts with a bare
transcript and a bare "male", while `selfgen` tends to answer the plain
`Transcribe the speech into text` conversationally, which is the pattern the results
above measure. To batch several files, pass a list of conversations to `pipe.generate`.


## Citation

If you find this tutorial useful, please cite the papers it is based on:

```
@ARTICLE{lu2026desta25audio,
author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Huang, Sung-Feng and Yang, Chih-Kai and Yu, Chee-En and Chen, Chun-Wei and Chen, Wei-Chih and Huang, Chien-yu and Lin, Yi-Cheng and Lin, Yu-Xiang and Fu, Chi-An and Kuan, Chun-Yi and Ren, Wenze and Chen, Xuanjun and Huang, Wei-Ping and Hu, En-Pei and Lin, Tzu-Quan and Wu, Yuan-Kuei and Huang, Kuan-Po and Huang, Hsiao-Ying and Chou, Huang-Cheng and Chang, Kai-Wei and Chiang, Cheng-Han and Ginsburg, Boris and Wang, Yu-Chiang Frank and Lee, Hung-yi},
journal={IEEE Transactions on Audio, Speech and Language Processing}, 
title={DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment}, 
year={2026},
volume={},
number={},
pages={1-16},
keywords={Training;Adaptation models;Metadata;Training data;Speech processing;Music;Keyboards;Buildings;Pipelines;Benchmark testing;Cross-modal alignment;dataset construction;instruction-tuning;large audio language model},
doi={10.1109/TASLPRO.2026.3675792}}
```

```
@inproceedings{lu2025speechifeval,
  title={{Speech-IFEval}: Evaluating instruction-following and quantifying catastrophic forgetting in speech-aware language models},
  author={Lu, Ke-Han and Kuan, Chun-Yi and Lee, Hung-yi},
  booktitle={Interspeech 2025},
  year={2025}
}
```

```
@INPROCEEDINGS{Lu2025Developing,
  author={Lu, Ke-Han and Chen, Zhehuai and Fu, Szu-Wei and Yang, Chao-Han Huck and Balam, Jagadeesh and Ginsburg, Boris and Wang, Yu-Chiang Frank and Lee, Hung-Yi},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Developing Instruction-Following Speech Language Model Without Speech Instruction-Tuning Data}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889444}
}
```