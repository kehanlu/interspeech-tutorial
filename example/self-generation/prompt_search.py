#!/usr/bin/env python
"""
Compare self-generation prompts on a LibriSpeech sample: which instruction gives
task-independent (diverse) responses without confusion / meta-comments?

Every variant sees the same random utterances. The instruction text is used verbatim
on both sides, so whatever wins can go straight into generate_vllm.py --prompt:

    generation:  <audio>{transcription} (Gender: {gender})</audio>\\n\\n{instruction}
    training:    <audio><|AUDIO|></audio>\\n\\n{instruction}

    python example/self-generation/prompt_search.py --audio-root /path/to/data --n 500

Metrics per variant (results/self-generation-prompt-search/<tag>/summary.md):
    fail%        judge label confused/refusal or meta-comment (talks about the text, tag, audio, gender)
    meta_regex%  response mentions audio/transcript/tag/gender/sentence/text/recording (cheap cross-check)
    intent_H     entropy (bits) of the judge's response-type labels over non-failed responses; max log2(6)=2.58
    top_intent%  share of the most common response type among non-failed responses
    open_H       entropy (bits) of the first 3 words; top_open% = share of the most common opener
    distinct2    unique / total word bigrams across all responses of the variant
    echo%        responses where >=50% of word 4-grams come from the transcription (a re-punctuated transcript)
    refusal%     safety-style refusals ("can't assist", "let's talk about something …")
    copy%        share of response word 4-grams that also appear in the transcription
    words        mean response length

Token entropy from vLLM logprobs (bits, raw model distribution at T=1, not the sampling T/top-p):
    tokH_1 / tokH_8 / tokH_all   mean per-token entropy at the first token / first 8 tokens / all tokens.
                                 Top-k logprobs plus the remaining mass as one bucket -> a lower bound
    surprisal    mean -log2 p of the sampled tokens (all tokens)
    tail         mean probability mass outside the top-k (how loose the lower bound is)
"""

import argparse
import collections
import json
import math
import os
import random
import re
import sys

from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vllm import METADATA_TEMPLATE, MODEL_ID, TRAIN_SUBSETS, load_librispeech  # noqa: E402

AUDIOBOOK = "The audio is a passage read aloud from a book."
NO_META = "Do not mention the audio, the transcription, or the speaker attributes."

VARIANTS = {
    "desta_orig": "Response as you are a conversation partner",
    "respond_natural": "Respond directly as you are a natural conversation partner",
    "what_hear": "What can you hear from the audio?",
    "empty": "",
    "any_way": "Listen to the audio and respond to it in whatever way you find most natural.",
    "friend_moves": ("Reply to the speaker the way a thoughtful friend would. You might react, ask a question, "
                     "share a related thought, disagree, joke, or build on what they said."),
    "tag_explained": ("The audio is wrapped inside <audio> </audio>: it is what the speaker said, followed by "
                      "speaker attributes. Respond directly as a natural conversation partner. " + NO_META),
    "audiobook_respond": f"{AUDIOBOOK} Respond directly as a natural conversation partner. {NO_META}",
    "audiobook_respond_nometa_off": f"{AUDIOBOOK} Respond directly as a natural conversation partner.",
    "imagine_listen": ("Imagine you are listening to someone read a passage aloud from a book. "
                       "Respond to them directly as a natural conversation partner."),
    "imagine_listen_nometa": ("Imagine you are listening to someone read a passage aloud from a book. "
                              f"Respond to them directly as a natural conversation partner. {NO_META}"),
    "imagine_story": ("Imagine a friend just read this part of a book aloud to you. Talk with them about what "
                      "happens in it, as a natural conversation partner."),
    "audiobook_open": (f"{AUDIOBOOK} Respond to it in any way you like: react, discuss, question, imagine what "
                       f"happens next, or relate it to something else. {NO_META}"),
    "overheard": f"You just heard someone say this. Say whatever you would naturally say back. {NO_META}",
    "continue_plain": "Continue the passage.",
    "audiobook_continue": f"{AUDIOBOOK} Continue the story from where it stops. {NO_META}",
    "no_opener": ("Respond directly as a natural conversation partner. " + NO_META + " Do not start with "
                  "generic openers such as \"That sounds like\", \"Oh\", \"Hmm\", or \"I'm not sure\"."),
}

INTENTS = {
    "A": ("confused", "says it does not understand, asks what is meant, or refuses"),
    "B": ("meta", "comments on the input itself: the text, sentence, transcription, tag, audio, or gender label"),
    "C": ("react", "reacts to or gives an opinion / feeling about the content"),
    "D": ("question", "mainly asks the speaker a follow-up question about the content"),
    "E": ("interpret", "summarizes, paraphrases, or explains what the passage means or its context"),
    "F": ("continue", "continues the story, imagines what happens next, or writes creatively"),
    "G": ("inform", "gives facts, advice, or related knowledge"),
    "H": ("other", "anything else"),
}
FAIL_INTENTS = {"confused", "meta"}
REFUSAL_RE = re.compile(r"(can't assist|cannot assist|not going to engage|can't help with|won't engage|"
                        r"let's talk about something)", re.I)
META_RE = re.compile(r"\b(audio|transcri\w*|tag|gender|sentence|text|recording|passage|snippet)\b", re.I)

JUDGE_TEMPLATE = """Classify the RESPONSE a listener gave after hearing a speaker.

SPEAKER SAID: {transcription}
RESPONSE: {response}

Categories:
{categories}

Pick the single category that best describes the main thing the RESPONSE does. Answer with one letter only."""


def entropy(counter):
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values() if c) if total else 0.0


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def ngrams(tokens, n):
    return set(zip(*(tokens[k:] for k in range(n))))


def token_entropy(token_ids, logprobs):
    """Per generated token: (entropy lower bound, surprisal of the sampled token, mass outside top-k), in bits."""
    stats = []
    for tok, step in zip(token_ids, logprobs):
        probs = [math.exp(lp.logprob) for lp in step.values()]
        tail = max(0.0, 1.0 - sum(probs))
        h = -sum(p * math.log2(p) for p in probs + [tail] if p > 0)
        stats.append((h, -step[tok].logprob / math.log(2), tail))
    return stats


def aggregate_token_stats(per_response):
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0
    flat = [t for r in per_response for t in r]
    return {
        "tokH_1": mean([r[0][0] for r in per_response if r]),
        "tokH_8": mean([t[0] for r in per_response for t in r[:8]]),
        "tokH_all": mean([t[0] for t in flat]),
        "surprisal": mean([t[1] for t in flat]),
        "tail": mean([t[2] for t in flat]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", required=True, help="the directory holding the unpacked LibriSpeech/")
    ap.add_argument("--subsets", nargs="+", default=TRAIN_SUBSETS)
    ap.add_argument("--out-dir", default=os.path.join("results/self-generation-prompt-search", os.environ.get("SLURM_JOB_ID", "local")))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logprobs", type=int, default=20, help="top-k logprobs per generated token")
    args = ap.parse_args()

    samples = load_librispeech(args.audio_root, args.subsets)
    samples = random.Random(args.seed).sample(samples, args.n)
    for s in samples:
        s["metadata"] = METADATA_TEMPLATE.format(transcription=s["transcription"], gender=s["gender"])

    llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.95,
              max_num_seqs=2048, max_num_batched_tokens=32768, seed=args.seed,
              max_logprobs=args.logprobs, logprobs_mode="raw_logprobs")
    gen_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                logprobs=args.logprobs,
                                seed=args.seed)
    judge_params = SamplingParams(temperature=0.0, max_tokens=1)

    # generate all variants in one call
    keys, conversations = [], []
    for name in args.variants:
        for i, s in enumerate(samples):
            content = f"<audio>{s['metadata']}</audio>"
            if VARIANTS[name]:
                content += f"\n\n{VARIANTS[name]}"
            keys.append((name, i))
            conversations.append([{"role": "user", "content": content}])
    responses, token_stats = {}, {}
    for k, o in zip(keys, llm.chat(conversations, gen_params)):
        completion = o.outputs[0]
        responses[k] = completion.text.strip()
        token_stats[k] = token_entropy(completion.token_ids, completion.logprobs)

    # judge response type
    categories = "\n".join(f"{letter}. {desc}" for letter, (_, desc) in INTENTS.items())
    judge_convs = [[{"role": "user", "content": JUDGE_TEMPLATE.format(
        transcription=samples[i]["transcription"], response=responses[(name, i)], categories=categories)}]
        for name, i in keys]
    intents = {}
    for k, o in zip(keys, llm.chat(judge_convs, judge_params)):
        letter = o.outputs[0].text.strip()[:1].upper()
        intents[k] = INTENTS.get(letter, ("other", ""))[0]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for name in args.variants:
        texts = [responses[(name, i)] for i in range(args.n)]
        labels = [intents[(name, i)] for i in range(args.n)]
        ok_labels = collections.Counter(l for l in labels if l not in FAIL_INTENTS)
        openers = collections.Counter(" ".join(words(t)[:3]) for t in texts)
        bigrams = [b for t in texts for b in zip(words(t), words(t)[1:])]
        rows.append({
            "variant": name,
            "fail%": 100 * sum(l in FAIL_INTENTS for l in labels) / args.n,
            "meta_regex%": 100 * sum(bool(META_RE.search(t)) for t in texts) / args.n,
            "intent_H": entropy(ok_labels),
            "top_intent%": 100 * ok_labels.most_common(1)[0][1] / sum(ok_labels.values()) if ok_labels else 0.0,
            "open_H": entropy(openers),
            "top_open%": 100 * openers.most_common(1)[0][1] / args.n,
            "distinct2": len(set(bigrams)) / max(len(bigrams), 1),
            "copy%": 100 * sum(len(ngrams(words(t), 4) & ngrams(words(samples[i]["transcription"]), 4))
                               for i, t in enumerate(texts)) / max(sum(len(ngrams(words(t), 4)) for t in texts), 1),
            "echo%": 100 * sum(len(ngrams(words(t), 4) & ngrams(words(samples[i]["transcription"]), 4))
                               >= 0.5 * max(len(ngrams(words(t), 4)), 1) for i, t in enumerate(texts)) / args.n,
            "refusal%": 100 * sum(bool(REFUSAL_RE.search(t)) for t in texts) / args.n,
            "words": sum(len(words(t)) for t in texts) / args.n,
            **aggregate_token_stats([token_stats[(name, i)] for i in range(args.n)]),
            "intents": dict(collections.Counter(labels).most_common()),
            "top_openers": openers.most_common(5),
            "instruction": VARIANTS[name],
        })
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            for i, s in enumerate(samples):
                f.write(json.dumps({"metadata": s["metadata"], "response": texts[i], "intent": labels[i]},
                                   ensure_ascii=False) + "\n")

    cols = ["variant", "fail%", "meta_regex%", "intent_H", "top_intent%", "open_H", "top_open%", "distinct2", "copy%", "echo%", "refusal%", "words",
            "tokH_1", "tokH_8", "tokH_all", "surprisal", "tail"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: r["fail%"]):
        lines.append("| " + " | ".join(r[c] if isinstance(r[c], str) else f"{r[c]:.2f}" for c in cols) + " |")
    lines.append("")
    for r in rows:
        lines += [f"### {r['variant']}", f"- instruction: `{r['instruction']}`", f"- intents: {r['intents']}",
                  f"- top openers: {r['top_openers']}", ""]
    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(report)


if __name__ == "__main__":
    main()
