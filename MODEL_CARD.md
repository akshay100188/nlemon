---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
tags:
  - gpt
  - from-scratch
  - tinystories
  - small-language-model
  - reproducible-research
---

# Model card — nLemon-14

**nLemon-14-sft** · a 13.8M-parameter decoder-only transformer, pretrained from scratch and
instruction-tuned, on one 4 GB laptop GPU. *"Born to speak, disciplined to behave."*

| | |
|---|---|
| **Model of record** | `nlemon-14-sft-weights.pt` — the SFT checkpoint, certified twice (Phase 5 gate green; Phase 6 floors confirm DPO left it untouched) |
| architecture | decoder-only, 6 layers, 6 heads (head_dim 64), d_model 384, context 256, dropout 0.1, tied output head |
| tokenizer | own byte-level BPE, 8,000 merges, 1.2243 tokens/word |
| training data | TinyStories, 2,119,489 train docs / 364M words |
| pretraining | 20,000 steps, 327.7M tokens (0.70 epochs), 120 min on an RTX 3050 A |
| release file | 55.3 MB, **fp32 weights only**, optimizer state excluded |
| hash | recorded in [`release_manifest.json`](release_manifest.json), verifiable with `python -m src.release verify` |

---

## Parameter count — read this before recomputing it

**The shipped `state_dict` sums to 16,889,856. The published parameter count is 13,817,856.
Both are correct.**

```
16,889,856   sum over state_dict values
-  3,072,000   tied embedding, counted twice (vocab 8,000 x d_model 384)
= 13,817,856   unique parameters
```

The output head is **tied** to the token embedding — `lm_head.weight` *is* `tok_emb.weight`, the
same tensor stored under two names. Summing `state_dict` double-counts it; `model.parameters()`
de-duplicates and gives the published figure. Tying saves 3.07M parameters, 22% of the model, and
ties "predicting a token" to "representing a token."

This is stated because **a reader who recomputes from the shipped file will get the larger number**,
and a card that offered only one figure would contradict its own artifact to exactly the audience
that checks. `python -m src.release verify` reconciles the arithmetic against the file.

## Performance

| metric | value | notes |
|---|---|---|
| **validation perplexity** | **5.2662 ± 0.27%** | `base.pt`, full validation split, 4,682,459 tokens |
| | 5.5794 | `sft.pt` — **6.0% worse than base**, and expected; see below |
| bigram floor | 41.81 → 7.65× better | scored on the baselines' identical token array |
| unigram floor | 389.91 → 71.36× better | |
| uniform floor | 8,000 → 1,464× better | |

**The ± 0.27% is the measured, design-effect-adjusted standard error, not the iid one.** The naive
iid figure is 0.0979%. Tokens cluster inside documents — measured ICC 0.0296 over 21,990 documents
averaging 213 tokens, giving a design effect of 7.28 and `n_eff` 643,480 rather than 4.68M. **The
iid number would have overstated the precision by 2.7×.** Quoting it would have been the more
impressive and less true option.

**Why the headline is 5.2662 and not the 5.1981 in earlier commits.** `best_val_perplexity` was the
*minimum* over 40 validation passes, because training saves a checkpoint only when validation
improves — an **order statistic**, biased low by construction, sitting at the 4.2 percentile of its
own estimator's sampling distribution. The full-split pass has no sampling error to be lucky in.
Every figure clears the pre-registered ≤ 8.0 bar, so no verdict ever depended on the choice.

**Why `sft.pt` scores worse than `base.pt` on perplexity, and why that is not a regression.**
Instruction tuning specialises the model onto an instruction/response format, so its fit to the
*pretraining* distribution necessarily degrades. The +6.0% is the **price of the specialisation, not
a defect** — and it is printed here because a scorecard that reported only the metrics a stage
improves would be an advertisement rather than a measurement. What SFT was *for* is instruction
adherence, and that is measured separately: `subject_mention` 47.8% → 70.8% on 78 held-out subjects.

## Instruction-following (78 held-out subjects, 312 prompts, T=0.8 k=40)

| | base | sft | dpo |
|---|---|---|---|
| `subject_mention` | 47.8% | **70.8%** | 74.4% |
| `length_band` (103–190 words) | 37.8% | **92.9%** | 92.6% |
| `is_story` | 76.0% | **98.1%** | 98.7% |
| `not_degenerate` | 70.8% | **90.1%** | 90.1% |
| shuffled-subject control | 0.6% | 1.9% | 1.6% |

Subjects in this eval never appear in any training pair. The **shuffled control** is the validity
check: it re-scores each response against a *different* subject, so a model that simply names many
nouns scores high on it. At 1.9% it does not, which is what licenses reading `subject_mention` as
adherence rather than noun-spraying.

## Licensing

- **Weights, code, and this card:** MIT.
- **Training corpus:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, under **CDLA-Sharing-1.0**.

CDLA-Sharing-1.0's share-alike obligation attaches to the **Data**. Computational **Results**
obtained from analysing the data are explicitly outside it, and model weights are Results — which
is the basis for releasing the weights under MIT. The corpus is not redistributed here; the build
pulls it from its own source, and `data/manifest.json` records the exact revision and its license
so the provenance is checkable rather than asserted. *Stated for transparency, not as legal advice.*

## Limitations

- **Tiny, and trained on one narrow corpus.** TinyStories is simple English children's stories. The
  model has no knowledge, no reasoning, and no safety training. It is a methodology artifact.
- **256-token context.** Roughly 200 words.
- **DPO did not generalise.** `dpo.pt` reached ~81% pair-ranking accuracy on its 714 training pairs
  but transferred only +3.5pt to unseen subjects — below the 8.9pt the eval can resolve. It ships as
  a *characterized side-result*, never the headline. `sft.pt` is the model of record.
- **`oov_plausibility` is dormant on this model.** 2 out-of-vocabulary words in 45,031 — the band
  cannot be read at that count, and is reported as unreadable rather than as a pass.
- Not evaluated for bias, toxicity, or factuality. Do not deploy it.

## Reproducing

```bash
python -m src.evalset verify      # eval sets match their registered hashes
python -m src.eval scorecard      # the 5-gate Phase 7 scorecard
python -m src.release verify      # shipped file matches its recorded hash
python -m tests.test_residue      # encoding-repair regression
```

Requires the repo venv (`torch 2.12.1+cu130`). The harness **asserts** its device and fails loud
rather than falling back to CPU — a silent fallback once put a device difference inside a
cross-phase comparison ([ADR-047](ADR.md)).

Every number above is regenerable from recorded inputs. The full decision record, including the
gates that failed and why, is in [ADR.md](ADR.md) — 51 entries.
