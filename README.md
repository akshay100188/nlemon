# nLemon-14

> **Born to speak, disciplined to behave.**

A ~14M-parameter GPT taken through the **entire modern LLM lifecycle** on a 4GB laptop
GPU: **pretrain → SFT → DPO → eval → publish**. The model is born (learns English from
scratch on TinyStories), then tamed (learns to follow instructions, then to stay
on-topic under preference tuning).

The deliverable is not a service. It is a **checkpoint, a scorecard and a loss curve** —
all regenerable from one seed.

Release stages: `nLemon-14-base` → `-sft` → `-dpo`.

---

## Status

| Phase | Stage | State |
|---|---|---|
| 1 | Scaffold + data | ✅ done |
| 2 | Tokenizer (8k BPE) | ✅ done |
| 3 | Architecture | ✅ done |
| 4 | Pretraining | ✅ done |
| 5 | SFT | ⬜ not started |
| 6 | DPO | ⬜ not started |
| 7 | Eval harness | ⬜ not started |
| 8 | Public artifact | ⬜ not started |

No phase starts until the previous phase's gate is green.

---

## The model

Decoder-only GPT (nanoGPT lineage), pre-norm blocks, tied input/output embeddings.

| | |
|---|---|
| vocab | 8,000 (our own BPE) |
| d_model | 384 |
| layers | 6 |
| heads | 6 (head_dim 64) |
| context | 256 |
| dropout | 0.1 |
| precision | bf16 |
| **params** | **13,817,856** (98.7% of the 14M budget) |

| where the parameters live | | |
|---|---|---|
| token embedding | 3,072,000 | 22.2% |
| positional embedding | 98,304 | 0.7% |
| 6 blocks × 1,774,464 | 10,646,784 | 77.1% |
| final layernorm | 768 | 0.0% |

The output head is tied to the token embedding, which saves a further 3,072,000
parameters — 22% of the model — and ties "predicting a token" to "representing a token".

Model shape is never changed to fit VRAM. Memory pressure is absorbed only by
`micro_batch` / `grad_accum_steps` — see [ADR-004](ADR.md#adr-004--config--seed-as-the-single-source-of-truth).

---

## Reproduce

Requires Python 3.12 and, for the training stages, an NVIDIA GPU with a driver new
enough for CUDA 13.0 (≥ 580). CPU-only machines can run Phases 1–2; see
[`requirements.txt`](requirements.txt).

```bash
git clone <repo> nlemon && cd nlemon

python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python -m utils.device            # confirm CUDA / bf16 / VRAM
python config.py                  # print the resolved config + its hashes

python -m src.data                # Phase 1 — build the corpus      (~17 min)
python -m src.tokenizer train     # Phase 2 — 8k byte-level BPE
python -m src.tokenizer encode    # Phase 2 — corpus -> .bin shards
python -m src.tokenizer check     # Phase 2 — the gate

python -m src.model summary       # Phase 3 — parameter accounting
python -m src.model check         # Phase 3 — the gate

python -m src.baselines           # Phase 4 — perplexity floors
python -m src.train               # Phase 4 — pretrain to base.pt   (~2 h)
python -m src.sample --gallery    # Phase 4 — reproducible samples
python -m src.coherence reference # Phase 4 — corpus bands
python -m src.coherence gate      # Phase 4 — the gate
```

Every hyperparameter and the global seed live in
[`configs/nlemon_14m.yaml`](configs/nlemon_14m.yaml). Nothing is hardcoded elsewhere,
and every artifact records the hash of the config that produced it.

**Determinism is enforced, not assumed**
([ADR-016](ADR.md#adr-016--seeding-is-not-reproducibility-strict-determinism-is)).
Seeding alone turned out not to be enough: the same gate run three times gave final losses
of 0.01515 / 0.01034 / 0.01264. The seed fixes initialisation — the first-step loss was
identical every time — but the backward pass accumulates with atomics whose order varies.
`strict_determinism: true` makes training bit-reproducible for about 7% throughput. Without
it, every number in the eventual scorecard would sit downstream of a checkpoint nobody
could rebuild.

**Two kinds of hash** ([ADR-010](ADR.md#adr-010--per-stage-config-hashes-alongside-the-global-one)).
`config_hash` fingerprints the whole config and identifies a run. A `stage_hash`
covers only the fields a given stage actually depends on, so adding a Phase 4
hyperparameter does not make your Phase 1 corpus look stale. This is verified rather
than asserted: adding the two tokenizer fields moved `config_hash` from `be96725bd672`
to `53f4919fceb7` while `data_stage_hash` and both shard SHA-256s stayed put.

---

## Data

**Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(`roneneldan/TinyStories`) — synthetic short stories written with a small vocabulary
so that *tiny* models produce fluent, coherent English. That is what makes the SFT and
DPO effects visible at 14M parameters ([ADR-001](ADR.md#adr-001--corpus--tinystories)).

**License:** CDLA-Sharing-1.0 (as declared on the Hub). The exact license string, the
resolved dataset commit sha, per-split document counts and a SHA-256 of each shard are
recorded by the data script in `data/manifest.json` — verify against that file, not
against this paragraph.

The corpus is synthetic: no PII, no scraping, no real people or organisations.

**What the build produces** (train / validation):

| | train | validation |
|---|---|---|
| documents | 2,119,489 | 21,990 |
| words | 364,096,658 | 3,680,512 |
| size | 1,841 MiB | 18.6 MiB |
| words per doc (p50) | 150 | 149 |

**Cleaning.** The upstream text ships with 757,666 double-encoded smart quotes
(`â€™` for `’`). We repair them, because at an 8,000-token vocabulary every merge
spent on mojibake is a merge not spent on English. `ftfy` alone was not enough — it
*created* 140,572 spurious `†` characters out of quotes whose bytes were already
lost, which is worse, so an explicit residue map runs after it. The measured result is
**9 stray characters left in train and none in validation** — and all 9 are identified
and explained, not merely counted: one is legitimate French (`papier-mâché`), two are a
mojibake'd emoji, and six are a single non-English document out of 2.1M. Every build
prints the count and records a context snippet for each in `data/manifest.json`. The full
reasoning and the adjudication table are in
[ADR-009](ADR.md#adr-009--repair-upstream-mojibake-before-tokenizing).

`data/` is gitignored and fully regenerable with `python -m src.data`. Train and
validation come from the dataset's own splits, never a homemade one
([ADR-006](ADR.md#adr-006--use-the-datasets-own-trainvalidation-boundary)); 230 empty
documents are dropped and counted
([ADR-008](ADR.md#adr-008--drop-empty-documents-keep-short-ones)).

A deterministic peek at three training stories lives in
[`results/samples/corpus_peek.md`](results/samples/corpus_peek.md).

---

## Tokenizer

An 8,000-token **byte-level** BPE, trained on our own corpus — no borrowed vocabulary
([ADR-011](ADR.md#adr-011--byte-level-bpe)). Byte-level means the initial alphabet is all
256 bytes, so there is no OOV, no unknown token, and the encode→decode roundtrip is
lossless *by construction* rather than by luck. That matters here because the corpus still
contains 9 characters of unrepairable non-ASCII and one Traditional Chinese document
(ADR-009); a character-level vocabulary with an `<unk>` token would fail the gate on
exactly those inputs, or force the gate to be weakened.

Measured on held-out validation text: **1.2243 tokens per word, 4.18 characters per
token.**

**How much text does an 8k vocabulary need?** Measured, not guessed —
[results/tokenizer_subset_sweep.md](results/tokenizer_subset_sweep.md). Compression
plateaus at 25,000 documents: across a 32× increase in training text the spread is
0.0007 tokens/word. `tokenizer_train_docs` is nonetheless set to **200,000**, because
compression is not the only thing at stake — two tokenizers can compress identically while
disagreeing on which rare tokens earned a slot. At 200k the vocabulary is 97.4% identical
to one trained on 800k documents; at 25k it is only 92.6%.

## Architecture gate

Phase 3's gate is "param count matches the budget, and the model can overfit one batch."
Both are implemented, plus two checks the specified gate would have missed.

- **The parameter count is derived twice** — summed from the modules, and computed
  analytically from the config — and the gate fails unless they agree exactly
  ([ADR-013](ADR.md#adr-013--check-the-parameter-count-twice-from-independent-derivations)).
  Counting the model's own tensors only proves it is self-consistent: an accidentally
  untied head would still produce "the correct count for this model", and still land inside
  a ±10% budget.
- **The causal mask is tested separately**, because the overfit cannot test it
  ([ADR-014](ADR.md#adr-014--the-single-batch-overfit-does-not-test-the-causal-mask)).
  A model with no mask memorises a batch *faster* — it can read the answer off the next
  token. The test asserts the real property: poking the token at position *t* moves the
  logits at every position before it by exactly `0.0`, and moves them from *t* onward.
- **The gate threshold is chosen by its worst seed**, not a lucky run
  ([ADR-015](ADR.md#adr-015--gate-thresholds-are-chosen-by-their-worst-seed),
  [results/overfit_margin.md](results/overfit_margin.md)). At 300 steps the spread across
  seeds is wider than the distance to the target, so the same correct model passes or fails
  on initialisation alone.

## nLemon-14-base — it learned to speak

327,680,000 tokens (0.70 epochs), 20,000 steps, 120 minutes on the 3050 A.

| | |
|---|---|
| **val perplexity** | **5.1981** (loss 1.6483) |
| gate threshold | ≤ 8.0, agreed before the run |
| bigram floor | 41.81 → **6.9× better** |
| unigram floor | 389.91 → 64× better |
| uniform floor | 8,000 |

The threshold is not a number I liked the look of. It is 5.2× the strongest trivial
predictor measured on the same held-out shard with the same tokenizer
([results/perplexity_floors.md](results/perplexity_floors.md),
[ADR-017](ADR.md#adr-017--the-perplexity-threshold-is-anchored-to-measured-floors-and-agreed-first)) —
and it was fixed while the run was at 10% and its outcome unknown. Perplexity across
tokenizers is not comparable, so a figure borrowed from a 50k-vocab paper would have meant
nothing here.

**Curve:** [results/loss_curve.csv](results/loss_curve.csv) — train loss every 20 steps,
validation every 500. Val loss fell 9.08 → 1.65 with no divergence between train and val,
so nothing was memorised at 0.70 epochs.

**Samples:** [results/samples/base_samples.md](results/samples/base_samples.md) — four
fixed prompts, seeded, reproducible. They are recognisably little stories with dialogue and
a narrative arc. The logic wobbles exactly where 14M parameters would predict: characters
revisit places they just left, and objects change owners between paragraphs.

**The coherence half of the gate is measured, not asserted.** Five statistics computed on
real validation documents give a p5–p95 band, and generated text has to land inside it
([ADR-018](ADR.md#adr-018--coherence-bands-come-from-the-corpus-not-from-taste)):

| metric | generated (median) | corpus band |
|---|---|---|
| repeated 4-gram rate | 0.0155 | 0.0000 – 0.0333 |
| max immediate repeat run | 1.0 | 1.0 – 2.0 |
| mean sentence words | 9.63 | 7.07 – 14.27 |
| type/token ratio | 0.5274 | 0.4380 – 0.6275 |
| known-word rate | 1.0000 | 1.0000 – 1.0000 |

Both edges of each band are checked on purpose: too much repetition is degenerate, and too
*little* is also suspicious, because real children's stories repeat names deliberately. One
of these metrics has a degenerate band and says so in the ADR rather than pretending
otherwise.

## Repo layout

```
nlemon/
├── config.py                 # frozen dataclass, single source of truth
├── configs/nlemon_14m.yaml   # every hyperparameter + the seed
├── src/
│   ├── data.py               # corpus build           (Phase 1) ✅
│   ├── tokenizer.py          # 8k byte-level BPE      (Phase 2) ✅
│   ├── model.py              # GPT from scratch       (Phase 3) ✅
│   ├── train.py              # pretraining            (Phase 4) ✅
│   ├── baselines.py          # perplexity floors      (Phase 4) ✅
│   ├── coherence.py          # deterministic proxy    (Phase 4) ✅
│   ├── sample.py             # inference (temp/top-k) (Phase 4) ✅
│   ├── sft.py                # instruction tuning     (Phase 5)
│   ├── dpo.py                # preference tuning      (Phase 6)
│   └── eval.py               # deterministic scorecard(Phase 7)
├── utils/  seed.py · device.py
├── data/         # .txt / .bin shards + manifest.json   (gitignored)
├── checkpoints/  # base.pt · sft.pt · dpo.pt            (gitignored)
├── results/      # loss_curve.csv · scorecard.json · samples/
└── ADR.md        # decision log — every decision with its rejected alternative
```

---

## How this is judged

No stage is "done" on vibes. Each phase has a deterministic, re-runnable gate:
lossless tokenizer roundtrip, a single-batch overfit that proves the wiring, a
validation-perplexity threshold, a measured instruction-adherence delta, a measured DPO
win-rate. The alignment judge is a rule-based feature checker, not another model —
reproducibility is the point ([ADR-005](ADR.md#adr-005--deterministic-feature-checker-as-the-alignment-judge)).

## License

Code: MIT (see `LICENSE`). Data: see **Data** above — TinyStories carries its own
license and is not redistributed here.
