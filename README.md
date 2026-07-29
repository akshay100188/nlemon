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
| 2 | Tokenizer (8k BPE) | ⬜ not started |
| 3 | Architecture | ⬜ not started |
| 4 | Pretraining | ⬜ not started |
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
| **params** | **~14M** |

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
python config.py                  # print the resolved config + its hash
python -m src.data                # build the corpus  (Phase 1)
```

Every hyperparameter and the global seed live in
[`configs/nlemon_14m.yaml`](configs/nlemon_14m.yaml). Nothing is hardcoded elsewhere,
and every artifact records the 12-character hash of the config that produced it.

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
**18 stray characters left in train and none in validation**; the count is printed on
every build. The full reasoning is
[ADR-009](ADR.md#adr-009--repair-upstream-mojibake-before-tokenizing).

`data/` is gitignored and fully regenerable with `python -m src.data`. Train and
validation come from the dataset's own splits, never a homemade one
([ADR-006](ADR.md#adr-006--use-the-datasets-own-trainvalidation-boundary)); 230 empty
documents are dropped and counted
([ADR-008](ADR.md#adr-008--drop-empty-documents-keep-short-ones)).

A deterministic peek at three training stories lives in
[`results/samples/corpus_peek.md`](results/samples/corpus_peek.md).

---

## Repo layout

```
nlemon/
├── config.py                 # frozen dataclass, single source of truth
├── configs/nlemon_14m.yaml   # every hyperparameter + the seed
├── src/
│   ├── data.py               # corpus build           (Phase 1)
│   ├── tokenizer.py          # 8k BPE                 (Phase 2)
│   ├── model.py              # GPT from scratch       (Phase 3)
│   ├── train.py              # pretraining            (Phase 4)
│   ├── sft.py                # instruction tuning     (Phase 5)
│   ├── dpo.py                # preference tuning      (Phase 6)
│   ├── sample.py             # inference (temp/top-k) (Phase 7)
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
