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
| 5 | SFT | 🟥 attempt #1 **RED** → 🟩 attempt #2 **green** ([why](results/sft_gate.md)) |
| 6 | DPO | 🟥 **RED** — target moved +3.5pt against a +12.5pt bar, below the 8.9pt the eval can resolve; no floor breached ([why](#phase-6-dpo-red)) |
| 7 | Eval harness | ⬜ not started — proceeds on `sft.pt`; Phase 6 is RED-characterized, foundation intact |
| 8 | Public artifact | ⬜ not started |

No phase starts until the previous phase's gate resolves. A gate is **green** (proceed),
**RED-blocking** (the question is unanswered or the instrument is broken — stop and fix), or
**RED-characterized** (the question is answered, the answer is negative or null, the mechanism is
understood, and the next phase's foundation is intact — proceed, and ship the RED as the result).
Phase 5 attempt #1 was RED-blocking. Phase 6 is RED-characterized. A RED-characterized gate still
ships as a RED and never licenses re-registering the failed phase to chase a green
([ADR-045](ADR.md)).

**Phase 5 failed first, and that result stands.** Attempt #1 came back RED —
`subject_mention` reached 50.0% against a pre-registered 60.0%. The diagnosis was that
47.8% of the training pairs mentioned their subject at most once, because the miner
required the subject to *appear* in a document and appearing is not being about it
([ADR-030](ADR.md)). Attempt #2 re-derived the pool by corpus aboutness and re-registered
every bar from a freshly measured base floor ([ADR-032](ADR.md)). It is a second attempt
**beside** the first, not a replacement for it, and both are reported:

| | attempt #1 | attempt #2 |
|---|---|---|
| `subject_mention` | 35.0% → **50.0%** vs bar 60.0% — **RED** | 47.8% → **70.8%** vs bar 65.3% — green* |
| `length_band` | 43.5% → 95.5% vs bar 69.0% — pass | 37.8% → **92.9%** vs bar 67.8% — pass |
| `is_story` (floor) | 77.0% → 98.5% | 76.0% → **98.1%** |
| `not_degenerate` (floor) | 73.5% → 85.5% | 70.8% → **90.1%** |
| shuffled control | 7.0% → 8.5% (≤10%) | 0.6% → **1.9%** (≤6%) |
| pairs teaching subject ≤1× | 47.8% | **0.2%** |
| effective n | ~106 of 200 prompts | **~250** of 312 |

\* **A grazing green.** `subject_mention` cleared its bar by 5.6 points against an
8.8-point detection floor (one-sided p = 0.041; a cluster bootstrap puts 3.5% of resamples
below the bar). The *delta* of +23.1 points is decisive at 2.6× the floor — the fine-tune
demonstrably taught subject adherence — but the *margin over the bar* is not, and a rerun on
another seed could land in amber. The amber band had been registered below the bar and not
above it; [ADR-035](ADR.md) makes it two-sided.

### Phase 6, DPO: RED

**DPO did not move its target by an amount this eval can resolve.** The bar was registered
before the pairs were built, in the fixed read order *side-condition → floors → delta*, with
the headline read last ([ADR-039](ADR.md)):

| step | | sft | dpo | threshold | |
|---|---|---|---|---|---|
| 1 | side-condition `subject_mention` | 70.8% | **74.4%** | ≥ 65.3% | **holds** — Phase 5 still certified |
| 2 | floor `subject_mention` | 70.8% | 74.4% | ≥ 67.0% | green |
| 2 | floor `length_band` | 92.9% | 92.6% | ≥ 91.2% | green |
| 2 | floor `is_story` | 98.1% | 98.7% | ≥ 97.2% | green |
| 2 | floor `not_degenerate` | 90.1% | 90.1% | ≥ 88.0% | green |
| 3 | **delta** `subject_mention` | 70.8% | **74.4%** | **≥ 83.3%** | **RED** |

**Nothing was traded.** Every floor held and Phase 5's certification survives — the registered
worry that DPO would spend Phase 5's 5.5-point grazing margin buying reward did not happen. It
spent nothing because it did little.

**The verdict turned on one prompt, and that is recorded rather than used.** `subject_mention`
is 232/312; the AMBER− threshold is 0.744665, so 232 is RED and 233 would have been AMBER−. The
miss is 0.11pt where one prompt is worth 0.32pt — a third of a single observation — and the
threshold falls in a gap between attainable scores. **The verdict does not move**: the rule was
registered before the pairs existed and 232/312 is RED under it. But the label is also not where
the information is. At 233/312 the delta would be +3.8pt instead of +3.5pt, against a +12.5pt
bar and an **8.9pt detection floor**. Both readings say the same thing, so the honest headline
is not "RED by a hair" — it is *+3.5pt against a bar of +12.5, below what the eval resolves.*

**The RED is informative, not an execution failure.** [ADR-039](ADR.md) argued DPO cannot *teach*
subject-adherence because `sft.pt` already has it — DPO can only re-rank samples already in the
distribution — and anchored the bar at 44% of remaining headroom while flagging that anchor at
the time as *"the optimistic edge rather than a neutral estimate."* That caveat is now a
measurement: **re-ranking bought about a quarter of what teaching bought on the same metric.**

**The sharpest number is the transfer gap.** Ranking accuracy on the 714 training pairs over 293
subjects reached **~81%** from a 50% start — the preference was learned — while held-out
`subject_mention` moved +3.5pt. *The preference was learned and did not generalise* to subjects
the pairs never named. That is a more useful finding than the verdict.

The claim is stated at the width it was measured at ([ADR-044](ADR.md)). **Not** "DPO does not
generalise", but: *under the registered treatment — one epoch, 22 steps, lr 5e-7, deliberately
stopped before convergence — DPO reached ~81% pair-ranking accuracy and transferred +3.5pt to 78
unseen subjects.* The stopping point was checked rather than assumed: over the last half of
training, pair accuracy was flat (80.0% → 81.5%, t = +0.05) while margin still climbed
(t = +3.96). Accuracy is a sign statistic and margin a magnitude one, so the model had settled
*which* pairs it ranked correctly and was still increasing *how confidently* — sharpening, which
is exactly the over-optimisation the pre-committed "do not extend" was written to avoid. The
accuracy null is weakly powered (it could only resolve a gain larger than ~11 points), so the
margin result carries the reading and the accuracy result is suggestive.

An LR confound was checked and refuted: the warmup defect ([ADR-041](ADR.md)) never touched this
checkpoint — the curve's own `lr` column reads `5.000000e-07` from step 2 to 22, and the
defective run was killed at step 1 and wrote nothing.

Per the pre-committed response, the phase is **not extended**: no extra epochs, no higher LR, no
more pairs, no third attempt at a lower bar ([ADR-032](ADR.md), [ADR-043](ADR.md)).

### Parked experiments

Questions this project measured the edge of but did not answer. They are **parked, not owed** —
each is a fresh experiment with its own pre-registration, not debt against the phase that raised
it. The standing test for whether one may be started: *would it be built if the phase that
raised it had gone green?* If no, it is outcome-triggered and stays parked.

**P1 — Does DPO transfer to unseen subjects when run to accuracy-convergence?** Phase 6 stopped
deliberately at one epoch, with pair-ranking accuracy flat but margin still climbing, and
transferred +3.5pt to 78 unseen subjects. Two separable factors are confounded in that number
and neither was varied: *training length* (would running to accuracy-convergence rather than
stopping conservatively move transfer, or only sharpen margin?) and *subject coverage* (would
pairs spanning more than 293 subjects move it?). Pre-statable hypothesis: if the sign/magnitude
split in [ADR-044](ADR.md) is the right mechanism, extending training should move margin and
**not** transfer, while broadening subject coverage should move transfer. That is a genuine
two-armed experiment with an outcome that can embarrass the hypothesis, which is what makes it
worth registering rather than assuming.

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

python -m src.verify_docs                    # prose ratios == measured artifacts
python scripts/check_encoding.py --install    # encoding guard as a pre-commit hook
```

Two guards exist because two mistakes actually happened, not because they seemed prudent.

`src/verify_docs.py` checks every "Nx better" in the docs against
`results/perplexity_floors.json`, after a stale denominator shipped a wrong ratio
([ADR-020](ADR.md#adr-020--the-gate-metric-was-underpowered-and-the-published-ratio-must-be-the-conservative-one)).

`scripts/check_encoding.py` blocks commits containing mojibake or a BOM, after PowerShell's
`Get-Content | Set-Content` corrupted this repo's own source twice — in a project whose
Phase 1 headline is repairing 757,666 mojibake sequences. It also pins the *documented*
corruption by exact codepoint: running ftfy over the docs once "repaired" ADR-009's example
quotes into well-formed UTF-8 that said the wrong thing, and a byte-legality check passes
that happily. Clean-but-wrong is the corruption that survives a validity test — ADR-009's
own lesson, one level up.

Both print the size of what they examined, because
[ADR-021](ADR.md#adr-021--a-green-check-is-a-claim-and-claims-get-checked) is about checks
that pass while measuring the wrong thing. Three of those turned up in Phase 4 alone.

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

**Two perplexities appear below, and they are not interchangeable.** The **gate metric** is
5.1981 — `eval_iters: 100`, the procedure pre-registered before the run, and the number that
decided pass/fail. The **ratios** divide by 5.4636, the model scored on the *same token
array* as the baselines. If you divide 41.81 by the headline you get 8.04, not 7.65; the
ratios deliberately use the harsher denominator, which is harsher even than the converged
random-window estimate of ~5.2680.

| the gate | |
|---|---|
| **val perplexity, gate metric** | **5.1981** (loss 1.6483), `eval_iters: 100`, pre-registered |
| threshold | ≤ 8.0, fixed before any full-run result was seen |
| verdict | PASS — crossed the bar at step 3,000 |

| the baseline comparison — all ÷ 5.4636 | |
|---|---|
| val perplexity on the baselines' 1,992,060 tokens | 5.4636 (loss 1.6981) |
| bigram floor 41.81 | → **7.65x better** |
| unigram floor 389.91 | → 71.36x better |
| uniform floor 8,000 | → 1,464.24x better |

The gate metric averages 100 batches of random windows (409,600 tokens). The ratios use a
stricter measurement: the model scored on the **identical token array** as the baselines, so
numerator and denominator come from the same data
([results/perplexity_floors.md](results/perplexity_floors.md), regenerable with
`python -m src.baselines --with-model checkpoints/base.pt`).

That stricter number is *worse* (5.4636 vs 5.1981) and it is the one the ratios use, because
a headline should not rest on the friendlier of two measurements. The gap is sampling noise
plus subset: at `eval_iters: 100` the estimate moves between 5.198 and 5.280 across loader
seeds, and 5.1981 sat at the optimistic edge; the low-variance estimate over 4.1M tokens is
5.2680. Every figure clears ≤ 8.0, so the verdict never depended on the choice — but Phase
7's scorecard needs a large fixed evaluation set rather than 100 random batches, recorded
as a requirement in
[ADR-020](ADR.md#adr-020--the-gate-metric-was-underpowered-and-the-published-ratio-must-be-the-conservative-one).

**What "pre-registered" does and does not claim here.** The bar was set after the run
launched, and a 60-step pilot (perplexity 154.6) already existed — so neither "before step
0" nor "before any result" would be true. The narrower claim is the one that holds: 154.6
sits between the unigram and bigram floors, cannot distinguish a finish at 4 from one at
12, and did not inform the number. 8.0 came from the measured floors as a stated multiple
of bigram, no full-run figure was seen, and the gate stayed live until step 3,000
([ADR-017](ADR.md#adr-017--the-perplexity-threshold-is-anchored-to-measured-floors-and-agreed-first)).

The threshold itself is not a number I liked the look of. It is 5.2 times the strongest
trivial predictor, measured on the same held-out shard with the same tokenizer.
All four figures above are per-token negative log-likelihood under our own 8,000-token BPE:
perplexity is not comparable across tokenizers, and a word-level baseline against a
token-level model would silently change what the ratio means.

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

| metric | statistic | generated | corpus band |
|---|---|---|---|
| repeated 4-gram rate | median | 0.0155 | 0.0000 – 0.0333 |
| max immediate repeat run | median | 1.0 | 1.0 – 2.0 |
| mean sentence words | median | 9.63 | 7.07 – 14.27 |
| type/token ratio | median | 0.5274 | 0.4380 – 0.6275 |
| out-of-vocabulary rate | pooled, bootstrapped | 0.0000 | 0.0000 – 0.0021 |
| OOV plausibility | pooled, bootstrapped | n/a — invents nothing | -2.7517 – -1.8522 |

> **Scope of this green, narrowed after the fact.** These numbers are measured on
> *continuation* prompts — "Once upon a time" and friends, the shape a pretrained model was
> trained on. Phase 5 scored the same checkpoint on **instruction** prompts and base leaves
> the band: `oov_rate` 0.006097 against a ceiling of 0.002114, three times over, because a
> model that cannot parse "Write a story about…" emits mangled forms of the instruction verb
> (`tellow` ×24, `mrite` ×21, `telle` ×9). The row above says *invents nothing*; the honest
> version is *invents nothing when continuing text*. The coherence green holds for the
> distribution it was measured on and was silently broader than its measurement.
> See [ADR-031](ADR.md#adr-031--the-oov_plausibility-bands-first-live-firing-was-on-base-not-sft).

Both edges of each band are checked on purpose: too much repetition is degenerate, and too
*little* is also suspicious, because real children's stories repeat names deliberately. The
[decoding sweep](results/decoding_sweep.md) shows both failures happening — cold sampling
loops, hot sampling invents words — which is the argument for a two-sided band with the
evidence attached.

The last two metrics replaced a degenerate one
([ADR-022](ADR.md#adr-022--rare-events-get-pooled-and-bootstrapped-not-medianed)).
`known_word_rate` had a band of exactly [1.0, 1.0], because over 95% of real documents
contain no unknown words, so its p5 and p95 collapsed to a point. It passed, and it could
never have failed for the right reason. Out-of-vocabulary words are rare, so they are now
pooled across the whole sample set and their band bootstrapped from equal-sized groups of
real documents. And because membership cannot distinguish an invented name from garbage —
both are unknown — plausibility is scored by character trigram: invented names like
`hoppity` land at -2.23 and inside the band, garbage like `bcdfghj` at -4.27 and far
outside.

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
├── utils/  seed.py · device.py · io.py
├── scripts/check_encoding.py # encoding guard (pre-commit hook)
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
