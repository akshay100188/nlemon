# ADR log — nLemon-14

One entry per non-trivial decision: **context · decision · alternatives rejected ·
consequence.** A decision without a recorded rejected alternative is not a decision,
it is a default.

---

## ADR-001 — Corpus = TinyStories

**Context.** A ~14M-parameter model has to produce *fluent* English before SFT and DPO
can visibly change anything. If the base model outputs word salad, alignment has
nothing to tame and the whole "born → tamed" arc is unobservable.

**Decision.** Pretrain on TinyStories (`roneneldan/TinyStories`) — synthetic short
stories written with a deliberately small vocabulary so tiny models reach coherence.

**Rejected.**
- *Char-level corpus* — pretrains fine at this scale but stays too weak to align.
- *Project Gutenberg* — public domain and clean, but far too hard for 14M params;
  the base model would never get coherent enough to show an SFT delta.

**Consequence.** Fluency is reachable in hours on a 4GB GPU, and the SFT/DPO deltas
are visible to a reader. We inherit the dataset's synthetic-English register: nLemon-14
speaks children's-story English and nothing else. That is a stated limit, not a bug.

---

## ADR-002 — No database; files only

**Context.** Every stage produces state: corpus, tokenizer, weights, metrics, samples.

**Decision.** Files. `.txt`/`.bin` for data, `.json` for the tokenizer, `.pt` for
weights, `.csv` for metrics, `scorecard.json` for results, `.md` for samples.

**Rejected.** SQLite / a vector store — there are no queries to serve and nothing to
retrieve. The deliverable is a checkpoint, not a service; a DB would be complexity to
defend rather than justify.

**Consequence.** The whole run is inspectable with a text editor and diffable in git.

---

## ADR-003 — Website = static showcase

**Context.** The public surface needs to show the model, not host it.

**Decision.** A static page with curves, samples and the scorecard baked in.

**Rejected.** Live inference at launch — real infrastructure and real cost to defend,
for a model whose point is the *lifecycle*. Kept as an optional Phase-8 add-on
(an HF Space) so Phases 1–7 stay infra-free.

**Consequence.** Launch has no uptime risk and no bill.

---

## ADR-004 — Config + seed as the single source of truth

**Context.** The public claim is "clone → one command per stage → identical scorecard".
That claim dies the moment a hyperparameter lives in a script.

**Decision.** Every hyperparameter and the seed live in `configs/nlemon_14m.yaml`,
loaded into a frozen dataclass. Unknown keys in the YAML are a hard error, never a
silent drop. Every artifact records `config_hash`. VRAM pressure is absorbed **only**
by `micro_batch` / `grad_accum_steps` — model shape is never changed to make something
fit, because that would quietly invalidate every earlier number.

**Rejected.** CLI flags as the primary interface — they leave no trace in the artifact.

**Consequence.** Any result can be traced to the exact 12-char config hash that made it.

---

## ADR-005 — Deterministic feature-checker as the alignment judge

**Context.** Phases 5 and 6 need to score instruction-adherence and a DPO win-rate.

**Decision.** A rule-based checker (did the output mention the requested subject? is it
in the right length band? is it actually a story? is it degenerate/repetitive?).

**Rejected.** LLM-as-judge — non-deterministic, unreproducible, and it would undercut
the one claim this project is built to defend. It also imports a larger model's opinion
into an artifact that is supposed to stand alone.

**Consequence.** The judge is blunt and we say so publicly: it catches on-topic-ness,
length and degeneracy, and it cannot catch style or subtle quality. A blunt judge whose
limits are stated beats a sophisticated one whose output cannot be reproduced.

---

## ADR-006 — Use the dataset's own train/validation boundary

**Context.** Phase 1 needs a train/val split, and validation perplexity is the Phase 4
gate. A split we invent is a split we can get wrong.

**Decision.** Use TinyStories' upstream `train` and `validation` splits verbatim. The
data script hard-fails if either is missing rather than falling back to a homemade split.

**Rejected.** Carving our own random val slice out of `train` — it introduces a
`val_fraction` hyperparameter, risks near-duplicate stories straddling the boundary
(the corpus is synthetic and repetitive by design), and makes our perplexity
incomparable to anyone else's on the same dataset.

**Consequence.** The split is fixed upstream and pinned by commit sha in
`data/manifest.json`, so val perplexity is stable across machines and reruns.

---

## ADR-007 — Pin the dataset by commit sha, record license in the manifest

**Context.** `revision: main` on a Hub dataset is a moving target.

**Decision.** The config requests a revision; the data script resolves it to an exact
commit sha, and writes that sha, the Hub-declared license, per-split counts and a
SHA-256 of each shard into `data/manifest.json`.

**Rejected.** Trusting the download to be stable — an upstream re-upload would silently
change the corpus under a checkpoint that claims to be reproducible.

**Consequence.** The corpus step is auditable: a reader can verify they built the same
bytes we did before comparing any number.

---

## ADR-008 — Drop empty documents, keep short ones

**Context.** The upstream train split contains 230 documents (of 2,119,719) that are
empty after stripping, plus 3 shorter than ten words. The validation split has none.

**Decision.** Drop documents that are empty after stripping; keep everything else,
including the very short ones. The count dropped is printed and recorded per split in
`data/manifest.json`.

**Rejected.**
- *Keep the empties* — an empty document writes two separators back to back, so the
  model would see `<|endoftext|>` immediately following `<|endoftext|>` 230 times. That
  is a small but real nudge toward emitting an instant end-of-text, which is precisely
  the degeneracy the Phase 6 gate is supposed to punish. Cheap to remove, so remove it.
- *Also drop the very short documents* — that would need a minimum-length threshold,
  i.e. a new hyperparameter to pick and defend. "Empty after stripping" is a property of
  the document, not a tuning knob. Three short documents in 2.1M change nothing.

**Consequence.** 0.011% of the train split is discarded. It is counted in the manifest
rather than silently absorbed, so the shard document count will not match the upstream
dataset count and the reason is on the record.

---

## ADR-009 — Repair upstream mojibake before tokenizing

**Context.** The upstream TinyStories text is double-encoded in places: UTF-8 bytes that
were once decoded as cp1252 and re-saved. A measured scan of the shards found **757,666**
such sequences in train and 7,156 in validation — overwhelmingly `â€™` for `’` and
`â€œ` / `â€` for curly double quotes, plus a long tail of `Ã©`, `Ã±`, `Â­`.

**Decision.** Two passes, in order, with both counted in `data/manifest.json`:

1. `ftfy.fix_text` on every document — repairs ~626k of the sequences.
2. An explicit residue map for what ftfy provably cannot fix (`src/data.py: RESIDUE`).

**The second pass is not optional, and finding out why was the point.** Measuring after
pass 1 showed 130,379 sequences left *and* **140,572 newly-created U+2020 DAGGER
characters that were not in the source at all** — verified by counting daggers in the raw
dataset: zero. Some curly quotes lost their third UTF-8 byte upstream, and for one variant
of that corpse ftfy re-encodes into a valid dagger. That is strictly worse than the
original garbage: obvious mojibake is a rare character the tokenizer wastes a little
vocabulary on, whereas a dagger is a plausible character it learns *cleanly* and will emit
with confidence. So the residue map rewrites both the corpse and the dagger to an ASCII
`"` — correct for the opening and the closing case alike, and consistent with ftfy's own
default of uncurling quotes to ASCII.

**Rejected.**
- *Leave it and let BPE deal with it* — the expensive option in disguise. With a
  vocabulary of only 8,000, merges spent on `â€™` are merges not spent on English, and
  the model would learn to emit the garbage faithfully because it is genuinely frequent.
  Phase 2's lesson is that tokenization silently shapes everything downstream; this is
  that lesson arriving early.
- *Trust `ftfy` and stop there* — what we would have shipped without measuring the
  result. It would have quietly traded 757k visible defects for 140k invisible ones.
- *Hand-rolled `s.encode("cp1252").decode("utf-8")`* — works for the common case and
  throws or corrupts on the tail. Not worth re-deriving a solved problem badly; ftfy
  earns its place on pass 1.
- *Guess open-vs-close curly quotes from context* — a heuristic with no ground truth to
  check it against, for a distinction ftfy discards anyway.
- *Strip non-ASCII entirely* — would also delete legitimate accented names.

**Consequence.** One extra dependency (`ftfy`, pure python) and a data build that takes
minutes rather than seconds. **Measured result: 757,666 → 9 stray characters across
1.84 GiB of train text, and validation fully clean.** The build prints the residual count
every run and records a context snippet for each in `data/manifest.json`, so a future
upstream change that reopens this cannot pass unnoticed.

Our shards are therefore **not** byte-identical to the upstream text. The manifest records
the upstream commit sha, the per-split repair counts and our shard SHA-256, so the
transformation is verifiable from both ends.

### Footnote: what the remaining 9 characters are

Every residual character was located and read in context before this ADR was closed. The
first audit found 18; **9 of those were self-inflicted** and are now fixed. The rest are
recorded here rather than chased.

| Codepoint | n | What it actually is | Action |
|---|---|---|---|
| `U+00C2` Â | 9 | Orphaned mojibake prefix: `Â` followed by **ASCII** — `wasn<Â>'t`, `loved.<Â>`, `he<Â>'d`. `C2` plus an ASCII byte is not a valid mojibake pair, so ftfy cannot interpret it and leaves it. | **Fixed** by a rule for the orphaned prefix. Audited: all 9 were of this kind, none a real capital A-circumflex. |
| `U+00E2` â | 1 | **A false positive — legitimate French.** `papier-mâché`, correctly spelled. | **Left alone.** Deleting it would corrupt correct text. |
| `U+00E2` â | 2 | A mojibake'd emoji: `'Iâ¤ï¸ U'` was `'I❤️ U'`. Same failure as the quotes — the third byte died upstream. | **Left alone.** Unrecoverable, and 2 occurrences in 364M words. |
| `U+20AC` € | 6 | **One non-English document.** A single story (of 2,119,489) starts in English and continues in double-encoded Traditional Chinese. A whole-corpus scan confirms exactly **1 document contains CJK, 46 characters total**. | **Left alone.** Repairing a corpus-wide encoding fault is in scope; hand-fixing one anomalous document is not. |

The false positive is the reason `SUSPICIOUS` is deliberately broader than `RESIDUE`: the
audit flags what is *usually* damage and records context, and a human decides. A cleaner
that silently deleted every `â` would have eaten `papier-mâché`. The residual count will
therefore sit at 9 rather than 0 on a correct build — a known, explained floor, not an
open defect.

---

## ADR-010 — Per-stage config hashes alongside the global one

**Context.** Phase 2 added two fields to the config. That changed `Config.hash()` from
`be96725bd672` to `53f4919fceb7` — and therefore made every Phase 1 artifact look stale,
despite the corpus on disk being byte-for-byte identical. Left alone, this gets worse every
phase: by Phase 6 the global hash has moved five times and stops meaning anything, which is
precisely when we need it to mean something.

**Decision.** Keep the global hash, and add `Config.stage_hash(stage)` over a declared
subset of fields (`STAGE_FIELDS` in `config.py`). Artifacts record both. The contract: two
builds with the same *stage* hash must produce identical artifacts for that stage,
whatever else changed in the config.

**Rejected.**
- *Only the global hash* — conflates "the config changed" with "this artifact changed".
  Every later phase would force a spurious re-verification of every earlier one.
- *Only stage hashes* — loses the single fingerprint that identifies a whole run, which is
  what the Phase 7 scorecard needs to be reproducible from.
- *Deriving the field list automatically* (e.g. tracing attribute access) — clever, fragile,
  and it would silently drop a dependency the day someone reads a field indirectly. An
  explicit list is auditable; a reviewer can check it against the code.

**Consequence.** `STAGE_FIELDS` is now a thing that can be wrong: omit a field a stage
really depends on and the stage hash will under-report a real change. It is a short,
reviewable list next to the config it describes, and the Phase 1 rebuild verified the
mechanism — adding two tokenizer fields left the data stage hash and both shard SHA-256s
untouched.

---

## ADR-011 — Byte-level BPE

**Context.** Phase 2's gate is a **lossless** encode→decode roundtrip, exact string match.
The corpus also still contains 9 characters of unrepairable non-ASCII (ADR-009) and one
document of Traditional Chinese.

**Decision.** Byte-level BPE (GPT-2 lineage): the initial alphabet is all 256 bytes, and
the pre-tokenizer/decoder pair is `ByteLevel` with `add_prefix_space=False`.

**Rejected.**
- *Character-level BPE with an `<unk>` token* — makes the roundtrip gate unpassable by
  construction: any character outside the vocabulary decodes back as `<unk>`, so the exact
  string match fails on exactly the rare inputs we most want to survive. Passing would then
  require weakening the gate, which is backwards.
- *A pretrained tokenizer (tiktoken / GPT-2's)* — the project's claim is a model built from
  scratch; borrowing a 50k vocabulary trained on the open web would import someone else's
  corpus statistics and waste most of the vocabulary on text this model will never see.
- *`add_prefix_space=True`* — invents a leading space the decoder must strip, a classic
  silent roundtrip break.

**Consequence.** There is no OOV and no unknown token; every byte sequence roundtrips by
construction rather than by luck. The cost is that rare non-ASCII costs several tokens
each, which is the correct trade at 8k vocabulary — and the gate now tests a property the
design guarantees, rather than hoping the corpus stayed clean.

Verified: lossless roundtrip on 2,000 held-out validation documents (1,591,027 characters,
zero mismatches), and separately on the adversarial set this corpus actually contains —
`papier-mâché`, the mojibake'd emoji, the Traditional Chinese document, leading/trailing
whitespace, and the empty string.

---

## ADR-012 — Train the tokenizer on a bounded subset, encode everything

**Context.** The train shard is 1.84 GiB. BPE training holds counts in memory, and this
build machine had ~1.8 GiB of RAM free when Phase 2 started. ADR-004 forbids absorbing a
resource limit by changing the artifact's shape.

**Decision.** `tokenizer_train_docs` (200,000) bounds how much text the BPE *trains* on.
The entire corpus is still encoded. The number is measured, not guessed:
`results/tokenizer_subset_sweep.md`, regenerable via `python -m src.tokenizer sweep`.

**Rejected.**
- *Train on the full corpus* — buys nothing measurable. Compression plateaus at 25,000
  documents; across a 32x increase in training text the spread is 0.0007 tokens/word.
- *Take the cheapest rung that plateaus (25,000)* — this is the interesting one. Judged on
  compression alone, 25k is indistinguishable from 800k. But vocabulary **overlap** at 25k
  is only 92.6%, versus 97.4% at 200k. Two tokenizers can compress identically while
  disagreeing about which rare tokens earned a slot, and those disagreements are exactly
  where a 14M model's vocabulary budget is won or lost. Measuring only the headline metric
  would have picked the worse tokenizer for the right-looking reason.
- *Shrinking `vocab_size` to fit RAM instead* — that is a silent model-shape change, which
  ADR-004 exists to forbid.

**Consequence.** One more config knob to defend, backed by a regenerable table. Training
takes seconds rather than minutes, and the sweep doubles as the honest answer to "why 8k,
and why that much text?" — a Phase 2 quiz question with evidence attached rather than an
opinion.

Also verified: training the BPE twice from the same seed and corpus produces a
byte-identical vocabulary, so the tokenizer is part of the reproducible chain rather than
a lucky artifact.

---

## ADR-013 — Check the parameter count twice, from independent derivations

**Context.** Phase 3's gate is "param count matches the ~14M budget". The obvious
implementation sums `model.parameters()` and compares against 14M.

**Decision.** Do that *and* derive the count analytically from the config in
`expected_params()`, then require the two to agree exactly. The gate fails if they differ,
even when both sit inside the budget.

**Rejected.**
- *Count the modules only* — this is a tautology dressed as a test. It proves the model is
  self-consistent with whatever got built. If the output head were accidentally untied,
  the count would rise by 3.07M and *still* be "the correct count for this model", and
  still land inside a ±10% budget. The check would pass while the architecture was wrong.
- *A hardcoded expected integer* — correct exactly once, then silently wrong the first
  time anyone changes `d_model`. The formula tracks the config; a magic number does not.
- *A wider tolerance instead of a formula* — tolerance hides structural mistakes, which are
  the only kind this gate can catch.

**Consequence.** Two derivations must be kept in step, so a real architecture change means
editing the formula too. That is the point: it forces the change to be deliberate. Measured
agreement: **13,817,856 counted == 13,817,856 analytic**, 98.70% of the 14M budget. Weight
tying is asserted separately rather than inferred from the total.

---

## ADR-014 — The single-batch overfit does not test the causal mask

**Context.** Phase 3's second gate is "overfit a single batch, loss drives to ~0, proving
the wiring is correct". It is the standard nanoGPT-era sanity check.

**Decision.** Keep it, but add a separate causal-mask test, because the overfit alone
cannot detect a missing mask.

**Why.** A model with **no** causal mask memorises a batch *faster*, not slower — it can
read each answer off the following token. So a green overfit is fully compatible with
broken masking. That bug does not show up in training at all; it shows up as a model that
looks excellent on teacher-forced loss and emits gibberish the moment it generates
autoregressively, several phases later, with nothing pointing back here.

The mask test asserts the actual property: changing the token at position *t* must not move
the logits at any position *< t*. It also asserts the logits *do* move from *t* onward,
otherwise the first half would pass vacuously on a model that ignores its input.

**Rejected.**
- *Trusting `is_causal=True`* — it is one keyword away from silently doing nothing, and the
  whole point of this project is checks that survive someone editing the code later.
- *Inspecting the attention weights* — brittle, and it tests the implementation rather than
  the property. The behavioural check survives a rewrite to a different attention kernel.

**Consequence.** Measured: poking position 16 of 32 moves earlier logits by exactly `0.0`
and later logits by `1.22`. The gate now covers a failure the specified gate would have
missed.

---

## ADR-015 — Gate thresholds are chosen by their worst seed

**Context.** Adding the causality check to the gate flipped the overfit result from 0.079
to 0.593 and turned the gate red. The model had not changed at all — the new check consumed
RNG before the overfit ran, so the model initialised differently.

**Decision.** Two changes. `overfit_one_batch` reseeds at entry, so its verdict cannot
depend on what ran before it. And the step count is chosen by measuring the **worst** of
several seeds, not a single lucky run: `results/overfit_margin.md`, regenerable with
`python -m src.model margin`.

**Rejected.**
- *Keep 300 steps and loosen the target* — the measurement says the worst seed at 300 steps
  reaches 0.42, so the target would have to rise past 0.5 to be reliable. A gate that
  accepts a loss of 0.5 as "drives to ~0" is not testing the thing it claims to test.
- *Keep 300 steps and re-run until green* — the failure mode this project exists to avoid.
- *Raise the learning rate instead* — measured and rejected on evidence: at 3e-3 the worst
  seed is further from the target at every step count, and one run ended at 2.12 after
  touching 0.14. It oscillates rather than converging.

**Consequence.** The gate costs ~34 seconds instead of ~17. In exchange, its worst observed
seed lands ~9x under the target instead of 4x over it. The general rule this establishes for
later phases: a threshold reported from one seed is an anecdote, and the spread across seeds
must be smaller than the distance to the threshold, or the gate is measuring luck.

---

## ADR-016 — Seeding is not reproducibility; strict determinism is

**Context.** `utils/seed.py` has seeded python, numpy and torch and set the deterministic
cuDNN flags since Phase 1, and the README has claimed "clone, one command per stage,
identical scorecard" since the first commit. Phase 3 is the first stage that actually
*trains* something, so it is the first chance to test that claim rather than assert it.

It does not hold. Running the identical gate three times gave final losses of **0.01515,
0.01034 and 0.01264**. The first-step loss was identical every time (9.11523), so
initialisation was never the problem: the seed fixes the starting weights, and the backward
pass then accumulates gradients with atomics whose ordering varies run to run. cuDNN's
deterministic flag does not cover the fused attention kernel or cuBLAS reductions.

Left alone, this makes the Phase 7 scorecard irreproducible by construction — every number
in it would be downstream of a checkpoint that cannot be rebuilt exactly.

**Decision.** `strict_determinism: true` in the config, wired into `set_seed`, which now
calls `torch.use_deterministic_algorithms`. `CUBLAS_WORKSPACE_CONFIG` is set at *import*
of `utils.seed`, because cuBLAS reads it once when the CUDA context is created and setting
it later is silently too late.

**Rejected.**
- *Leave it and report mean ± spread* — honest, and a legitimate choice for a paper, but it
  abandons the specific claim this project was built to defend. The claim is reproducible
  artifacts, not reproducible-ish ones.
- *Assume the seed was enough* — what we shipped in Phases 1 and 2. It was true there only
  because neither stage trains anything: the corpus and the tokenizer are deterministic for
  unrelated reasons, which made the gap invisible until now.
- *Only turn it on for eval* — eval is forward-only and already deterministic. The
  irreproducible artifact is the checkpoint, so determinism has to cover training.

**Consequence.** Measured cost: **73,014 to 68,130 tokens/sec, about 7%**. Measured
benefit: the gate is now bit-identical across runs (0.01075 / 0.00729, twice). Phase 4
pretraining will pay that 7% for a checkpoint that can actually be rebuilt.

The residual risk is that a later phase needs an op with no deterministic CUDA
implementation, which will raise rather than silently drift. That is the correct failure
mode: it forces the reproducibility claim to be re-stated in public rather than quietly
dropped.

---

## ADR-017 — The perplexity threshold is anchored to measured floors, and agreed first

**Context.** Phase 4's gate is "validation perplexity below an agreed threshold". A number
with nothing behind it is decoration: 8.0 sounds strict, but nobody reading the README
knows whether it is.

**Decision.** Measure what trivial predictors score on the same held-out shard
(`src/baselines.py`), then set the bar as a stated multiple of the strongest one, and agree
it **before** the training run finishes.

Measured floors on 2,000,000 held-out tokens:

| baseline | knows | perplexity |
|---|---|---|
| uniform | nothing | 8,000.00 |
| unigram | which tokens are common | 389.91 |
| bigram | the previous token only | 41.81 |

Threshold agreed at **5.2 times the bigram floor: perplexity <= 8.0**. Achieved: **5.1981**
on the gate metric, **5.4636** on the identical token array the baselines use — **7.65x
better than bigram**, 71.36x better than unigram, 1,464.24x better than uniform.

**Exact timing of the agreement, because "pre-registered" is doing real work here.** The
order was: measure the floors, run a 60-step pilot (validation perplexity 154.6), launch
the full run, then propose and agree the bar — all inside a few minutes. Two claims that
would be convenient are both false, and the record should not make either:

- *"The bar was fixed before step 0."* False. The full run was already underway.
- *"The bar was fixed before any result existed."* Also false, and this one is the subtler
  trap: the 60-step pilot **was** a result, and it existed first. Anyone reading the logs
  will find it.

The true claim is narrower and survives that reading. **The only result in hand was a
60-step pilot at perplexity 154.6, which cannot forecast where a 20,000-step, 0.70-epoch
run lands** — 154.6 is between the unigram floor (389.91) and the bigram floor (41.81),
i.e. the model had barely learned token frequencies and had not yet reached the weakest
context-using baseline. Nothing in it distinguishes a finish at 4 from a finish at 12, and
8.0 was not derived from it: the bar came from the measured floors, as a stated multiple of
bigram. No figure from the full run informed it, and the gate did not cross 8.0 until step
3,000 — live for the first 15% of training.

So the defensible sentence is: *"fixed before any full-run result was seen, from a pilot
that could not predict the outcome"* — not "before step 0", and not "before any result".

**Rejected.**
- *Pick a threshold after seeing the result* — the failure this project exists to avoid.
- *"Below the bigram baseline"* — too weak to mean anything. A model that barely beats a
  two-token lookup table has not learned to speak.
- *An absolute number from the literature* — perplexity is not comparable across
  tokenizers. Our 8,000-token vocabulary makes any figure from a 50k-vocab paper
  meaningless here, which is exactly the sort of borrowed authority the ADR log exists to
  refuse.

**Consequence.** The claim is defensible in the specific form "7.65x better than a bigram
model on the identical held-out tokens with the same tokenizer", checkable with
`python -m src.baselines --with-model checkpoints/base.pt`. Every figure in that table is
per-token negative log-likelihood over the same BPE, which is the only way the ratio means
what it says: a word-level baseline against a token-level model would compare different
denominators and inflate the result.

The gate also **recomputes** perplexity from the checkpoint rather than reading
`train_summary.json` — the summary is what the run *said*, and the gate should verify it
independently (same reasoning as ADR-013).

---

## ADR-018 — Coherence bands come from the corpus, not from taste

**Context.** The other half of Phase 4's gate is "samples are coherent little stories, not
word salad". That is a human judgement and it stays one: the Product Owner reads
`results/samples/base_samples.md` and calls it. But a purely human gate cannot be re-run,
and Phase 5 needs a deterministic checker anyway (ADR-005), so the automatic half is built
here.

**Decision.** Measure five statistics on **real validation documents**, take the p5-p95
band of each, and require the median over generated samples to fall inside it
(`results/coherence_reference.md`). The question becomes falsifiable: *does this resemble
the corpus?*

Both edges are checked deliberately. Excess repetition is degenerate; **too little is also
suspicious**, because real children's stories repeat names and phrases on purpose, and a
model that never repeats is not imitating this corpus.

**Rejected.**
- *Hand-picked thresholds* — "repetition below 0.3" is one person's guess wearing the
  costume of a measurement. Every number here is derived from the data it judges.
- *A one-sided test* — would pass text that is statistically unlike the corpus in the
  "better than real" direction, which for an imitation task is still wrong.
- *Mean instead of median across samples* — one degenerate sample would drag the average
  down and condemn fifteen good ones, or be averaged away by them. Per-sample detail is
  recorded either way so a red result can be read rather than guessed at.
- *LLM-as-judge* — already rejected in ADR-005 and still rejected.

**Consequence.** Measured result at `base.pt`: all five metrics in band, e.g. repeated
4-gram rate 0.0155 in [0.0000, 0.0333], mean sentence length 9.63 words in [7.07, 14.27].

**Known weakness, stated rather than hidden:** `known_word_rate` has a degenerate band of
[1.0000, 1.0000] — over 95% of real validation documents use only words that appear in the
100,000-document known-word set, so the p5 and p95 edges collapse onto the same point. It
passed cleanly here (the model emitted no out-of-corpus words at the median), but as a
*band* it is a point test, and any single invented word at the median flips it. If a later
phase needs to distinguish "invented a plausible name" from "emitted garbage", this metric
needs a wider known-word set or a different formulation. It is recorded as a limitation
now rather than discovered as a mystery later.

---

## ADR-019 — Finish the run rather than optimise the loader mid-flight

**Context.** Throughput during pretraining was erratic: 71,000 tokens/sec when the OS page
cache was warm, falling to ~22,000 once free RAM dropped from 6.7 GiB to 3.4 GiB. GPU
utilisation at the low point was 40% at 11.6 W and 49 C — the GPU was idle, waiting for
data. `ShardLoader` issues 64 random reads per optimizer step into an 888 MiB memmap, so
once the shard stopped fitting in cache, random-access latency set the pace.

**Decision.** Let the run finish. Record the cause and the fix without applying it.

**Rejected.**
- *Kill and restart with a batched or pre-shuffled loader* — would have discarded 8,500
  completed steps to recover maybe an hour, and would have produced a checkpoint under a
  different data-access order, invalidating the loss curve already written.
- *Quietly not mention it* — the run took 120 minutes where ~70 was achievable. The honest
  version of "trainable in hours on a 4GB laptop" includes why it was slower than it needed
  to be.

**Consequence.** 120 minutes for 327,680,000 tokens (0.70 epochs). The known fix, for
whoever picks this up: draw each batch from a contiguous block, or shuffle the shard once at
build time so sequential reads suffice. Phases 5 and 6 fine-tune on far less data, so this
is not on their critical path — the reason it is an ADR rather than a task.

---

## ADR-020 — The gate metric was underpowered, and the published ratio must be the conservative one

**Context.** Review of the Phase 4 write-up caught a wrong ratio: the README stated a
bigram improvement of 6.9 when 41.81 / 5.1981 = 8.04. The cause was a **stale denominator**
— the ratios were worked out by hand during a progress update, when validation perplexity
was 6.097 at step 8,000, and carried into the final prose without being recomputed against
the finished number. Both published ratios understated the model.

Chasing the correct denominator surfaced a second, more interesting problem: there is more
than one defensible value.

| measurement | tokens | val perplexity |
|---|---|---|
| gate as pre-registered, 100 batches, loader seed 1338 | 409,600 | **5.1981** |
| same setting, loader seed 1339 | 409,600 | 5.2796 |
| same setting, loader seed 1340 | 409,600 | 5.2664 |
| random windows, 500 batches | 2,048,000 | 5.2635 |
| random windows, 1,000 batches | 4,096,000 | 5.2680 |
| contiguous 2M prefix, identical tokens to the baselines | 1,992,060 | 5.4636 |

`eval_iters: 100` carries roughly ±0.04 perplexity of sampling noise, and the gate's 5.1981
sat at the **optimistic edge of its own spread**. Not wrong — it is the pre-registered
procedure run honestly — but not a number to build a headline ratio on.

**Decision.** Three things.

1. Published ratios use the model scored on the **identical token array** as the baselines
   (`--with-model`), giving **7.65x better than bigram**. That is the *worst* of the
   candidate numbers, and it is the one that ships.
2. Ratios are computed in code into `perplexity_floors.{md,json}` and verified against the
   prose by `python -m src.verify_docs`. Hand arithmetic in narration is exactly how the
   stale denominator survived, so narration no longer gets to do arithmetic.
3. The gate metric stays as pre-registered at `eval_iters: 100`. Raising it now would be
   changing the measurement after seeing the result — the sin this entry exists to catch —
   and it would move `train_stage_hash`, making `base.pt` look stale over a parameter that
   does not affect a single weight.

**Rejected.**
- *Publish 5.1981 with the corrected ratio of 8.04* — arithmetically right, but it puts the
  friendliest of six measurements in the headline.
- *Silently swap in the conservative number* — the gap between measurements is itself the
  finding; burying it would waste it.
- *Re-run the gate with more batches and publish that* — post-hoc measurement changes break
  pre-registration even when they make the result look worse.

**Consequence.** The published claim moves from an incorrect 6.9 to a correct and
conservative 7.65, and every candidate figure clears the ≤ 8.0 bar, so the verdict never
depended on the choice. Two follow-ups are now on the record:

- **Phase 7's scorecard must use a large fixed evaluation set**, not 100 random batches.
  Reproducible-by-seed is not the same as low-variance, and a scorecard needs both.
- **`eval_iters` does not belong in `STAGE_FIELDS["train"]`.** It changes the reported
  metric, not the weights, so it belongs to an eval stage. Left in place for now because
  editing the field list would move `base.pt`'s recorded hash for no real change; the fix
  lands when Phase 7 introduces an `eval` stage.

---

## ADR-021 — A green check is a claim, and claims get checked

**Context.** Building the doc verifier for ADR-020, I made it skip ratio figures inside
double quotes, so an ADR could cite the number it was correcting without tripping its own
guard. Quote-pairing across a whole file is unreliable: the pairing drifted, and the
checker **silently skipped three real claims in ADR-017 while printing "all published
ratios match"**. I only noticed because the output listed one ADR claim where I knew there
were four.

That is the third instance of the same failure in a single phase:

| # | The check | What it was actually measuring |
|---|---|---|
| ADR-015 | single-batch overfit gate | initialisation luck — the same correct model passed or failed on RNG |
| ADR-018 | `known_word_rate` coherence band | nothing, at p5 = p95 = 1.0 the "band" is a point |
| ADR-021 | doc-ratio verifier | the subset of claims its regex happened to reach |

Each one was **green**. None of them was measuring what its name said. A red check gets
investigated; a green one ends the conversation, which is exactly why a wrong green is more
dangerous than a wrong red.

**Decision.** Exemptions are removed from the verifier — every `Nx better` in the docs is
checked, with no escape hatch. Historical or incorrect figures are phrased without that
pattern ("a ratio of 6.9") so they are visibly not claims. More generally, a check that can
skip inputs must report what it skipped, and a check whose pass condition cannot fail is not
a check.

**Rejected.**
- *Fix the quote-pairing* — a better regex would have made the same class of bug quieter
  rather than absent. The problem was that skipping was silent, not that the skip rule was
  imprecise.
- *Trust the green and move on* — three for three this phase says otherwise.

**Consequence.** Every check added from here states, in its own output, the size of what it
examined: the encoding guard prints "41 tracked files, 10 golden examples intact", the
verifier lists each claim it checked by line number, and the coherence gate prints the
sample count. A count you can eyeball against expectation is what turned this one up.

This is also the honest spine of the Phase 4 write-up. The interesting story is not "the
model learned to speak" — small models on TinyStories do that. It is that three separate
times in one phase, the thing that nearly fooled me was a check reporting success.

**Postscript, from the very next check written.** The `RESIDUE` regression test
(`tests/test_residue.py`) was built to defend an ordering rule that ADR-009 and a comment in
`src/data.py` both described: the orphan-prefix rule "must run last, after the pair rules
strand it". The test asserted that reordering `RESIDUE` should break the fixtures — and the
assertion **failed**, because reordering broke nothing.

The documented hazard was not real. `ftfy` repairs `Â`+no-break-space and `Â`+soft-hyphen
before `RESIDUE` runs, so those pairs never reach the rules; the 9 real orphans were `Â`
followed by *ASCII*, which ftfy cannot touch. 400 permutations give identical output and no
rule pattern contains another: the set is order-**in**dependent. The test now protects that,
plus the structural invariant that keeps it true, and both wrong descriptions are corrected.

A documented hazard that does not exist is its own kind of misinformation — it makes the
next person preserve an ordering for a reason that was never true, and hesitate to touch it.
Worth noting the mechanism: the test found this only because it was written to *prove its
own fixtures had teeth* rather than merely to pass. That is now the cheapest available
defence against this whole family.

---

## ADR-022 — Rare events get pooled and bootstrapped, not medianed

**Context.** ADR-018 recorded `known_word_rate` as a known weakness: its band was
[1.0000, 1.0000], because over 95% of real validation documents contain no
out-of-vocabulary words, so the p5 and p95 edges collapsed onto a point. It passed at
`base.pt`, but a point is not a band. Phase 5 is where it bites — SFT will produce names
the corpus never used, and the question becomes "invented a plausible name" versus "emitted
garbage", which vocabulary membership **cannot answer**: both are unknown words.

**Decision.** Replace it with two metrics that measure the right things in the right way.

1. **`oov_rate`, pooled.** Counted over every word in the whole sample set rather than
   per-document. A rare event needs a big enough denominator to vary at all. Its band is
   bootstrapped: real documents are drawn in groups the same size as the generated set
   (16), 400 times, and the band is the spread across groups — the same statistic at the
   same sample size as the thing being judged.
2. **`oov_plausibility`.** Mean character-trigram log-probability of the words that *are*
   out of vocabulary, under a model trained on corpus words. Character shape answers what
   membership cannot.

**Measured, and it separates cleanly.** Corpus band [-2.7517, -1.8522]:

| | mean score | verdict |
|---|---|---|
| invented names (`timmothy`, `hoppity`, `sparklewing`) | -2.229 | in band |
| rare real English (`kaleidoscope`, `marmalade`) | -2.696 | in band |
| garbage (`xqzvt`, `bcdfghj`, `tttkkkzz`) | **-4.268** | far outside |

Every garbage string scores below -4.0 and every plausible one above -3.4, against a band
floor of -2.75. `oov_rate` is now [0.0000, 0.0021] — a real distribution instead of a point.

**Rejected.**
- *Widen the known-word set to the full corpus* — the obvious fix, and it makes the problem
  worse: more known words pushes the rate closer to 1.0 and the band tighter onto its point.
  The degeneracy was never about the word list, it was about medianing a rare event.
- *Keep the metric and loosen the band by hand* — inventing a threshold to paper over a
  measurement that carries no information.
- *Drop the metric* — it is the one Phase 5 most needs.

**Consequence.** The gate reports `oov_plausibility` as "no out-of-vocabulary words to
score" at `base.pt`, because the pretrained model invents nothing — correctly recorded as
not-applicable rather than silently passed as 0.0, which would have read as maximally
implausible. The metric activates precisely when SFT starts inventing, which is when it is
needed. `known_word_rate` is retired from the banded set.

**Untested where it matters.** Both bands were built from validation *fixtures*, and
`oov_plausibility` has never yet fired on live model output — `base.pt` produces no
out-of-vocabulary words at all. The first SFT gallery is the first time this band meets the
condition it was designed for, and instruction-following is exactly what might push the
model off the TinyStories distribution. That cell gets read first, not last.

---

## ADR-023 — Hold out subjects, not phrasings

**Context.** Phase 5's gate measures instruction-adherence on "held-out prompts". There are
two ways to hold out, and they answer different questions.

**Decision.** The held-out split is over **subjects**. All four prompt templates appear in
both the training pairs and the evaluation set, so phrasing is held constant; the 80
evaluation subjects are disjoint from the 320 training subjects.

**Rejected.** *Holding out phrasings* — training on "Write a story about X" and evaluating
on "Can you write a story about X?" measures generalisation across four sentence forms. It
is a real question and an easier one, and it does not survive being asked "did it just
memorise your templates?" Holding out the subject asks whether the model follows an
instruction about a thing it was never instructed about.

**Consequence.** The gate must name which question it answers, and it does: adherence on
subjects the fine-tune never saw. Two construction details make that claim honest:

- **Subjects are mined from the corpus, not hand-listed** — words following a determiner,
  above a frequency floor. A hand-written list would encode my guesses about what
  TinyStories is about.
- **The split is rank-stratified, not random.** Drawing 20% at random would have made the
  held-out pool systematically rarer than the training pool, so the evaluation would be
  harder for a reason unrelated to instruction-following. Measured after stratifying:
  median corpus frequency 1,959 (train) versus 1,988 (held-out).

A **headedness filter** was needed and is worth recording. The determiner test alone
promotes adjectives: "a brave bird" makes `brave` look like a subject, and "Write a story
about a brave." is not an instruction. A candidate must be the *head* of its phrase — 
followed by punctuation or a word that cannot be a noun — at least 50% of the time. That
dropped 153 modifier-like words (`special`, `best`, `loud`, `great`, `beautiful`, `red`).
The filter is corpus-derived rather than a hand-written adjective list I would have to keep
complete.

---

## ADR-024 — Report the sub-scores separately; never blend an adherence scalar

**Context.** "Instruction adherence" is the Phase 5 headline, and the obvious presentation
is one number.

**Decision.** Four sub-scores, reported separately, split into two groups that are **not**
combined:

| sub-score | role | why |
|---|---|---|
| `subject_mention` | **delta** | carries the instruction-following claim |
| `length_band` | **delta** | did it produce a story-shaped response |
| `is_story` | floor | near-saturated before SFT; almost no headroom |
| `not_degenerate` | floor | a fine-tune that degenerates must fail |

**Why `is_story` is deliberately excluded from the delta.** `base.pt` emits stories
unconditionally — that is what pretraining on TinyStories produces. So this sub-score is
close to saturated before SFT touches anything, and a green reading is not evidence the
fine-tune worked. It is `known_word_rate`'s degenerate band wearing a different hat
(ADR-022), caught this time *before* it was published as signal rather than after. It earns
its place as a floor: a fine-tune that stops producing stories must fail.

**Rejected.** *A single blended adherence number* — subject-mention and length-band move for
different reasons, and averaging them hides which one SFT actually bought. A model that
learned to name the subject but produce stubs would look identical to one that learned
length but ignored the instruction.

**Consequence.** The gate pre-registers a threshold per delta sub-score, not one on a
composite. Two numbers are harder to headline and impossible to game by improving the easy
one.

---

## ADR-025 — The gate pins its own decoding, separately from the global default

**Context.** Phases 5 and 6 compare checkpoints. A comparison is about the checkpoints only
if everything else is held still, and decoding is the easiest thing to vary by accident:
`results/decoding_sweep.md` shows the same model landing in or out of the corpus bands
depending on temperature alone. Comparing `sft.pt` to `base.pt` at different settings would
measure the decoder.

**Decision.** Two separate guards, because there are two separate failures.

1. *Between stages*: both checkpoints are scored in the same run at the same settings.
2. *Across time*: the gate reads `sft_gate_temperature` / `sft_gate_top_k` /
   `sft_gate_new_tokens` — **its own** config keys, not the shared `coherence_*` decode
   defaults — and records the values used in its result file. If Phase 6 changes the global
   decode default, a re-run of the SFT gate still recomputes at the pair it was
   pre-registered against.

**Rejected.** *Aliasing the gate to the global decode default* — the tidier option, and it
silently couples a pre-registered measurement to a value a later phase has every reason to
change. The duplication is the point: two keys that happen to hold the same number today
but are allowed to diverge, one of which is frozen by pre-registration.

**Consequence.** `sft_gate_*` duplicates `coherence_*` at 0.8 / 40 today. Anyone changing
one must decide about the other, which is the intended friction.

---

## ADR-026 — Separate chance, echo, and instruction-following in the base floor

**Context.** The Phase 4 threshold was anchored to bigram: an **adversarial** floor, an
alternative model computed independently of the thing under test. Phase 5's anchor is
`base.pt`'s own adherence — a **self-baseline**, one fine-tune step from the model being
judged. Still the right anchor, because the real question is "did SFT buy adherence over
base prompted identically". But it does not come with bigram's independence, so the floor
needs decomposing before a delta on top of it means anything.

The specific hazard: the prompt contains the subject word. A model that merely continues its
context will repeat it. That is ordinary conditioning, not instruction-following, and a raw
subject-mention rate cannot tell the two apart.

**Decision.** Report a **shuffled-subject control** alongside every subject-mention rate:
each response re-scored against a different response's subject, via a derangement so nobody
keeps their own. This splits the floor three ways:

- **chance** — the shuffled rate. How often a story mentions some unrelated common noun
  anyway. Measured on real corpus responses: **5.5%**.
- **echo** — matched minus shuffled, for a model that was never instructed.
- **instruction** — what SFT must buy on top of both.

**Rejected.**
- *Reading the raw base rate as the floor* — it blends chance and echo, so a delta over it
  understates or overstates depending on which dominates, with no way to tell which.
- *Treating any high base subject-mention as a checker leak* — the sharper reading. A true
  leak is the checker finding the subject in text the model was handed; that is already
  prevented by scoring the continuation only, with the prompt stripped. Prompt echo is a
  real property of language models, not a defect, and the shuffled control is what
  distinguishes them. High matched *and* high shuffled would be the checker matching
  everything; high matched with low shuffled is echo plus whatever else is there.

**Consequence.** The floor is reported as three numbers rather than one, and the
pre-registered delta is stated against the matched base rate with the chance rate visible
beside it. The corpus ceiling is measured the same way: 100% matched against 5.5% shuffled,
confirming subject-mention is not trivially satisfiable.

**Measured verdict on the leak.** `base.pt` scores 35.0% matched against 6.5% shuffled.
The shuffled rate is essentially the corpus chance rate (5.5%), so the checker is not
matching everything — it is not broken. The 28.5 points above chance are prompt echo. The
delta SFT must buy sits on top of that, not on top of zero.

---

## ADR-027 — The Phase 5 gate, pre-registered before the training code existed

**Context.** ADR-017 records the Phase 4 wrinkle honestly: the perplexity bar was agreed
minutes *after* the run launched. Nothing from the run informed it, but "pre-registered"
had to be qualified. Phase 5 is the chance to do it properly.

**Decision.** Thresholds agreed and committed **before `src/sft.py` existed and before any
fine-tune ran**. The git history is the evidence: this config block lands in its own commit,
ahead of the commit that adds the training code.

| sub-score | base floor | **bar** | delta | z | role |
|---|---|---|---|---|---|
| `subject_mention` | 35.0% | **≥ 60%** | +25.0 pts | 5.2 | delta |
| `length_band` | 44.5% | **≥ 70%** | +25.5 pts | 5.1 | delta |
| `is_story` | 69.0% | ≥ 69.0% | must not regress | — | floor |
| `not_degenerate` | 74.5% | ≥ 74.5% | must not regress | — | floor |
| shuffled control | 6.5% | ≤ 10% | — | — | validity |

The detection floor at n=200 is ~9.3 points, so both deltas are ~2.7x noise: large enough
to be decisive, small enough that a 14M model could genuinely miss them.

**Why the length bar is set higher than the subject bar.** `base.pt` writes short — 95.6
words against a 102–200 band and a corpus mean of 157.8 — so much of its 44.5% failure is
systematic under-length that training on full-length responses should fix almost
mechanically. The easier metric gets the higher bar; otherwise the gate rewards the fine-tune
for the thing it was always going to do.

**Rejected.**
- *One blended adherence threshold* — ADR-024.
- *Setting the bars after seeing SFT's first result* — the Phase 4 wrinkle, uncorrected.
- *A bar at the ceiling* — 100% subject-mention is what the corpus scores by construction,
  not something a 14M model owes.

**Consequence.** The gate can fail, and the shuffled control can invalidate a pass: if
subject-mention rises while the shuffled rate rises with it, the model has learned to name
more nouns rather than the right one, and the headline number stops meaning adherence.

## ADR-028 — The token budget censors the training data, and the generation cap was set wrong

**Status.** Accepted, Phase 5, before `src/sft.py` existed.

**Context.** `context_len` is 256 tokens. Instruction pairs allow responses up to 220
words, which at the measured 1.29 tokens/word does not fit. Pairs that overflow are
dropped rather than truncated: cutting a story mid-sentence would teach the model to stop
mid-sentence, which is the exact pathology SFT exists to remove.

But the filter does not *thin* the length distribution, it **censors** it. Length is what
breaks the budget, so the longest responses are removed preferentially and the right tail
is cut off rather than sampled less often. If that cut landed below the upper edge of the
pre-registered `length_band`, the training set could not contain examples reaching the top
of the band, and the model could not produce what it was never shown. A bar the data is
structurally incapable of teaching is not a bar the model can miss.

The censoring point is not a fixed word count. Prompt length varies (10-11 tokens), so the
headroom left for the response varies with it: the same response fits behind a short
prompt and does not behind a long one. The check therefore has to be the **joint**
distribution of response-words against tokens-consumed, not a single threshold.

**Measurement.** `python -m src.sft_data census`, on the 40,000 pairs, encoded by the same
code that builds the training tensors — so the census describes the data actually trained
on rather than a second implementation of the filter free to disagree with the real one.

Dropped 2,192 of 40,000 (5.5%), every one for the same reason. Survival is 100% up to 159
words, 99.9% at 160-169, 98.6% at 170-179, 93.5% at 180-189, 73.2% at 190-199, and 0 above
219. The registered band is 102-200; the surviving band (p5-p95) is 102-189, and 1,707
surviving examples sit at 190+ words with the longest at 217. **The band is spanned.** The
5.5% drop is not a structural force on the distribution inside the band, and the
pre-registered length bar is reachable from the data.

**But the same census exposed a worse problem in the gate's own decoding.** The gate
generated at most 200 new tokens, which buys ~155 words. Two consequences, both measured
on base:

* The band's upper edge of 200 words was **unreachable**. All 111 of base's `length_band`
  failures were too-short; zero were too-long. The band was silently one-sided, while
  being described as a two-sided corpus band.
* `is_story` had quietly become a **truncation detector**. It requires the response to end
  on terminal punctuation, and a response cut off at the cap does not. Base fails
  `is_story` in **66%** of its longest quartile versus **57%** of its shortest — the wrong
  way round for a story-quality metric. 35 of 200 responses (17.5%) ended without terminal
  punctuation.

SFT's job on `length_band` is to make responses longer. Longer responses hit the cap.
`is_story` is a floor that must not regress. **The two halves of the gate were in
mechanical conflict, and the conflict was caused by the cap rather than by the model** — a
regression would have read as "the model stopped producing stories" when it actually meant
"the model wrote longer stories and got cut off."

**Decision.** Raise `sft_gate_new_tokens` from 200 to **245**, and make the *delta* the
pre-registered quantity rather than the absolute bar.

245 is not a round number, it is the largest value that is safe. `generate()` feeds
`out[-context_len:]`, so the instruction stays inside the attention window only while
`prompt + generated <= 256`. The longest held-out prompt is 11 tokens. Above 245 the model
is being asked to keep writing about a subject it can no longer see, and `subject_mention`
would decay for a reason that has nothing to do with the fine-tune. At 245 the band's
upper edge becomes reachable (201 words at the cheapest encoding) and the surviving
training p95 of 189 words fits with 1 token to spare.

**Why this is not moving the goalposts.** Raising the cap does make `is_story` easier in
isolation — fewer truncations. So the floors are **re-measured on base at the new cap and
re-anchored to what that measurement says.** The bar moves with the floor. Keeping
`is_story` at 0.69 while raising the cap would have been the one change that genuinely made
the gate easier. The two deltas (+25.0 and +25.5 points) do not move; they were agreed
before any of this and they are now stored in the config as
`sft_gate_*_delta`, with the absolute bars derived as `base + delta` so they cannot drift
from the rule.

**Consequence.** Two pre-flight validity checks now guard the Phase 5 gate, and they are
the same pattern applied to different halves of it. In `src/checker.py`, **base's own
score** says whether the checker is honest — a high base `subject_mention` would mean a
broken checker, not a weak floor. In `src/sft_data.py`, **the surviving data distribution**
says whether the bar is reachable. Neither is a check on the model; both are checks on the
gate, and both have to pass before the fine-tune is a meaningful test rather than a rigged
one.

`coherence_new_tokens` stays at 200. Phase 4's artifacts were published at that value and
retro-fitting a closed phase's decoding to make its numbers look better is the opposite of
the discipline here. The Phase 5 gate reads its own `sft_gate_*` keys precisely so the two
can differ without either being silently redefined (ADR-025).
