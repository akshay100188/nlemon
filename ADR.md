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

**Scope, added retroactively in Phase 5 (ADR-031).** This gate generates from
`GALLERY_PROMPTS`, which are all *continuations* — the shape a pretrained model was trained
on. The green is therefore a claim about coherence **under continuation prompts**, and the
text above did not say so. It should have: scored on *instruction* prompts, the same
`base.pt` leaves the `oov_rate` band at 0.006097 against a 0.002114 ceiling, emitting
mangled instruction verbs (`tellow`, `mrite`, `telle`). Nothing measured here was wrong; the
claim was broader than the measurement, which is the same defect as an unexplained remainder
and gets the same treatment — narrow the claim where it was published rather than leave a
reader to discover the boundary. A coherence gate that wants to be prompt-distribution
independent has to sample prompts from more than one distribution, and this one does not.

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

## ADR-029 — A floor set at the measured base is a coin flip, not a bar

**Status.** Accepted, Phase 5, before `sft.pt` was scored.

**Context.** `is_story` and `not_degenerate` are floors: they carry no part of the
instruction-following claim and exist only to fail a fine-tune that buys adherence by
burning fluency. The rule agreed for them was "must not regress below base", and the
literal implementation is a bar at base exactly — `is_story >= 0.770`,
`not_degenerate >= 0.735`.

That reads stricter than it is. The comparison is paired over 200 held-out prompts, and
its one-sided 95% detection floor is

    z * sqrt(2 * p * (1 - p) / n),  z = 1.645, n = 200

which is 6.9 points on `is_story` and 7.3 on `not_degenerate`. A fine-tune that is
behaviourally *neutral* on these metrics still flips individual prompts in both
directions, so its net change is centred on zero with a spread of several points — and it
lands below a bar set at base **about half the time**.

That is the same defect that got a 300-step overfit gate rejected in Phase 3: at 300 steps
the worst of three seeds landed at 0.42 against a 0.1 target and "the gate becomes a coin
flip". A gate that fails half the time when nothing is wrong does not measure the model, it
measures the sampling noise, and its verdict carries no information either way.

**Decision.** Set each floor at **base minus the detection floor**:

    is_story        0.770 - 0.069 = 0.701
    not_degenerate  0.735 - 0.073 = 0.662

A breach now means a regression the data can actually distinguish from noise. The SE is
computed at the unpaired upper bound, which is larger than the true paired SE, so every
number derived from it is conservative in the direction that makes the gate harder rather
than easier.

The alternative considered was keeping the bar at base and reporting significance
alongside a breach, as a two-part verdict. Rejected as the worse of two honest options: it
would report `PHASE 5 RED` on outcomes that are statistically indistinguishable from no
change, and a red that has to be read with a footnote explaining it is not really red is
the kind of thing that gets quoted without the footnote.

**Consequence.** All four bars are now derived from a rule rather than typed in, and
`src/checker.py gate` **recomputes every one of them from base's recorded scores and
asserts the result matches the frozen config value**, refusing to run if they disagree.
Deltas come from `base + sft_gate_*_delta`, floors from `base - z * SE`. A gate that reads
its threshold off a summary can be handed a wrong summary; a gate that recomputes it and
cross-checks cannot. This is the same reason Phase 4's `src/verify_docs.py` recomputes
every published ratio from `perplexity_floors.json` instead of trusting the prose
(ADR-021).

Note what this does *not* loosen. The floors moved down by ~7 points from base, but base
itself moved *up* by 8 points on `is_story` when the generation cap was corrected
(ADR-028), so the floor now sits at 0.701 against an originally-registered 0.690 — still
marginally stricter than the bar this phase started with, and with the coin-flip removed.

## ADR-030 — Appearing in a document is not being about it

**Status.** Accepted, Phase 5, **after** the gate came back RED. Recorded as a diagnosis of
a validity check that was owed and not run, not as grounds for re-reading the result.

**Context.** Phase 5 came back RED: `subject_mention` reached 50.0% against a pre-registered
bar of 60.0%. The other three sub-scores passed comfortably — `length_band` 43.5% to 95.5%,
`is_story` 77.0% to 98.5%, `not_degenerate` 73.5% to 85.5% — and the shuffled control stayed
inside its limit, so the failure is specific to subject adherence rather than general.

Digging into the 100 misses: 12 of the 50 held-out subjects were *never once* mentioned,
across any of the 200 responses. The list is the finding — `adventure`, `edge`, `floor`,
`leaves`, `middle`, `people`, `prize`, `screen`, `sound`, `stage`, `table`, `lizard` —
against a perfectly-scoring set of `bunny`, `frog`, `duck`, `prince`, `carrot`, `candy`,
`basket`.

**`lizard` breaks the tidy version of that story, and the exception is informative.** Eleven
of the twelve are abstract, spatial, or mass nouns that cannot be a protagonist. `lizard` is
a concrete animal that should behave like `bunny` and `frog` and does not, so aboutness
structure cannot be the whole explanation — the remaining cause is **corpus frequency**.
Lizards are rare in TinyStories, so the pairs were thin and the binding was learned weakly.
That means the retry's aboutness filter will drop `lizard` for a *different reason* than it
drops `moral`, and the re-derived pool therefore loses subjects to two causes at once:
structural unaboutability and plain rarity. Both must be reported separately, because a pool
that shrinks hard widens the confidence interval on the next delta.

**The cause.** `build_pairs` requires the subject to *appear* in the document. Appearing is
not aboutness. "Once upon a time a girl dropped her toy on the floor" contains "a floor" as
the head of a determiner phrase, so the miner accepts it, and the pair then teaches
`Write a story about a floor.` to mean a story that says "floor" once in passing. The model
learned exactly that, faithfully, and the checker then marks it wrong.

Measured on the 40,000 training pairs (`python -m src.instruct aboutness`):

    subject appears  0x        23    0.1%
                     1x    19,096   47.7%
                     2x     7,581   19.0%
                     3x     4,221   10.6%
                     4x     2,923    7.3%
                    5+x     6,156   15.4%

**47.8% of training pairs mention their subject at most once.** Weakest subjects by mean
occurrences: `moral` 1.00, `moment` 1.00, `sudden` 1.00, `distance` 1.05, `while` 1.07.
Strongest: `crane` 7.05, `swan` 7.04, `dove` 6.95, `goat` 6.72, `kitten` 6.18. The split is
concrete story protagonists against abstract and relational nouns.

The headedness filter cannot catch this, and it is worth being precise about why: in
"a floor", `floor` **is** the head. The filter was built to reject modifiers — "a brave bird"
promoting `brave` — and it does that correctly. Aboutness is an orthogonal property it was
never measuring. Nor would a stopword list have worked; the residue includes `while`
("after a while") and `sudden` ("all of a sudden"), where an idiom makes an abstract noun
the legitimate head of its own phrase.

**What this does not do.** It does not soften the RED. The bar was base plus 25.0 points,
anchored to base measured on **the same 200 prompts**, including the unanswerable ones. Both
sides paid the same penalty, so the comparison is apples-to-apples and the delta is the
honest quantity.

**And the "clean subset" number is weaker than first reported, because the subset is
outcome-selected.** On the 160 prompts whose subject was mentioned by at least one model,
base scores 38.8% and sft 62.5% — a delta of +23.8 against a registered +25.0. That still
misses, so the direction of the conclusion is unchanged, but the *criterion* is circular and
must be labelled: "answerable" was defined as **mentioned at least once by the run's own
generations**. `sft` cannot mention an unanswerable subject — had it done so, that subject
would have been classified answerable — so `sft`'s 0-for-40 on the excluded set is
definitional rather than measured, and nothing in that subset can falsify anything. The
subset was selected using the very outcome it is being used to diagnose.

The model-independent criterion is **corpus aboutness**: does the corpus contain enough
documents that are genuinely about this subject? That is a property of the data alone,
measurable before any model runs, and it is exactly the filter the retry uses at pair
construction. So the metric that gates pair construction *is* the answerability definition
this diagnosis needed, and the defensible clean delta falls out of the retry's own pool
derivation rather than being carved after the fact. Reported here as **+23.8 on an
outcome-selected subset**, which is worth less than it looks and is expected to fall once the
criterion is model-independent — a direction that firms the RED up rather than softening it.

Reporting the 160-prompt figure as *the result* would be the laundering move this project has
already named once, in the length case: a bar the data was structurally incapable of
clearing, pre-registered anyway, then re-read after the fact against a subset chosen because
it was kinder. The subset number is a diagnosis of *why*, and it stays labelled as one.

**Consequence.** `python -m src.instruct aboutness` is now a first-class command, and it is
the check that should have run before the `subject_mention` bar was registered. Phase 5 had
two pre-flight validity checks on the gate — base's own score for whether the *checker* was
honest (ADR-026), and the surviving length distribution for whether the *length* bar was
reachable (ADR-028) — and both passed. The third, whether the *subject* bar was answerable,
was never written. The pattern was correct and its application was incomplete: a delta bar
needs a ceiling measurement for **each** sub-score it covers, not for whichever one prompted
the thought.

The remedy for a Phase 5 retry is to require aboutness at pair-construction time — the
subject must occur at least twice, or the earliest-determiner heuristic must be replaced by
"most-repeated candidate" — and then to re-derive both the subject pool and the base floor
before re-registering. That is a data change, so it moves the `sft` stage hash, and it is a
new pre-registration rather than an adjustment to this one.

## ADR-031 — The oov_plausibility band's first live firing was on base, not sft

**Status.** Accepted, Phase 5.

**Context.** `oov_plausibility` was built in Phase 4 to separate a *plausible* invented word
from *garbage*, using a character-trigram model fitted to the known-word set. It then never
fired: the Phase 4 gate scored 16 continuations of the gallery prompts and found zero
out-of-vocabulary words, so the band went through an entire phase validated only against
fixtures. Its first real test was owed.

**Measurement.** `python -m src.checker oov` scores the 200 held-out instruction responses
from each stage — roughly 20k words from base and 28k from sft — which is the condition the
band was built for, unlike the gallery.

    base   200 responses, 20,173 words
      oov_rate          0.006097   band [0.000000, 0.002114]   OUT OF BAND
      oov_plausibility  -2.6651    band [-2.7517, -1.8522]     ok
      34 distinct OOV words

    sft    200 responses, 28,418 words
      oov_rate          0.000106   band [0.000000, 0.002114]   ok
      oov_plausibility  -3.0792    band [-2.7517, -1.8522]     out, but n=3
      3 distinct OOV words

**The class matters more than the cell lighting up,** and base's class is unmistakable:
`tellow` x24, `mrite` x21, `telle` x9, `arite` x7, plus `briite`, `micrite`, `brite`,
`crite`, `tellia`, `thellow`, `tellell`. These are mangled forms of *the instruction verbs*.
Base, handed "Write a story about..." or "Tell me a story about...", cannot parse it as an
instruction and emits corrupted echoes of the opening verb.

The two metrics say different true things and the pair is what makes the read work.
Plausibility at -2.6651 sits *inside* the band: these are word-shaped inventions, not noise,
which is correct — `tellow` and `brite` are pronounceable English-looking strings. The rate
at 3x the corpus ceiling says there are far too many of them. Either metric alone would have
reported half the story; `oov_rate` alone would have said "degenerate" without saying the
words were well-formed, and `oov_plausibility` alone would have said "fine".

**This also revises what Phase 4's coherence pass established.** That pass was measured on
continuation prompts, and base is inside every band there. Under *instruction* prompts base
goes out of band on `oov_rate`. The Phase 4 green was therefore conditional on the prompt
distribution in a way that was not visible at the time — instruction prompts are
out-of-distribution for a purely pretrained model, and that shows up as invented words.

**On sft.** `oov_rate` fell 57x and returned inside the band, on 3 words: `sosophie`,
`intooh`, `budy` — all collisions of real words rather than inventions. So the feared failure
did not happen: SFT did not buy adherence by breaking coherence. Every coherence-side metric
moved the same way (`is_story` 77.0 to 98.5, `not_degenerate` 73.5 to 85.5, `oov_rate`
out-of-band to in-band), which is the opposite of the trade-off the floors were watching for.

**Decision.** Report sft's `oov_plausibility` of -3.0792 as **too thin to read**, not as an
out-of-band finding. A pooled mean over 3 words cannot be compared to a band bootstrapped
from corpus groups holding orders of magnitude more (ADR-022); the number is real but carries
no evidence. `cmd_oov` prints the caveat inline rather than leaving a reader to infer it,
because "OUT OF BAND" next to a number is exactly the kind of thing that gets quoted without
its sample size.

## ADR-032 — Phase 5 attempt #2: the RED stands, and this is a second attempt beside it

**Status.** Accepted, **before** any retry code was written or any retry data derived.

**The framing, first, because it is the thing that keeps a retry from becoming a rescue.**

Phase 5 is RED. That is recorded at `3d8ba40` through `31f0789`, it is real on its own terms,
and it is real on the outcome-selected clean subset too (+23.8 against a registered +25.0).
**Nothing in this ADR converts it to green.** A retry does not un-record a failure; it adds a
second attempt beside the first. The permanent story of this phase is *"Phase 5 REDed, here
is why, here is attempt two"* — never *"Phase 5 passed on the second go."* Any future
write-up, README table, or scorecard that reports a green Phase 5 without reporting the RED
that preceded it is misreporting this project.

**Why the retry happens, and it is not the number.** Phase 6 fine-tunes DPO *on top of*
`sft.pt`. A checkpoint that does not reliably track what it was asked about is the foundation
preference learning would inherit, and the defect would propagate into a phase whose whole
premise is "the model already follows instructions; now teach it which answers are better."
The retry buys a sound foundation for the next phase, not a better scalar in this one.

**The test that separates a fair re-registration from bar-moving.** *Would this change be
made if the gate had passed?* Yes, unambiguously. 47.8% of the training pairs teach their
subject at one mention or fewer, so nearly half the supervision never demonstrated aboutness
at all. That is a broken teaching instrument regardless of which way the gate went, and
fixing it is validity work — the same family as fixing the leak detector or the generation
cap, both of which were also found by looking rather than prompted by a failure.

Contrast the move that was correctly refused: **rescoring the existing `sft.pt` against a
friendlier subject pool.** That change would only ever be made *because* the gate failed, it
retrains nothing, and it does not even work — `sft.pt` misses +25 on the clean subset anyway.
The distinction is not rhetorical, and it is checkable in the repo: attempt #2 changes the
**data** and **retrains**, which moves the `sft` stage hash. A rescore would have left every
hash untouched. The stage hash is what proves which act occurred.

**Three constraints, pre-committed.**

1. **The subject pool is re-derived by corpus aboutness, not by what a model happened to
   mention.** A model-derived criterion would be circular, which is the flaw ADR-030 records
   in its own +23.8 figure. Corpus aboutness is a property of the data, measurable with no
   model in the room.

2. **The base floor is re-derived from scratch and the delta is re-argued from scratch.
   +25.0 is not carried over.** It was calibrated against a pool that is about to change
   composition: filtering to concrete, answerable subjects should *raise* base's floor,
   because prompts like "Write a story about a bunny." are ones even a pretrained model
   stumbles into answering. `base + 25` measured off a higher floor is a different and harder
   claim, and pretending it is the same commitment would be the pre-registration equivalent
   of moving a goalpost while claiming not to have touched it. The new delta and the amber
   band are argued from the new pool's size and the new floor, before the run.

3. **One retry.** If attempt #2 REDs again on clean aboutness data, that is **not** grounds
   for a third filter. It is the capacity finding — a 14M-parameter model cannot hold subject
   adherence at the registered level while staying coherent — and the honest response is to
   report the measured ceiling and let Phase 6 proceed on the best `sft.pt` available with
   that ceiling documented. A retry sequence that runs until something turns green is the
   laundering move wearing a lab coat, and it is pre-committed against here so that the
   commitment exists before the result does.

**Consequence.** Every Phase 5 artifact carries an attempt number from here on, and the
scorecard reports both attempts. The `subject_mention` bar for attempt #2 is registered in a
commit of its own, ahead of the code that derives the new pairs, in the same shape as
`3d8ba40`.

## ADR-033 — 312 prompts are not 312 observations, and attempt #1's floor was understated

**Status.** Accepted, Phase 5 attempt #2, before the attempt #2 fine-tune.

**Context.** Every threshold in this phase is priced off a detection floor — the smallest
difference the eval can tell apart from noise. Attempt #1 computed that floor as
`1.96 * sqrt(2p(1-p)/n)` with `n = 200`, the number of held-out prompts, and reported
**9.3 points**. On that basis the +25.0 delta was described as **2.7x noise**.

The 200 prompts were not 200 independent observations. They clustered into 50 subjects, and
outcomes correlate hard inside a subject: a model that can talk about `bunny` gets every
bunny prompt right, and one that cannot get any of them right. Attempt #1's own data shows
it starkly — 20 subjects scored 100% and 12 scored 0%.

**Measurement.** A one-way random-effects ANOVA over subjects gives the intra-class
correlation, and the design effect follows as `deff = 1 + (m_bar - 1) * ICC`:

| | ICC | deff | n_eff (of 200) |
|---|---|---|---|
| `subject_mention` (base) | 0.298 | 1.89 | 105.7 |
| `subject_mention` (sft) | 0.333 | 2.00 | 100.0 |
| `is_story` | 0.120 | 1.36 | 147.1 |
| `not_degenerate` | 0.016 | 1.05 | 191.0 |
| `length_band` | 0.000 | 1.00 | 200.0 |

`subject_mention` — the one sub-score carrying the phase's claim — is also the one that
clusters hardest, which is not a coincidence: subject adherence is *a property of the
subject*, so of course it correlates within one.

**So the attempt #1 record needs correcting in two places, neither of which is cosmetic.**

The real two-sided detection floor was **12.9 points, not 9.3**. The +25.0 delta was
**1.9x noise, not 2.7x**. And `sft.pt`'s 10-point shortfall against the 60.0% bar sat
*inside* that floor — the RED still holds, because a bar is a bar and a one-sided test of
`sft >= 0.60` against an observed 50.0% at `n_eff = 100` gives p = 0.023, but it holds by
less margin than the write-up implied. "Missed the bar" is correct; "missed it comfortably"
would not have been.

The other correction is that `length_band`'s ICC is essentially zero. Response length has
nothing to do with which subject was asked for, so its prompts really were independent and
its floor really was ~8 points. Reporting one detection floor for all four sub-scores was
wrong in both directions at once, and it is the same error ADR-024 already named in a
different costume: **sub-scores that move for different reasons need their own statistics,
not a shared one.**

**Decision, in three parts.**

1. **Price every bar on `n_eff`, recomputed per sub-score.** `src/checker.py effective_n`
   derives ICC and the design effect from the per-prompt rows at gate time, so the floor
   tracks whichever eval set is in use rather than a number typed in once.

2. **Sample the held-out set balanced across subjects.** Attempt #1 built it by scanning
   until it had 200 pairs, which takes the *first* 200 matching documents and therefore
   over-samples common subjects: one subject carried 34 of the 200 prompts, 17% of the eval
   riding on a single word. Attempt #2 takes 4 prompts from each of all 78 held-out
   subjects — 312 prompts, largest cluster 1.3%.

   **Capping prompts-per-subject was considered and rejected by measurement, not taste.**
   It is the obvious fix and it is the wrong one: it lowers `m_bar` but lowers `n` faster.
   Priced against the measured ICC of 0.33, a 3-per-subject cap yields `n_eff = 84`, *worse*
   than the 106 it was meant to repair. More subjects is the lever; fewer prompts each is
   not. With `k` subjects and ICC `r`, `n_eff` is bounded above by `k/r` no matter how many
   prompts are added — 164 at attempt #1's 50 subjects, 236 at attempt #2's 78.

3. **Register an AMBER band.** `GREEN` at or above the bar, `RED` below `bar - 1.96*SE`, and
   `AMBER` in between: missed, but by less than this eval can resolve. Attempt #1 had no
   amber, which is precisely why its 10-point shortfall had to be *argued about after the
   fact* rather than classified by a rule agreed in advance. AMBER is not a pass, and per
   ADR-032 it does not license a third attempt — it is where the capacity finding lives.

**Consequence, and the part worth carrying past this project.** Balanced sampling did more
than raise `n_eff` arithmetically. ICC fell from **0.33 to 0.083**, mostly because the
aboutness-cleaned pool no longer contains subjects that are structurally 0% or 100%. The two
fixes compound: removing unanswerable subjects removes the between-subject variance that was
making the clustering expensive in the first place. `n_eff` went from ~106 to ~250 on a
prompt count that rose only from 200 to 312, and the detection floor fell from 12.9 points
to 8.8.

A gate is only as sharp as the number of independent observations behind it, and "how many
prompts did you run" is not that number.

## ADR-034 — Verify a correction reached the *registration*, not only the recount

**Status.** Accepted, Phase 5 attempt #2, before the attempt #2 verdict was read.

**Context.** Two of the three defects found while building attempt #2 were corrections to a
*measurement*. Both raised the same follow-up question, and it is not the same question as
"was the correction right":

> When you fix how something is measured, did the fix reach the **pre-registered bar**, or
> only the **retrospective analysis**?

A correction that lands in the recount and not the registration leaves the verdict zones
calibrated to the number you no longer believe. The result then gets scored against a stale
bar while the write-up quotes the corrected figure, and the two disagree silently. This is
cheap to check and it has now come up twice in one phase, so it is written down as a
standing step rather than a thing to remember.

**The two checks, and how they were answered.**

*Did attempt #2's `subject_mention` bar come from attempt #2's own floor?* Answered by
deriving the floor a **second, independent way** rather than re-reading the first. A cluster
bootstrap over subjects — 4,000 resamples of whole subjects, making no ANOVA or normality
assumption — gives 8.8 points against the design-effect method's 8.8, agreeing to 0.04. The
registered +17.5 is 2.00x and 1.99x against the two. And the inheritance test is arithmetic:
carrying "same shape as +25" would have registered a 72.8% bar, carrying attempt #1's
claimed 2.7x multiple would have registered 71.4%, and the actual registered bar is 65.3%.
It could not have inherited from either. **Clean.**

*Did `length_band`'s upper edge come from what the generator can emit?* Partly, and the
honest answer is that the right ceiling was reached by the wrong route. The band was derived
from the **training filter** (`prompt + response <= 257`, so response `<= 246` tokens), not
from the **generation cap** (response `<= 245`). Those are different constraints. They differ
by one token here, and not coincidentally: ADR-028 set the cap to `context_len - max_prompt`
and the training budget is `context_len + 1 - max_prompt`, so the two are structurally
coupled one token apart. A cap set *below* `context_len - max_prompt` would break that
coupling and the band would then need deriving generation-side. **Right answer, route not
verified in advance** — recorded so the coupling is a stated assumption rather than a
coincidence nobody noticed.

**A second lesson, on trusting a ratio over a measurement.** The tokens-per-word arithmetic
predicted that 190 words needs ~245 tokens at the median encoding and would therefore be
barely emittable, making the band's upper edge nearly inert. That prediction was wrong.
Base emitted **18 responses over 190 words out of 312**, with a p95 of 193 and a maximum of
217 — the edge binds. Generated text at that length encodes more cheaply than the
corpus-wide median tokens-per-word implies, so a corpus-average ratio is the wrong instrument
for a question about the tail. The empirical count overrides the ratio, and the ratio should
not have been reported as a finding before the count was available.

**This is the Phase 2 tokenizer-sweep error in different clothes, and the two belong
together in any write-up.** There, compression ratio *plateaued* at 25,000 training
documents — judged on that one averaged metric, 25k was indistinguishable from 800k — while
vocabulary overlap kept moving, so the plateau was a property of the summary statistic
rather than of the tokenizer. Here, a corpus-average ratio said the tail was unreachable
while the tail itself was demonstrably being reached. Same defect twice: **an averaged
statistic answering a question about the distribution's edge.** A mean is the wrong
instrument for a tail question, and it fails in the direction that looks reasonable, which is
why neither instance announced itself.

**Consequence.** Every future gate adds one step between "measurement corrected" and "run
launched": re-derive the bar from the corrected measurement and check it against the frozen
value, by a method that does not share an input with the original derivation. `derive_bars`
already asserts the frozen bars match the rule; what this adds is that the *rule's inputs*
get recomputed from the current data, and that at least one number in the chain is derived
twice by genuinely independent means (ADR-013's rule, applied to thresholds rather than to
parameter counts).


## ADR-035 — The amber band was registered on one side of the bar only

**Status.** Accepted, Phase 5 attempt #2, recorded with the verdict rather than after it.

**Context.** Attempt #2 came back GREEN on all four sub-scores. Three of them cleared their
bars by 25 points or more. `subject_mention` cleared by **5.6**, against a detection floor of
**8.8**.

The amber band registered before the run covers `bar - 1.96*SE <= sft < bar`: a *miss* too
small for the eval to resolve. There is an obvious symmetric case — a *pass* too small for
the eval to resolve — and it was not registered. The Product Owner named this failure mode
in advance, in the same message that asked whether the corrections had propagated into the
bars: *"a GREEN could be a floor-grazing result wearing a passing label."* The propagation
checks came back clean and the zones were confirmed sound, and then the result landed in the
un-registered zone anyway.

**Measurement.** Two questions, deliberately separated, because they have different answers:

*Is the delta real?* `subject_mention` moved 47.8% to 70.8%, **+23.1 points, 2.6x the 8.8
point floor**. Decisive. The claim "the aboutness fix taught subject adherence" is solid.

*Is the margin over the bar real?* 70.8% against a 65.3% bar is **+5.6 points, inside the
same floor**. A one-sided test gives z = 1.74, p = 0.041 — significant at 95%, barely. A
cluster bootstrap over subjects puts **3.5% of resamples below the bar** and its 2.5th
percentile at **64.4%, which is below the bar itself**. Not decisive.

A further wrinkle, and it cuts the same way. The bars were priced on **base's** clustering,
which is the correct pre-registration choice — base is measurable before the run, sft is not,
and pricing a bar on the result's own variance would make the threshold depend on the
outcome. But sft's clustering is worse: ICC 0.182 against base's 0.083, so sft's effective n
is 202 rather than 250 and its true floor is nearer 9.8 points. The margin sits further
inside the noise than the registered floor suggests, not less far.

The other three are decisive by any reading: `length_band` +25.1 over its bar (z = 16.7),
`is_story` +28.1 (z = 36.2), `not_degenerate` +25.5 (z = 15.1), with 0.0% of bootstrap
resamples below the bar in each case.

**Decision.** Report the verdict as **GREEN, with `subject_mention` flagged as a grazing
pass** — and do not quietly upgrade it in the write-up. The pre-registered rule says GREEN
and the rule stands; changing it now, after seeing where the number fell, would be the
mirror image of the laundering this phase has twice refused. What is owed is the caveat, in
the same place as the headline: *the delta is decisive, the margin over the bar is not, and a
rerun on a different seed could plausibly land in AMBER.*

**Consequence.** The amber band becomes two-sided for every future gate:

    GREEN    sft >= bar + z*SE     cleared by more than the eval can resolve
    AMBER+   bar <= sft < bar + z*SE     cleared, but inside the noise
    AMBER-   bar - z*SE <= sft < bar     missed, but inside the noise
    RED      sft < bar - z*SE

A one-sided amber encodes an assumption nobody stated: that only a near-miss is ambiguous,
while a near-hit is a clean pass. Both are the same distance from the same threshold and both
deserve the same label. Registering only the lower half meant the gate could produce an
unqualified GREEN from a result it could not distinguish from a failure — which is precisely
the property the amber band was invented to remove.

Note what this does **not** change about attempt #2: three sub-scores clear decisively, the
shuffled control held (0.6% to 1.9% against a 6.0% limit, while matched rose 23.1 points),
and the `subject_mention` *delta* is 2.6x noise. The finding is about how confidently one
number may be quoted, not about whether the fine-tune worked.

## ADR-036 — The shell is a hazard class, not three coincidences

**Status.** Accepted, end of Phase 5. Recorded as one entry replacing three incident notes.

**Context.** Three separate times this project has been damaged or misled by the *toolchain
around* the model rather than by the model, and all three were PowerShell idioms behaving
exactly as documented:

1. **`Get-Content | Set-Content -Encoding utf8`** re-encoded tracked source files and
   corrupted them — twice, the second time on `README.md` during the Phase 4 audit, and then
   compounded by running `ftfy` over the docs, which "repaired" the intentional mojibake
   examples in ADR-009. Fixed by restoring from `HEAD` and editing through tooling that does
   not round-trip encodings; guarded permanently by `scripts/check_encoding.py` with pinned
   golden byte sequences.

2. **Here-strings and inline quoting** silently mangled commit messages and produced
   `SyntaxError` in inline Python often enough that the standing rule became: write a
   scratchpad file, never inline a multi-line string.

3. **`Select-Object -First 12`** terminated the pipeline once it had twelve objects, killing
   the console attached to the attempt #2 fine-tune and reporting **exit 255**. The training
   itself ran to completion — all 1,722 steps, confirmed by `sft_train_summary.json` and a
   106-row curve — but the log stops at exactly twelve lines and the harness recorded a
   failure. A **false red**, produced entirely by the observing command.

**Decision.** Record these as one hazard class rather than three incidents, because the
lesson is singular and the count is the evidence: **the instrumentation around a measurement
can lie in the same ways the measurement can, and it gets far less scrutiny because nobody
thinks of `Select-Object` as part of the experiment.**

Each of the three has the same shape as a defect this project already guards against in the
model pipeline. Encoding corruption is an unverified transform (ADR-009). A mangled commit
message is an artifact that does not match what produced it (ADR-010). And exit 255 on a
successful run is **a green check lying in the other direction** — the mirror of ADR-021,
where `verify_docs` printed success while skipping real claims. Here the tool printed failure
while the work succeeded. A status that disagrees with the artifact is untrustworthy in both
directions, and the response is the same either way: check the artifact, not the status.

**Consequence.** Three standing rules, all already in force:

* Never pipe a long-running process through a pipeline-terminating cmdlet. Redirect to a
  file and read the file.
* Never inline multi-line strings into a shell. Write the file, pass the path.
* Never let a tool re-encode a tracked file; `scripts/check_encoding.py` runs pre-commit.

And one line for the write-up, which is the point of consolidating: *the toolchain around the
model is as capable of lying as the model is.* That is this project's thesis wearing work
clothes, and it took three self-inflicted wounds to notice that the thesis applied to the
workbench and not only to the thing on it.

## ADR-037 — Phase 6 inherits a metric with no margin, and the gate is a floor because of it

**Status.** Accepted at Phase 6 scoping, **before** any preference pairs are constructed and
before any DPO code exists.

**Context.** Phase 5 attempt #2 closed GREEN, with `subject_mention` flagged as a grazing
pass (ADR-035). Phase 6 fine-tunes DPO on top of that checkpoint, so `sft.pt` is Phase 6's
base and the grazing is inherited. Stated as a number rather than a caveat:

| sub-score | sft.pt | Phase 5 bar | margin | ICC | n_eff | floor | margin / floor |
|---|---|---|---|---|---|---|---|
| `subject_mention` | 70.8% | 65.3% | **+5.5 pt** | 0.182 | 202 | 8.9 pt | **0.63 — none** |
| `length_band` | 92.9% | 67.8% | +25.1 pt | 0.025 | 290 | 4.2 pt | 6.0x |
| `is_story` | 98.1% | 69.9% | +28.1 pt | 0.000 | 312 | 2.2 pt | 13.1x |
| `not_degenerate` | 90.1% | 64.5% | +25.5 pt | 0.000 | 312 | 4.7 pt | 5.4x |

Floors are priced on **sft.pt's own clustering**, which is worse than base's for the metric
that matters: ICC 0.182 against 0.083, so `n_eff` is 202 rather than 250 and the floor is 8.9
points rather than 8.8. The margin sits further inside the noise than Phase 5's registration
showed, not less far.

**The ambiguity is permanent at this eval design, so no compute can buy it away.** `n_eff` is
bounded above by `k / ICC` = 78 / 0.182 = **429** however many prompts are added. At that
ceiling the detection floor is 6.1 points, still above the 5.5-point margin:

    312 prompts   n_eff 202   floor 8.9pt   not resolvable
    624 prompts   n_eff 274   floor 7.6pt   not resolvable
    ceiling       n_eff 429   floor 6.1pt   not resolvable

Only *more held-out subjects* would help, and that means re-splitting the pool, which means
retraining `sft.pt` — i.e. redoing Phase 5. Not worth it, but it must be said out loud so
nobody later proposes "just run more prompts" as though it were a fix.

**The risk this creates.** Preference optimization optimizes for the preference signal. If
that signal even slightly favours fluency, length, or story-ness over subject fidelity, the
model will spend its 5.5 grazing points buying preference reward and land back below the bar
that an entire extra attempt was run to clear. The other three sub-scores have 5.4x to 13.1x
of room to absorb a trade; `subject_mention` has none. **DPO must not assume subject-adherence
has margin to spend. It does not.**

**Decision.** Phase 6's gate on these four is a **non-regression floor**, not a delta. The
question is not "did preference improve subject adherence" but "did preference cost me the
adherence already paid for" — the same shape as `is_story`-as-floor in Phase 5, applied to
the metric that can least afford to move.

*Registered before pair construction:*

| sub-score | GREEN (p > 0.20) | AMBER (0.05 < p <= 0.20) | RED (p <= 0.05) |
|---|---|---|---|
| `subject_mention` | >= 67.0% | 63.4% - 67.0% | < 63.4% |
| `length_band` | >= 91.2% | 89.5% - 91.2% | < 89.5% |
| `is_story` | >= 97.2% | 96.3% - 97.2% | < 96.3% |
| `not_degenerate` | >= 88.0% | 86.1% - 88.0% | < 86.1% |

Plus a **side-condition on `subject_mention` alone: `dpo >= 65.3%`.** Below the Phase 5 bar,
Phase 5's certification is void whatever the regression test says. This is the tighter of the
two candidate rules — the mechanical ADR-029 line would have been 63.4%, which would have let
DPO land at 64% and report "no regression" while sitting below the bar Phase 5 was certified
against. It costs roughly an 11% false-breach rate on a behaviourally neutral DPO, and that
rate gets reported alongside any breach rather than discovered afterwards.

**Why the amber band is shaped differently here, and this is a correction to how ADR-035
transfers.** ADR-035 installed a two-sided amber for a **delta bar**, where *clearing* is the
achievement and a narrow clearance is the ambiguous case. Applied mechanically to a floor
defined as `sft - k*SE`, it breaks: holding steady at `sft` is by construction only `k*SE`
above the floor, so with an amber half-width of `1.96*SE` **every metric lands AMBER on a
neutral DPO** — an artifact of the definition, not a finding. For a floor, *holding* is the
achievement, so the three zones come from the one-sided significance of the **drop** instead:
no evidence of regression, suggestive, established. A rule that produces the same verdict for
"nothing changed" as for "something might have" is not classifying anything.

**Consequence and scope limit.** This ADR registers the floors only. It deliberately does not
choose the preference-pair construction, which is where the subject-versus-fluency trade will
either be designed out or designed in, and which is a decision to make in the pair design
rather than discover at the gate. What Phase 6's *delta* metric should be — the thing DPO is
supposed to improve — is likewise open, because it depends on what the preference signal is
built to express.

## ADR-038 — Three DPO axes measured saturated; Phase 6 targets the floor metric

**Status.** Accepted at Phase 6 scoping, **before** any preference pair or DPO code exists.

**Context.** Phase 6's signal had to be chosen before pairs were built, because the pairs are
the signal made concrete. Three candidates were on the table, and the fork was posed as
technique-demonstration (fluency preference, the canonical DPO signal) versus
capability-improvement (story resolution). The answer to the fork was *technique
demonstration* — that is what nLemon is — executed on resolution, because a fluency delta
looked unmeasurable and the demonstration value would be carried by defending the subject
floor rather than by the choice of signal.

Then the pre-flight measured all three, which is what pre-flights are for.

**Resolution — saturated.** `src/resolution.py` mines closure markers *from the corpus*,
final sentences against non-final by smoothed log-odds, never hand-listed — the same
discipline that built the coherence bands from percentiles instead of opinion (ADR-018). The
markers are unmistakably TinyStories: `ending`, `vowed`, `onward`, `moral`, `cherish`,
`happily`, `reminder`, `memories`.

    corpus 49.0%   base.pt 35.6%   sft.pt 47.1%   headroom 1.9 pt

And the reading "saturated axis" was separated from the reading "coarse detector" by watching
the **gap** across detector widths rather than the levels:

| markers | corpus | sft | base | gap c−s | sft−base |
|---|---|---|---|---|---|
| 20 | 15.1% | 12.2% | 10.9% | +2.9 | +1.3 |
| 60 | 34.6% | 29.8% | 26.0% | +4.8 | +3.8 |
| 120 | 49.0% | 47.1% | 35.6% | **+1.9** | **+11.5** |

The gap *shrinks* as the detector widens while `sft − base` grows to +11.5. The instrument
is not blind — it detected SFT's own gain on this exact axis — it has nothing left to see.

**Fluency — saturated, and this one had been dismissed by assertion.** Killing resolution by
measurement made the unmeasured claim about fluency the weakest thing on the table, so it got
the same test:

| | corpus | sft.pt |
|---|---|---|
| repeated 4-gram rate | 0.0081 | 0.0122 |
| max repeat run > 2 | 0.3% | **0.0%** |
| type/token ratio | 0.533 | 0.524 |
| oov rate | 0.00000 | 0.00004 |
| mean sentence words | 10.46 | 10.54 |

`sft.pt` is worse than the corpus on repeated 4-grams (both far inside the `[0, 0.0333]`
band), **better** than the corpus on repeat runs, and matched everywhere else. The assertion
happened to be correct; it is now measured, which is the only version that counts.

**Aboutness depth — saturated, and informatively so.** The last candidate: SFT taught subject
*mention*, and cross-entropy cannot distinguish a story that names the subject once from one
genuinely about it. Unconditionally this looked like the biggest opportunity on the board —
`>=2 occurrences` showed sft 62.8% against a 100% corpus, +37.2 points. Conditioning on
mentioning at all dissolves it:

| | mentions | \|≥2 | \|≥3 | \|≥4 | mean\|≥1 |
|---|---|---|---|---|---|
| corpus | 100.0% | 100.0% | 80.1% | 56.1% | 4.28 |
| sft.pt | 70.8% | 88.7% | 78.3% | 65.2% | **4.89** |

When `sft.pt` mentions the subject, it mentions it **more** than the corpus does. The +37.2
was the `subject_mention` gap counted a second time under a different name — the same
double-counting shape as the outcome-selected subset in ADR-030, caught earlier this time.

**Decision.** Phase 6 targets **`subject_mention`**, and the floor/target roles swap.

Post-SFT this model sits at corpus parity on every quality axis the project can measure. The
one exception is the metric SFT was worst at and the one ADR-037 registered as a floor: 70.8%
against a 100% ceiling, **29.2 points**, the only headroom left. Targeting anything else
would be optimizing a variable already at parity, and the delta would be unmeasurable by
construction.

* **Delta target:** `subject_mention`.
* **Floors:** `length_band`, `is_story`, `not_degenerate` — the three with 5.4x to 13.1x of
  room, which is where a trade would now show.
* **Validity, promoted to load-bearing:** the shuffled-subject control. It was a check on the
  Phase 5 headline; here it is the *only* thing standing between this signal and
  noun-spraying, because a naive subject-mention objective is precisely an instruction to say
  the word more often.
* The `dpo_certification_floor_subject_mention` of 65.3% registered in ADR-037 becomes
  trivially satisfied if the target is met. It stays registered rather than being deleted:
  it costs nothing, and a DPO run that *lowered* the metric it was aimed at is exactly the
  outcome worth catching.

**What this phase can honestly claim, stated before the result exists.** Not "DPO taught a
new capability" — the measurement above forecloses that story. The claim is narrower and
better evidenced: **SFT's objective plateaued at 70.8% across two full attempts, and DPO's
contrastive objective is a structurally different attack on the same axis.** Cross-entropy
raises the likelihood of documents that happen to contain the subject; it has no way to
express *"this response is about the right thing and that one drifted."* Whether contrast
closes a gap likelihood could not is a real question with a real answer in either direction,
and two attempts of plateau is the evidence that the question is worth asking.

**Consequence.** `src/resolution.py` is kept despite failing as a target. A corpus-derived
narrative-closure metric that detects an 11.5-point SFT gain is a genuine addition to the
Phase 7 eval harness; it just cannot be Phase 6's objective. Negative results that cost a
module are cheaper than positive results that cost a phase.

## ADR-039 — A bar priced to the mechanism, not to the appearance of stringency

**Status.** Accepted, registered in its own commit **ahead of the pair-building code**.

**Context.** Every prior delta bar in this project was pushed *up*, toward the decisive
multiple and away from the friendly one, because each measured a capability the model was
being **taught**: `<= 8` perplexity could fail on undertraining, `+25` subject-mention could
fail on capacity, and a low bar there would have been decorative.

Phase 6 is not that. It measures how far a **refinement** technique can re-rank a
distribution the model already has. DPO cannot teach subject-adherence — `sft.pt` already
has it at 70.8% — it can only shift weight toward the subject-faithful samples the model
already produces, and the ceiling on that is bounded by what is in the distribution to
re-rank.

**Decision.** `subject_mention` delta **+12.5 points, bar 83.3%**, which is **1.41x** the
8.9-point detection floor — deliberately weaker than Phase 5's 2.0x.

Matching 2.0x would have meant +17.8, asking DPO to close 61% of remaining headroom with no
evidence it can. That is the low-bar trap pointed the other way: not *"a bar so low it cannot
fail"* but *"a bar so high it fails for a reason that is not the model's fault."* A RED at
+17.8 would say only that re-ranking had been asked to do teaching's job, which is a fact
about the registration rather than about DPO.

The anchor, with its bias stated rather than buried: Phase 5's SFT closed **44%** of its
subject headroom **by teaching**. 44% of the remaining 29.2 points is +12.8 — the
**optimistic** edge, not a neutral estimate, because refinement should not be expected to
match teaching. +12.5 rounds just inside it, which is the honest direction to round.

**The multiple being lower than Phase 5's is not a defect to apologise for.** It is the
correct signal that this bar measures something a weaker mechanism does. The write-up says
so directly: *the multiple is lower because the claim is smaller — DPO refines, it does not
teach, and the bar is priced to what refinement can honestly deliver.* Pricing a bar to the
mechanism is more rigorous than pricing it to look stringent.

**Two-sided amber** per ADR-035, since this is a delta bar and not a floor:

    GREEN    dpo >= 92.2%          cleared by more than the eval can resolve
    AMBER+   83.3% - 92.2%         cleared, but grazing
    AMBER-   74.4% - 83.3%         missed, but inside the noise
    RED      < 74.4%               resolvable miss

**Pre-committed amber response**, because at 1.41x an amber is genuinely likely and the
answer must not be chosen after seeing the number. Critically, **the answer is not "run DPO
longer."** An undertraining amber extends steps; a DPO amber must not. DPO past its useful
point sharpens toward the preference signal until it starts trading everything else against
it, so extending is the single move that converts a weak result into a broken one.

* **amber delta + all floors green** — DPO refined weakly but honestly. Report it, close the
  phase, do not extend.
* **amber delta + any floor breached** — DPO is spending the grazing margin. Stop, report the
  trade, do not extend.

Those are **opposite diagnoses with opposite responses**, which is precisely why the floors
are read before the delta.

**Gate read order, fixed here so it cannot be chosen after the fact:**

1. **side-condition** — `subject_mention >= 65.3%`. Is Phase 5 still certified?
2. **floors** — `length_band`, `is_story`, `not_degenerate`.
3. **delta** — `subject_mention >= 83.3%`. Did DPO improve its target?

The headline improvement is read **last**. Both things that can void the phase sit upstream
of it, and `sft.pt` has only 5.6 points of margin over the side-condition — the least room
anywhere in this phase, and the first thing a fluency-adjacent leak in the pairs would spend.

**Pair construction, registered with the bar because the design determines what the bar can
mean.** On-policy: sample k responses from `sft.pt` per training prompt, chosen = one that is
about the subject, rejected = one that is not. Both sides come from the model's own
distribution, so DPO re-ranks what it already samples rather than being pulled off it.

*The matched control is the load-bearing decision.* Chosen and rejected must be comparable on
**length and `is_story`**, so subject fidelity is the only systematic difference. Without it
the preference signal would encode "longer is better" or "ends with a period is better" —
inside the pairs, where no gate can see it, with the floors sitting green while DPO optimised
the exact trade they exist to catch. This is the answer to the sharpest question the phase
invites: *how do you know DPO taught subject-fidelity and not length?* Because the pairs were
matched on length, so length could not be the signal.

*And the match must hold jointly, not marginally.* Matching on length and separately on
`is_story` is not enough if the aboutness filter layered on top correlates with length — a
story genuinely about its subject may well run longer, and if it does, matched pairs
**un-match themselves at the point of selection**. The check is therefore length parity of
chosen against rejected **after** the aboutness filter, not before. If they have drifted,
re-match after filtering rather than before.

Chosen must clear **aboutness >= 2**, not bare mention. Selecting on bare mention is literally
an instruction to say the word more often; the shuffled control catches noun-spraying at the
gate, and requiring depth in the pair design is how it is avoided rather than merely detected.

**Rejected alternatives, and the second is the dangerous one.** *A corpus story about a
different subject* — contrast too easy, teaches "match the topic noun", and off-policy. *A
corpus story about the right subject as chosen* — this is not merely off-policy, it is **the
phase's confound in disguise**: chosen would be human text and rejected model text, so DPO
would learn "sound like the corpus", which is the fluency signal wearing a subject label. It
is the option a future reader is most likely to think was clever, so it is named here.

**Held-out discipline.** Pairs are built from the **310 train subjects only**; the **78
held-out subjects never appear**. That is what keeps the Phase 6 number a generalisation
claim rather than a memorisation one, the same standard as Phase 5, and it must not slip when
the eval is built: held-out 78 are eval subjects, train 310 are pair subjects, and they do
not cross.

## ADR-040 — A long run was killed on two numbers I did not measure

**Status:** accepted. **Phase:** 6 (preference-candidate sampling).

Preference sampling needed 7,440 generations. `generate` runs at batch size 1, which held the
GPU at ~35% utilisation and 370 MiB of a 4 GiB card. So batching was the right fix. It is not
what this ADR is about.

**What actually happened: I killed a running job on two figures, and neither was measured.**

1. *"This will take 28 hours."* Extrapolated from the wall-clock of the Phase 5 scoring runs.
   Those runs load the model, load the tokenizer, and score — and I divided the whole thing by
   the generation count as though it were all generation. The real unbatched rate was
   0.84 s/gen, which is **~4.2 hours**, not 28. I inflated the cost of continuing by 6.7x.
2. *"It's roughly a quarter done."* This one was worse, because it was not an extrapolation at
   all. **It was a guess presented in the same register as a measurement.** Python block-buffers
   stdout when it is redirected to a file, so the log was empty and I had no progress data of
   any kind. I had no way to know, and I said a number anyway.

On those two figures the run was killed. It may have been nearly finished. **The root cause was
never speed — it was that I could not see, and filled the gap with confident arithmetic instead
of saying "I cannot tell."**

**The fix is visibility, and it is deliberately redundant:**

* `flush=True` on every progress line, so the log is live rather than block-buffered.
* **`data/pref_samples_partial.json`, rewritten every N prompts with `{done, of, elapsed_min}`.**
  Progress lands on disk in a parseable form, readable by anything, surviving any console.

The redundancy paid for itself on the very next run. The relaunched job's task log came back
**0 bytes despite `flush=True`** — the console capture failed the same way again. The on-disk
checkpoint was the *only* reason progress was visible at all, and it is what the 66%-complete
and rate readings were taken from. **The mechanism that was arguably belt-and-braces is the one
that worked; the primary one failed twice.** Progress must be written to a file, not printed.

**And the estimate was still wrong once, in a smaller way.** With visibility restored I read the
first checkpoint — 50 prompts in 1.69 min — and projected ~42 min. The run took **63**. The
first 50 were an unrepresentative burst; the rate between two later checkpoints was 19.2
prompts/min against the early 29.6. The lesson is narrow and repeats ADR-023's shape: *an
average taken over an early, small, unrepresentative slice is not a rate.* The second estimate,
taken as a difference between two on-disk checkpoints, landed within 5 minutes of the truth.
**Two points beat one point, even when one point is a measurement.**

**On the batching itself, which is the easy half.** `generate_batch` keeps a separate
`torch.Generator` per sequence, so sequence *i* always draws from *i*'s generator and batching
changes only *when* forward passes happen, never which token is drawn. That is an identity
claim, so it is asserted rather than argued: **8/8 sequences bit-identical to the unbatched
path** before any pairs were built. Had it silently diverged, the pairs would have come from a
different distribution than the gate measures — ADR-025's defect in a new location.

Sequences that have emitted EOT keep being fed through the model to hold the tensor rectangular
and keep drawing from their generators; everything after their EOT is discarded, which is what
preserves the identity. All prompts in a batch must share a length: position embeddings are
learned and indexed from zero, so left-padding would shift every position and change the input.
Grouping by exact prompt length avoids padding rather than compensating for it.

Measured sweep, not assumed: **B=8 optimal, B=16 slower, B=32 OOM.** There is no KV cache, so
each step re-runs the full window and the arithmetic grows quadratically with batch — the
sweet spot is low and had to be found by trying, not derived.

**A separate correction, recorded because it was a bar that could not be cleared.**
`pref_prompts` was configured at 2000. There are 310 train subjects and 4 templates, and
`cmd_sample` dedups on `(subject, prompt)` — so **1240 distinct prompts is the entire
population, and 2000 was unreachable.** Sampling to 2000 would have drawn repeats from an
identical distribution and reported them as diversity. Corrected to 1240 with the derivation in
the config comment. Same family as ADR-028: a number that the data was structurally incapable
of reaching, sitting in the config looking like a choice.

**The rule this installs.** A progress figure is a measurement or it is not said. "I don't know
how far along it is" is a complete and acceptable answer; a guessed percentage is not, because
it is indistinguishable from a real one at the point where someone acts on it. Before a long job
starts, the question is not "how fast is it" but **"how will I know how far it has got"** — and
the answer has to be a file.

## ADR-041 — The step-0 assertion passed in a mode training did not use

**Status:** accepted. **Phase:** 6 (DPO). **Found:** during the first DPO run, before any gate
was read. **Both defects below were fixed and committed before the Phase 6 result existed.**

### The assertion that passed while the objective was wrong

Phase 6's startup assertion is this phase's mask-assertion equivalent: policy and reference are
both `sft.pt`, so every bracketed term in the DPO objective is zero and the loss must be exactly
`-log(0.5)`. It ran, it printed `0.693147`, it passed at `1.9e-09`.

Then the first logged training step came out at **0.804782**.

The assertion ran under `policy.eval()`. The next line was `policy.train()`. **`dropout: 0.1`.**
The reference logprobs are precomputed once in eval mode, so the moment training began, the
policy's logprobs carried dropout noise the reference's did not — and the bracketed term was no
longer a comparison of two evaluations of the same function. It was noise against a constant.

*Diagnosed by contrast, not by inference.* Same weights, same batch, no optimiser step, only the
mode flag changed:

| mode | loss | spread over repeats |
| --- | --- | --- |
| `eval()` | 0.693147 (dev `1.9e-09`) | **exactly 0.0** — deterministic |
| `train()` | 0.677 … 0.922 | **0.245** — stochastic |

Zero variance with dropout off proves the weights match. A quarter-nat swing from nothing but
the mode flag proves the excess is dropout, not a policy/reference mismatch — the two have
opposite signatures, and a weight mismatch would have been *repeatable*.

*And the magnitude closes exactly.* Per-pair logit sd at initialisation is **0.6723**. Taking
the empirical expectation `E[-log sigma(x)]` over those logits gives **0.824300** against a
measured **0.824299** — ratio 1.00. Dropout accounts for the entire excess; nothing else was
going on.

A remainder worth naming, because the first attempt to close it failed: the second-order
approximation `log2 + Var/8` predicted only +0.0565 against +0.1312, off by 2.32x. Two reasons,
and the first was my error. I had used the sd of the **batch-mean** logit (0.3741) where Jensen
acts on **per-pair** logits — a factor of `sqrt(micro_batch)` wrong by construction. Even with
the right sd, a quadratic Taylor term is invalid at sd 0.67. **The exact expectation is the
right instrument here and the approximation is not**, which is the ADR-033 lesson in a new
place: the convenient closed form was measuring something adjacent to the quantity of interest.

*Why it matters beyond tidiness.* The per-pair noise sd was 0.67 at initialisation. The
preference signal DPO is trying to learn is smaller than that. Training would have proceeded
down a plausible-looking curve while the gradient was mostly noise — **ADR-014's exact shape,
which this assertion was written to prevent, occurring in the assertion itself.**

### The fix, and the half of it that generalises

Dropout is now off for the entire DPO run: the policy stays in `eval()` mode throughout. The
model has no batchnorm, so this disables dropout and changes nothing else, and eval mode is not
`no_grad` — gradients still flow.

But turning dropout off is the local fix. **The general fix is that the assertion now has two
halves:**

* **value** — the loss must equal `-log(0.5)`.
* **determinism** — the same batch, twice, must give a **bit-identical** loss.

A value check alone passes happily while the objective is stochastic, because dropout noise is
zero-mean *inside the bracket*. Only repeatability separates "policy equals reference" from
"policy equals reference **on average**". The second half is the one that would have caught
this, and it is the one to carry into any future assertion of this kind.

**The rule: an assertion must run in the configuration the thing being asserted about actually
runs in.** An assertion evaluated under conditions training does not use is not an assertion
about training. This one was off by a single line — `policy.train()` immediately after the
check — and that line silently invalidated it.

### A second defect found in the same run: warmup longer than the run

`dpo_warmup_steps: 50`, in a run of **22 steps** (714 pairs / effective batch 32, one epoch).
Linear warmup therefore reached `5e-7 * 22/50 = 2.2e-7` at the *final* step and averaged
~1.15e-7. **The registered learning rate of 5e-7 was never once applied.** The value was carried
over from a schedule measured in thousands of steps.

This is ADR-028 and the `pref_prompts` correction in ADR-040 wearing a third face: *a
configured number that the run's own structure makes unreachable, sitting in the config looking
like a deliberate choice.* Three instances now, in three different phases. The pattern is not
"someone picked a bad value" — it is that **a config value and the structure that consumes it
are validated separately, so nobody checks the value against the structure.** Any schedule
parameter expressed in steps should be asserted against the step count it will actually see.

Corrected to 2, roughly 10% of the run, so 5e-7 is the operative rate for 20 of 22 steps — which
is what ADR-039 registered.

### On changing two things before reading the gate

Both changes alter the treatment: one removes noise from the gradient, the other multiplies the
effective learning rate by about 4x. The test from Phase 5 applies — *would I make this change if
the gate had passed?* Yes to both, and neither is a judgement call: an objective that fails its
own repeatability check is broken, and a warmup longer than its run is arithmetic, not taste.

The protection is the ordering, and it is the only thing that makes this legitimate: **no gate
had been read, no eval had been run, and no `dpo.pt` existed** — the first run was killed at step
1 of 22 and wrote no checkpoint. This ADR and both fixes are committed **before** the Phase 6
result is generated. Had either defect surfaced after a disappointing gate reading, the honest
move would have been to report the RED first and the fix second, exactly as Phase 5 did.

Post-fix, the curve confirms itself in band: step 1 logs **0.693147** — the objective now starts
where it must under the conditions it trains under. (Ranking accuracy reads 0.000 at step 1
because `logits > 0` is false at exactly zero; a perfect tie, correctly reported.)
