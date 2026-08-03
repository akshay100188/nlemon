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
| `U+00C2` Â | 9 | **Our own bug.** `U+0080`–`U+00BF` encode to UTF-8 as `C2 xx`, so their mojibake is `Â` + the character. The no-break-space and soft-hyphen rules rewrote the *second* half and stranded the first — `he<Â>'d`, `loved.<Â>`, `wasn<Â>'t`. | **Fixed.** A rule for the orphaned prefix now runs last, after the pair rules. Audited: all 9 were strandings, none a real capital A-circumflex. |
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
