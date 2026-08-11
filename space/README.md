---
title: nLemon-14
emoji: 🍋
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---

# nLemon-14

A 14-million-parameter language model built from scratch — data, tokenizer,
architecture, pretraining, and alignment — end to end on a 4&nbsp;GB laptop GPU.
It writes simple children's stories, and it stays on its subject about 71% of the
time. That limit is measured, not hidden.

Pick a subject, get a short story, and read the numbers next to it. The point of
the project isn't that the model speaks; it's that every gate it was measured
against was pre-registered, allowed to fail, and several times did.

- Code: <https://github.com/akshay100188/nlemon>
- Write-up: <https://akshaybhatnagar.me>
- Model card: <https://huggingface.co/akshay100188/nlemon-14>

## Licensing

- **This Space, `app.py`, and the model weights:** MIT.
- **Training corpus:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  (revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`) under
  **CDLA-Sharing-1.0**. That licence's share-alike applies to the *Data*;
  computational **Results** obtained from analysing the data are explicitly
  outside it, which is the basis for releasing the weights under MIT. The corpus
  itself is not redistributed here.

## How this Space is built

**Do not hand-copy files into this Space.** It is assembled from the source repo
by `scripts/build_space.py`, which vendors `model.py`, `config.py`, `utils/` and
`tokenizer.json` from the repo and **asserts each copy is byte-identical to its
source**. Hand-copying is how a Space starts serving an architecture that has
quietly drifted from the one that was measured.

The app asserts its four integration seams at startup and refuses to serve if any
fails:

1. the model is built from the **checkpoint's own stored config**, not from
   constants in `app.py`, and loaded with `strict=True`;
2. the prompt is sent as `PROMPT + "\n"` — the wire format the model was trained
   and evaluated on;
3. `EOT_ID` is derived from the tokenizer and the checkpoint's `doc_separator`
   (it is token `0`), never hardcoded;
4. the downloaded weights' **sha256 is checked against the hash Phase 7
   certified**, so the Space cannot demo an uncertified artifact.

There is deliberately **no reference-architecture fallback**. A fallback would
load the checkpoint into an approximation of the real model and serve a
configuration the scorecard never certified.

Runs on the **free CPU tier** — a 14M model does a forward pass in milliseconds,
so no GPU is needed.
