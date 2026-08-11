# nLemon-14 — deployment runbook

**Corrected against the artifacts, 2026-08-11.** An earlier handoff named the weights
`nlemon-14-sft-fp32.pt`. **The certified artifact is `nlemon-14-sft-weights.pt`, sha256
`e3f6b7ce7e5a1a74337a3a3ae2f2968ff1131e9de79843606cef705748e7668b`.** The file was not renamed to
match the doc: the certification attaches to the bytes, not the name, and regenerating a certified
artifact to make documentation true is the tail wagging the dog. **The doc yields to the file.**

---

## Pre-flight — complete

All five seams are asserted at Space startup rather than trusted, and the app refuses to serve if
any fails. Three of the original four constants were wrong; see the commit for
`2521ef1` and the notes below.

| seam | status |
|---|---|
| model construction | built from the **checkpoint's own stored config**, `strict=True` load. `GPT()` with no args raises — it takes a `Config`. |
| wire format | `prompt + "\n"`, byte-compared in test against the repo's own `wire()` helper. The bare prompt would be silently out-of-distribution. |
| `PROMPT_TEMPLATE` | `"Write a story about {subject}."` — `TEMPLATES[0]` from `src/instruct.py`, byte-for-byte |
| `EOT_ID` | **derived** from the tokenizer and the checkpoint's `doc_separator` (it is `0`), never hardcoded. Stop test is `is not None`, because the id is falsy. |
| weights identity | downloaded file's sha256 checked against the certified hash; mismatch refuses to serve |

There is **no reference-architecture fallback.** A fallback would load the checkpoint into an
approximation of the real model and serve a configuration the scorecard never certified.

**Cold start:** 3.5 s load + 1.7 s generation on this machine. HF free CPU is slower hardware plus
a 55.3 MB Hub pull, so the first visitor after a sleep waits noticeably longer. Known and accepted.

**Licensing:** weights/code MIT; corpus TinyStories rev `f54c09fd…` under CDLA-Sharing-1.0, whose
share-alike binds the *Data* while computational *Results* sit outside it. Stated on both the model
card and the Space. Not legal advice.

---

## Authentication

The token lives at **`~/.nlemon-deploy.env`**, outside this repo, because the Space build walks the
repo and a token inside a repo is one bad glob from being published. Load it without echoing it or
writing it into shell history:

```bash
set -a; . ~/.nlemon-deploy.env; set +a
hf auth whoami          # PASS: prints the username
```

A **Write**-capable token is required — a read token cannot create repos or upload. Revoke it after
the deploy; nothing in the running Space depends on it.

## Build the Space (never hand-copy)

```bash
python -m scripts.build_space          # asserts every vendored file byte-identical to source
python -m scripts.build_space --verify  # re-check an existing build at any time
```

Output lands in `build/space/` (gitignored — it is derived). Authored Space files live in `space/`.

## Step A — model repo, first, because the Space depends on it

```bash
hf repo create nlemon-14 --repo-type model          # confirm flag with --help; CLI versions differ
hf upload akshay0689/nlemon-14 checkpoints/nlemon-14-sft-weights.pt --repo-type model
hf upload akshay0689/nlemon-14 MODEL_CARD.md README.md --repo-type model
```

**Then re-verify the uploaded bytes — do not trust the upload:**

```bash
hf download akshay0689/nlemon-14 nlemon-14-sft-weights.pt --local-dir /tmp/nl_verify
sha256sum /tmp/nl_verify/nlemon-14-sft-weights.pt
# PASS: e3f6b7ce7e5a1a74337a3a3ae2f2968ff1131e9de79843606cef705748e7668b
# Anything else: the uploaded artifact is not the certified one. Re-upload; do not proceed.
```

## Step B — Space repo

```bash
hf repo create nlemon-14 --repo-type space --space-sdk gradio
cd build/space && hf upload akshay0689/nlemon-14 . --repo-type space
```

## Step C — verify live

1. **Logs tab.** PASS: build completes and the log prints `weights sha256 verified`,
   `unique parameters 13,817,856`, `EOT_ID derived as 0`, `sft_stage_hash ecad1e4b412d`. Any hash
   or seam failure exits with a message rather than serving.
2. **Smoke-test in the browser.** A subject chip returns a coherent story in seconds; the same seed
   twice gives a byte-identical story; temperature 0.2 loops and 1.5 with wide top-k invents words —
   both documented failure directions, visible on demand.
3. **Incognito window** (no HF login) loads and generates → the Space is public.
4. **Surfaces agree.** Model card and Space both show `5.2662 ± 0.27%`; the card carries the
   tied-head arithmetic (`16,889,856 − 8000×384 = 13,817,856`); neither contradicts the shipped file.

**Done when** the re-downloaded hash matches `e3f6b7ce…`, the Space generates in an incognito
window, and every surface shows the same numbers.
