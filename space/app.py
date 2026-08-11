"""nLemon-14 - public try-it Space.

A 14M-parameter transformer trained from scratch on TinyStories. This app is the
storefront: constrained prompt in, a small story out, and the measured numbers
sitting next to it so a visitor reads the model for what it is.

THE FOUR INTEGRATION SEAMS ARE NOT CONSTANTS TO TRUST - they are asserted at
startup and the app refuses to serve if any fails. Each one was wrong in the
first draft of this file, and each would have been invisible to a visitor:

  1. MODEL       built from the checkpoint's OWN stored config, so the
                 architecture comes from the certified artifact rather than from
                 numbers retyped here. `GPT(cfg)` takes a Config - calling
                 `GPT()` raises. Loaded with strict=True: no silent key drift.
  2. WIRE FORMAT the model was trained and evaluated on `prompt + "\\n"`.
                 Sending the bare prompt is a silent out-of-distribution
                 request, and `src/sft_data.py` carries a warning about exactly
                 this. The newline is not cosmetic.
  3. EOT_ID      derived from the tokenizer and the checkpoint's doc_separator,
                 never hardcoded. It is token 0 for this tokenizer, which is
                 also why the stop test must be `is not None` and not truthiness.
  4. WEIGHTS     the downloaded file's sha256 is checked against the hash Phase 7
                 certified. If the Hub serves a different artifact, this app
                 stops rather than quietly demoing an uncertified model.

There is deliberately NO reference-architecture fallback. A fallback that loads
the checkpoint into an approximation of the real model would serve a
configuration the scorecard never certified - the same class of defect as
measuring two stages on two devices (ADR-047).
"""

import hashlib
import sys

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from config import Config
from model import GPT

# -- config --------------------------------------------------------------
MODEL_REPO = "akshay100188/nlemon-14"
WEIGHTS_FILE = "nlemon-14-sft-weights.pt"      # the certified release asset
WEIGHTS_SHA256 = "e3f6b7ce7e5a1a74337a3a3ae2f2968ff1131e9de79843606cef705748e7668b"
TOKENIZER_FILE = "tokenizer.json"

# TEMPLATES[0] from src/instruct.py, byte-for-byte. The model was trained on four
# templates; this is one of them, so the prompt is in-distribution.
PROMPT_TEMPLATE = "Write a story about {subject}."
MAX_NEW_TOKENS = 245        # the Phase 5 cap: largest that keeps prompt + response in the window

# The pinned pair every published number used.
PINNED_TEMPERATURE = 0.8
PINNED_TOP_K = 40

DEVICE = "cpu"              # 13.8M params: a forward pass is milliseconds on CPU


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_everything():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=WEIGHTS_FILE)

    digest = _sha256(path)
    if digest != WEIGHTS_SHA256:
        raise SystemExit(
            f"WEIGHTS HASH MISMATCH - refusing to serve.\n"
            f"  expected {WEIGHTS_SHA256}\n  got      {digest}\n"
            f"The Hub served an artifact that is not the one Phase 7 certified.")
    print(f"[nLemon] weights sha256 verified: {digest[:16]}...")

    state = torch.load(path, map_location=DEVICE, weights_only=True)

    # The architecture comes from the artifact, not from constants in this file.
    cfg = Config(**state["config"])
    model = GPT(cfg)
    model.load_state_dict(state["model"], strict=True)   # strict: no silent drift
    model.eval().to(DEVICE)

    tok = Tokenizer.from_file(TOKENIZER_FILE)
    eot = tok.token_to_id(cfg.doc_separator)
    if eot is None:
        raise SystemExit(
            f"tokenizer has no id for doc_separator {cfg.doc_separator!r} - the "
            f"tokenizer.json in this Space does not match the checkpoint.")

    n_unique = sum(p.numel() for p in model.parameters())
    print(f"[nLemon] model built from checkpoint config: "
          f"vocab {cfg.vocab_size}, d_model {cfg.d_model}, layers {cfg.n_layers}, "
          f"ctx {cfg.context_len}")
    print(f"[nLemon] unique parameters {n_unique:,} (tied head)")
    print(f"[nLemon] sft_stage_hash {state.get('sft_stage_hash')}  "
          f"step {state.get('step')}")
    print(f"[nLemon] EOT_ID derived as {eot} from doc_separator "
          f"{cfg.doc_separator!r}")
    print(f"[nLemon] wire format: PROMPT + newline (as trained)")
    return model, tok, cfg, eot


MODEL, TOKENIZER, CFG, EOT_ID = load_everything()


@torch.no_grad()
def generate(subject, temperature, top_k, seed):
    subject = (subject or "").strip()
    if not subject:
        return "Type or pick a subject, then press Tell the story."

    # The newline is the wire format the model was trained and measured on.
    wire = PROMPT_TEMPLATE.format(subject=subject) + "\n"
    ids = TOKENIZER.encode(wire).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Sampling on a CPU generator: the draw depends only on the seed, not on the
    # machine, which is why the same seed gives the same story anywhere.
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))

    new_ids = []
    for _ in range(MAX_NEW_TOKENS):
        window = idx[:, -CFG.context_len:]
        out = MODEL(window)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits[:, -1, :] / max(float(temperature), 1e-6)

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            vals, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < vals[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=g)
        tok = int(nxt.item())
        if EOT_ID is not None and tok == EOT_ID:   # EOT is token 0: test identity
            break
        new_ids.append(tok)
        idx = torch.cat([idx, nxt], dim=1)

    story = TOKENIZER.decode(new_ids).strip()
    return story or "(the model produced nothing this time - try another seed)"


# -- UI ------------------------------------------------------------------
EXAMPLE_SUBJECTS = [
    "a lost kitten",
    "a brave little frog",
    "a duck who could not swim",
    "a prince and a friendly dragon",
    "a swan on a quiet lake",
    "a carrot that grew too big",
]

FRAMING = """
# nLemon-14
### Born to speak, disciplined to behave.

nLemon-14 is a 14-million-parameter language model built from scratch - data,
tokenizer, architecture, pretraining, and alignment, end to end on a 4&nbsp;GB
laptop GPU. It writes simple children's stories.

It is roughly 500x smaller than the models behind most chat assistants, so it
will sometimes lose the thread: start a story about a lizard and it may finish
about a rabbit. That wobble is measured, not hidden - it stays on its subject
about 71% of the time, and the write-up shows exactly why. Pick a subject below
and see what fourteen million parameters can do.
"""

SCORECARD = """
<div class="scorecard">
  <div class="sc-title">Measured, on held-out data</div>
  <div class="sc-grid">
    <div class="sc-cell"><span class="sc-num">5.2662</span><span class="sc-pm">&plusmn; 0.27%</span><span class="sc-lab">validation perplexity<br><em>deff-adjusted precision, not the iid figure</em></span></div>
    <div class="sc-cell"><span class="sc-num">7.65x</span><span class="sc-lab">better than the bigram baseline<br><em>beating it means modelling context, not local statistics</em></span></div>
    <div class="sc-cell"><span class="sc-num">70.8%</span><span class="sc-lab">subject adherence<br><em>on 78 subjects never seen in training</em></span></div>
    <div class="sc-cell"><span class="sc-num">13,817,856</span><span class="sc-lab">parameters<br><em>384 wide &middot; 6 layers &middot; 6 heads &middot; 256 context</em></span></div>
  </div>
</div>
"""

BUILT = """
<div class="built">
  <div class="built-title">How it was built</div>
  <p>Eight phases, each with a gate fixed before the run and allowed to fail.
  Several went red. Every red was diagnosed and published rather than smoothed
  over - the through-line of the project is not that the model speaks, but that
  every gate it trusted, it first had to catch lying.</p>
  <p class="built-links">
    <a href="https://github.com/akshay100188/nlemon" target="_blank">Code</a>
    <span>&middot;</span>
    <a href="https://akshaybhatnagar.me" target="_blank">Write-up</a>
    <span>&middot;</span>
    <a href="https://huggingface.co/akshay100188/nlemon-14" target="_blank">Model card</a>
  </p>
</div>
"""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#FCFCFA",
    body_text_color="#16233A",
    block_background_fill="#FFFFFF",
    block_border_color="#E7E9EE",
    button_primary_background_fill="#F5D50A",
    button_primary_background_fill_hover="#EAC900",
    button_primary_text_color="#16233A",
    button_primary_border_color="#F5D50A",
)

CSS = """
:root { --navy:#16233A; --blue:#2F5F9E; --lemon:#F5D50A; --line:#E7E9EE; }
.gradio-container { max-width: 940px !important; margin: 0 auto !important; }

#framing h1 { font-family:"Fraunces", Georgia, serif; font-weight:600;
  font-size:2.9rem; letter-spacing:-0.02em; margin-bottom:0.1rem; color:var(--navy); }
#framing h3 { font-family:"Fraunces", Georgia, serif; font-weight:400; font-style:italic;
  color:var(--blue); margin-top:0; font-size:1.15rem; }
#framing p { font-size:1.02rem; line-height:1.6; color:#2b3a52; max-width:62ch; }

#story-card textarea, #story-card .prose {
  font-family:"Fraunces", Georgia, serif !important; font-size:1.18rem !important;
  line-height:1.75 !important; color:var(--navy) !important;
  background:#FFFCE9 !important; border:1px solid #F0E6A8 !important;
  border-radius:14px !important; padding:1.4rem 1.6rem !important; min-height:230px;
}
#story-label { font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;
  color:var(--blue); font-weight:600; margin-bottom:0.4rem; }

button.primary, #go { font-weight:600 !important; font-size:1rem !important;
  letter-spacing:0.01em; border-radius:10px !important; }

.scorecard { border:1px solid var(--line); border-radius:14px; padding:1.3rem 1.5rem;
  background:#FBFCFE; margin-top:0.5rem; }
.sc-title { font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--blue); font-weight:700; margin-bottom:1rem; }
.sc-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.1rem 1.6rem; }
.sc-cell { display:flex; flex-direction:column; gap:0.15rem; }
.sc-num { font-family:"JetBrains Mono", monospace; font-size:1.5rem; font-weight:600;
  color:var(--navy); line-height:1.1; }
.sc-pm { font-family:"JetBrains Mono", monospace; font-size:0.9rem; color:var(--blue); }
.sc-lab { font-size:0.82rem; color:#5a6577; line-height:1.35; margin-top:0.15rem; }
.sc-lab em { color:#8a93a3; font-style:italic; }

.built { margin-top:1.4rem; padding-top:1.2rem; border-top:1px solid var(--line); }
.built-title { font-family:"Fraunces", serif; font-size:1.1rem; color:var(--navy);
  font-weight:600; margin-bottom:0.4rem; }
.built p { font-size:0.92rem; line-height:1.6; color:#4a5568; max-width:66ch; }
.built-links a { color:var(--blue); text-decoration:none; font-weight:600; }
.built-links a:hover { text-decoration:underline; }
.built-links span { color:#b7bdc8; margin:0 0.5rem; }

#advanced { border:1px dashed var(--line); border-radius:10px; }
.note { font-size:0.82rem; color:#8a93a3; line-height:1.45; }
"""

with gr.Blocks(theme=theme, css=CSS, title="nLemon-14") as demo:
    gr.Markdown(FRAMING, elem_id="framing")

    with gr.Row():
        with gr.Column(scale=5):
            subject = gr.Textbox(
                label="A story about...",
                placeholder="a lost kitten",
                lines=1,
            )
            gr.Examples(
                examples=[[s] for s in EXAMPLE_SUBJECTS],
                inputs=subject,
                label="Subjects it knows well",
            )
            go = gr.Button("Tell the story", variant="primary", elem_id="go")

            with gr.Accordion("Decoding - move these to watch it break",
                              open=False, elem_id="advanced"):
                gr.Markdown(
                    '<p class="note">Pinned at temperature 0.8, top-k 40 - the pair every '
                    "published number used. Cool it toward 0.2 and it loops; heat it past 1.2 "
                    "with a wide top-k and it starts inventing words. Both failures are real "
                    "and measured.</p>"
                )
                temperature = gr.Slider(0.1, 1.6, value=PINNED_TEMPERATURE, step=0.1,
                                        label="Temperature")
                top_k = gr.Slider(0, 200, value=PINNED_TOP_K, step=5, label="Top-k (0 = off)")
                seed = gr.Number(value=1337, precision=0,
                                 label="Seed (same seed = same story)")

        with gr.Column(scale=6):
            gr.Markdown('<div id="story-label">The story</div>')
            story = gr.Textbox(
                show_label=False,
                lines=10,
                elem_id="story-card",
                interactive=False,
                placeholder="Your story will appear here.",
            )

    gr.HTML(SCORECARD)
    gr.HTML(BUILT)

    go.click(generate, inputs=[subject, temperature, top_k, seed], outputs=story)
    subject.submit(generate, inputs=[subject, temperature, top_k, seed], outputs=story)


if __name__ == "__main__":
    demo.launch()
