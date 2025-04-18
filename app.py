#!/usr/bin/env python3
"""app.py
Gradio UI for on‑demand text‑to‑audio synthesis.

Key change
~~~~~~~~~~
Audio is **no longer written to disk**.  Generated clips are kept in memory and
fed directly to :class:`gr.Audio` widgets (``type="numpy"``).  Users can still
preview and download each clip individually via the built‑in download button,
but no permanent archive is created under ``./results``.
"""
from __future__ import annotations

from typing import List

import gradio as gr
import numpy as np

from gen_wav import SAMPLE_RATE, device, gen_wav, initialize_model  # type: ignore
from vocoder.bigvgan.models import VocoderBigVGAN  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_AUDIO_PLAYERS: int = 10  # simultaneous preview widgets

SAMPLER = initialize_model(
    config="configs/text_to_audio/txt2audio_args.yaml",
    ckpt="useful_ckpts/maa1_full.ckpt",
)
VOCODER = VocoderBigVGAN("useful_ckpts/bigvgan", device=device)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

def generate_and_update(
    prompt: str,
    ddim_steps: int,
    duration: int,
    n_samples: int,
    scale: float,
):
    """Generate *n_samples* audio clips and build Gradio update payloads."""
    # ``gen_wav`` returns a list[numpy.ndarray] with values in [-1, 1].
    wavs = gen_wav(
        sampler=SAMPLER,
        vocoder=VOCODER,
        prompt=prompt,
        ddim_steps=ddim_steps,
        scale=scale,
        duration=duration,
        n_samples=n_samples,
    )

    updates: List[gr.update] = []
    for i in range(MAX_AUDIO_PLAYERS):
        if i < len(wavs):
            # Gradio expects (sample_rate, np.ndarray) for ``type="numpy"``.
            updates.append(gr.update(value=(SAMPLE_RATE, wavs[i]), visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Assemble and return the Gradio Blocks interface."""
    with gr.Blocks() as demo:
        gr.Markdown("## Text‑to‑Audio Generator 🎙️")

        prompt_in = gr.Textbox(
            label="Prompt",
            value="a bird chirps",
            placeholder="Describe the sound you want to generate…",
        )
        ddim_in = gr.Slider(label="DDIM Steps", minimum=1, maximum=500, value=100)
        duration_in = gr.Slider(label="Duration (s)", minimum=1, maximum=60, value=10)
        samples_in = gr.Slider(
            label="Number of Samples",
            minimum=1,
            maximum=MAX_AUDIO_PLAYERS,
            value=1,
            step=1,
        )
        scale_in = gr.Slider(
            label="Guidance Scale", minimum=0.0, maximum=10.0, value=3.0, step=0.1
        )
        generate_btn = gr.Button("Generate")

        gr.Markdown("### Preview & Download")
        audio_players = [
            gr.Audio(label=f"Clip {i+1}", visible=False, type="numpy")
            for i in range(MAX_AUDIO_PLAYERS)
        ]

        generate_btn.click(
            fn=generate_and_update,
            inputs=[prompt_in, ddim_in, duration_in, samples_in, scale_in],
            outputs=audio_players,
        )
    return demo


def main() -> None:
    """Launch the Gradio interface in the default browser."""
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
