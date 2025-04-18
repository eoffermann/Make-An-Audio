#!/usr/bin/env python3
"""Text-to-Audio Gradio UI (`app.py`).

This script launches an interactive **Gradio** web application that turns text
prompts into audio clips *on-demand*.

It wires together three high-level components:

1. **Diffusion Sampler** - A pre-trained latent-diffusion model that converts a
   text embedding into a mel-spectrogram-like latent representation.
2. **Neural Vocoder (BigVGAN)** - Translates the latent representation into a
   time-domain waveform at :data:`SAMPLE_RATE`.
3. **Gradio Front-End** - Provides a simple web UI for prompt entry and audio
   preview/downloading.  Nothing is written to disk unless the user explicitly
   clicks each widget’s *Download* button.

Execution is self-contained and *stateless*: no caching, no temp files, and no
server-side persistence.  All heavy lifting happens in GPU memory (*if* the
imported ``device`` points at CUDA) and gets released when the app exits.

Example
-------
Run the application from the command line:

```bash
$ python app.py          # or ./app.py after chmod +x
```

The default browser should automatically open to
``http://127.0.0.1:7860`` showing the UI.
"""
from __future__ import annotations

from typing import List

import gradio as gr
import numpy as np  # NumPy is required by Gradio when ``type="numpy"``.

from gen_wav import SAMPLE_RATE, device, gen_wav, initialize_model  # type: ignore
from vocoder.bigvgan.models import VocoderBigVGAN  # type: ignore

# ---------------------------------------------------------------------------
# Configuration & Model Initialisation
# ---------------------------------------------------------------------------

#: Maximum number of *simultaneous* ``gr.Audio`` preview widgets shown.
MAX_AUDIO_PLAYERS: int = 10

# Instantiate the diffusion sampler and the neural vocoder **once** at import
# time so that we pay model-loading latency only on application start-up.
SAMPLER = initialize_model(
    config="configs/text_to_audio/txt2audio_args.yaml",
    ckpt="useful_ckpts/maa1_full.ckpt",
)
VOCODER = VocoderBigVGAN("useful_ckpts/bigvgan", device=device)


# ---------------------------------------------------------------------------
# Generation Callback
# ---------------------------------------------------------------------------

def generate_and_update(
    prompt: str,
    ddim_steps: int,
    duration: int,
    n_samples: int,
    scale: float,
) -> List[gr.update]:
    """Generate *n_samples* audio clips and create Gradio update payloads.

    Args:
        prompt: Natural-language description of the desired sound.
        ddim_steps: Number of DDIM inference steps for the diffusion sampler.
        duration: Desired clip length in **seconds**.
        n_samples: Number of distinct clips to synthesize (≤ 10 ``MAX_AUDIO_PLAYERS``).
        scale: Classifier-free guidance scale.

    Returns:
        A list of :class:`gr.update` objects (length == ``MAX_AUDIO_PLAYERS``)
        where each entry either:

        * reveals an audio player with ``(sample_rate, waveform)`` *tuple* for
          Gradio to render, **or**
        * hides the widget (``visible=False``) if no clip exists for that slot.

    Note:
        *Generated waveforms* are kept **in-memory only**.  Downloading is left
        to the built-in feature of each ``gr.Audio`` component.

    TODO:
        • Catch and surface ``RuntimeError`` if GPU memory is insufficient.
        • Validate *duration* and *ddim_steps* against model limitations.
    """

    # ---------------------------------------------------------------------
    # Waveform synthesis - returns a ``List[np.ndarray]`` with values in [-1, 1].
    # ---------------------------------------------------------------------
    wavs = gen_wav(
        sampler=SAMPLER,
        vocoder=VOCODER,
        prompt=prompt,
        ddim_steps=ddim_steps,
        scale=scale,
        duration=duration,
        n_samples=n_samples,
    )

    # ---------------------------------------------------------------------
    # Gradio expects (sample_rate, waveform) when ``type='numpy'``.  Build a
    # fixed-length list so that every output target is accounted for.
    # ---------------------------------------------------------------------
    updates: List[gr.update] = []
    for i in range(MAX_AUDIO_PLAYERS):
        if i < len(wavs):
            updates.append(
                gr.update(value=(SAMPLE_RATE, wavs[i]), visible=True)
            )
        else:
            updates.append(gr.update(value=None, visible=False))

    return updates


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Compose the Gradio Blocks and return the root component.

    The layout is kept intentionally minimal:

    * **Input panel** - Prompt, DDIM steps, duration, sample count, guidance.
    * **Generate** button - Triggers diffusion + vocoder inference.
    * **Preview & Download** - Dynamically reveals up to
      :data:`MAX_AUDIO_PLAYERS` audio players.
    """

    with gr.Blocks() as demo:
        gr.Markdown("## Text-to-Audio Generator 🎙️")

        # ------------------------- Input Widgets ------------------------ #
        prompt_in = gr.Textbox(
            label="Prompt",
            value="a bird chirps",
            placeholder="Describe the sound you want to generate…",
        )
        ddim_in = gr.Slider(label="DDIM Steps", minimum=1, maximum=500, value=100)
        duration_in = gr.Slider(
            label="Duration (s)", minimum=1, maximum=60, value=10
        )
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

        # ------------------------ Output Widgets ------------------------ #
        gr.Markdown("### Preview & Download")
        audio_players = [
            gr.Audio(label=f"Clip {i + 1}", visible=False, type="numpy")
            for i in range(MAX_AUDIO_PLAYERS)
        ]

        # ------------------------- Event Binding ------------------------ #
        generate_btn.click(
            fn=generate_and_update,
            inputs=[prompt_in, ddim_in, duration_in, samples_in, scale_in],
            outputs=audio_players,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Gradio interface in the user’s default browser."""

    # `inbrowser=True` attempts to open the local server URL automatically.
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()