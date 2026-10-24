# 🎬 Auteur

**An AI film director in your terminal.** Pitch an idea; Auteur writes the treatment,
plans the shots, casts and storyboards them, shoots every shot with **LTX-2.5** and
cuts the film — in the spirit of OpenArt's Director mode, but local-first and in your shell.

```
 idea ─▶ treatment ─▶ shot list ─▶ cast sheets ─▶ storyboard ─▶ shots ─▶ final cut
         └──────── ChatGPT via openai-oauth ────────┘            LTX-2.5      ffmpeg
                   (gpt-5.x writes, gpt-image draws)        (layer-streamed)
```

- **Director** — your ChatGPT sign-in through [openai-oauth](https://github.com/EvanZhouDev/openai-oauth):
  no API key, best available model picked automatically.
- **Camera** — [LTX-2.5](https://ltx.io/model/ltx-2-5) image-to-video *with synchronized audio*
  (dialogue in the prompt is spoken). The 22B DiT, the Gemma 4 text encoder and the text
  connectors are **streamed from disk one block at a time, AirLLM-style**, so it runs on a 16 GB laptop.
- **Consistency** — character reference sheets feed every keyframe; each shot is
  image-to-video from its keyframe, with a deterministic seed.
- **Edit** — notes in plain English ("make S03 a slow push-in and cut S05") update the plan;
  only the shots that changed get re-boarded or re-shot.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/auteur/main/install.sh | sh
```

The installer sets up [uv](https://docs.astral.sh/uv/) if you don't have it, plus ffmpeg and
Node.js (via Homebrew, apt or dnf). It then installs `auteur` as an isolated tool in
`~/.local/bin`, with Python 3.12, torch and the pinned diffusers commit that has LTX-2.5.

| env var | effect |
|---|---|
| `AUTEUR_REPO` / `AUTEUR_REF` | git repo and branch/tag to install from |
| `AUTEUR_SOURCE` | install from a local path, `git+…` URL or tarball instead |
| `AUTEUR_LITE=1` | skip the local LTX engine (render with `animatic` or `ltx-pipelines`) |
| `AUTEUR_NO_DEPS=1` | don't install ffmpeg / node |
| `AUTEUR_NO_MODIFY_PATH=1` | don't edit your shell profile |

From a checkout, run `./install.sh`. To uninstall: `uv tool uninstall auteur`.

## The session

Run `auteur`. It opens full-screen, like Claude Code or Codex: output starts at the top and
flows down, the input bar stays pinned at the bottom, and your shell comes back untouched when
you exit.

```
  …output: treatment, shot list, storyboard thumbnails, renders…
  ⠹ Boarding 3/8 shots…                 ← live area: spinners, progress, pickers, queued messages
──────────────────────────────────────────────────────────────────────────────
› make @S03 a slow push-in▌
──────────────────────────────────────────────────────────────────────────────
 ◆ the-door-below  3/8 shot  ·  director ● gpt-6-luna  ·  camera airstream · draft  ·  16:9
```

Type plain text or slash commands:

```
› a lighthouse keeper finds a door in the sea        ← an idea starts a film
› make @S03 a slow push-in and lose the narration    ← a note revises it
› /roll                                              ← board → shoot → cut
```

```
FILM     /new [idea] [-t tpl] [-d sec] [-a ratio]   /open <film>   /projects   /templates
         /pitch  /plan  /cast  /board [S01…]  /shots  /show [S03]
CAMERA   /shoot [S01…] [-b backend] [-q quality] [-f]   /cut [--music file]   /roll   /play
         /backend  /quality  /aspect
SETUP    /login [chatgpt|hf]  /status  /model  /fetch  /doctor  /proxy  /config
SESSION  /help  /clear  /exit
```

Other session features:

- **Tab completion** with descriptions for commands, their arguments, film names, shot ids
  and models. Type `@` to mention a shot.
- **Status line** at the bottom: open film and progress, director model and whether it's
  online, camera settings.
- **Keys:** Esc then Enter inserts a newline. Ctrl-C interrupts the running step; on an empty
  prompt, press it twice to quit. Ctrl-D exits, Ctrl-L clears. History is kept in
  `~/.config/auteur/history`.
- **Scrolling:** mouse wheel, PageUp/PageDown or Shift-↑/↓; End jumps back to the newest output.
  The mouse wheel captures the mouse, so hold ⌥ (iTerm2) or Fn (Terminal.app) to select text,
  or turn it off with `/config mouse false`.
- **Keep typing:** anything you submit while a step is running is queued and runs next.
- **Resuming:** `auteur -c` reopens your last film; `auteur -p <film>` opens a specific one.

### First run

```
/login            → ChatGPT: your director (writing + storyboards), via openai-oauth
/login hf         → Hugging Face token, only needed to download LTX-2.5
/fetch            → ~71 GB of weights + fp8 layer cache
```

Before `/fetch`, accept the license on <https://huggingface.co/Lightricks/LTX-2.5-Diffusers>. It is the
LTX-2 Community License, free under $10M annual revenue. `/fetch` downloads only the files the
pipeline uses: the repo ships two transformer shardings and duplicate connectors, and a blind
download would pull about 140 GB. It then checks every tensor name against the model before building
the fp8 layer cache.

Without the weights you can still make the whole film as an animatic: `/backend animatic`.

### Scripting

Every step is also a plain subcommand: `auteur make "idea" -b animatic`, `auteur shoot -s S02`,
`auteur revise "…"`, `auteur cut -o`, `auteur doctor`, … (`auteur --help`). Piped input also
works: `printf '/open my-film\n/cut\n' | auteur`.

## Pipeline

| step | command | what happens |
|---|---|---|
| ✦ pitch | `/pitch` | title, logline, synopsis, style bible, cast, locations |
| ▤ plan | `/plan` | shot list: size, lens, move, action, dialogue, sound, keyframe + LTX prompts |
| ☺ cast | `/cast` | 3-view reference sheet per character |
| ▦ board | `/board S01 S04` | one keyframe per shot, conditioned on the cast sheets |
| ● shoot | `/shoot -q standard` | LTX-2.5 per shot (`-b animatic` for an instant preview) |
| ✂ cut | `/cut --music bed.mp3` | conform, crossfades, title & end cards, music bed |
| ✎ revise | just type a note | the director edits the plan; only stale shots are redone |

Quick-start formats (`/templates`): short-film, trailer, music-video, product-ad, micro-drama,
brand-film, explainer, ugc-ad, anime, real-estate. Aspects: `16:9 9:16 1:1 4:3 2.39:1`.

## Rendering backends

| backend | where | how |
|---|---|---|
| `airstream` *(default)* | Mac (MPS), CUDA, CPU | diffusers LTX-2.5 + Auteur's AirLLM-style layer streamer |
| `ltx-pipelines` | NVIDIA | Lightricks' official `ltx_pipelines.distilled` with `--offload disk --quantization fp8-cast`. Set `/config ltx2_repo /path/to/LTX-2` and `/config ltx_checkpoints /path/to/LTX-2.5` |
| `animatic` | anywhere | ffmpeg Ken Burns moves over the keyframes, matched to each shot's camera move. Renders in seconds |

Quality presets (`-q`): `draft` 768×448, `standard` 1024×576, `hq` 1920×1088 two-stage
(half-res pass → ×2 latent upsampler → refine). All shots use the 8-step distilled schedule.

### How AirStream works

Like AirLLM, but for a video diffusion model:

1. The big modules are built on PyTorch's `meta` device, so they use no memory.
2. A forward-pre-hook on each repeated block reads that block's file in one sequential read,
   dequantizes it on a thread pool, moves it to the GPU and runs the block. A forward-hook then
   drops it.
3. A background thread prefetches the *next* block while the current one computes. The call order
   is learned on the first pass.
4. `/fetch` pre-splits the weights into one file per block, with per-row **fp8** for Linear
   weights. This halves the bytes read on every denoising step, much like AirLLM's compression option.
5. The text encoder streams **once per batch of prompts**, not once per shot. It is then released
   before the DiT starts.

Streaming produces bit-identical output to the stock diffusers pipeline (see
`tests/test_ltx_stream.py`). What stays in memory: about 2 DiT blocks (≈0.9 GB each in bf16),
the VAEs, the vocoder and activations.

**Speed.** Each denoising step re-reads the whole DiT from disk (18 GB with the fp8 cache), because
it doesn't fit in memory on a 16 GB machine. Measured on a 16 GB M5 MacBook Air with the real
weights:

| stage | time |
|---|---|
| load the pipeline | 11 s |
| text encoder (12B, streamed once per batch of prompts) | 53 s |
| one DiT step (22B, 48 blocks streamed) | 31 s at 256×256 |
| video + audio decode | 8 s |

A draft shot runs 8 steps, and GPU work grows with resolution, so expect a few minutes per shot.
Close other memory-hungry apps while shooting: if the machine starts swapping, a step can take
many times longer. Preview with `-b animatic`, then shoot.

## Configuration

`/config` shows the settings; `/config <key> <value>` sets one (stored in
`~/.config/auteur/config.json`). Any key can be overridden with `AUTEUR_<KEY>`. Useful keys:
`text_model`, `image_model`, `reasoning_effort`, `backend`, `quality`, `aspect`, `device`,
`weights`, `layer_cache`, `compress`, `home` (default `~/Movies/Auteur`).

Each film is a folder: `project.json` (every decision and render state), `cast/`, `board/`,
`clips/`, `cut/`, and `<slug>.mp4`.

## Development

```bash
uv sync --extra render   # dev environment
uv run pytest            # 21 tests, ~1.5 min
```

- `tests/test_ltx_stream.py` builds a tiny random LTX-2 pipeline in the real diffusers layout.
  It checks that layer-streamed image-to-video and text-to-video match `from_pretrained` exactly,
  and covers the fp8 cache, two-stage HQ and mp4 + audio output.
- `tests/test_repl.py` drives the interactive session through prompt_toolkit: ideas, notes,
  slash commands, errors, completion and resume.
- `tests/test_e2e.py` runs the whole CLI (`make`, `revise`, `shoot`, `cut`, the views) against
  a fake openai-oauth server, including the repair path for invalid JSON from the director.

## Notes

- openai-oauth is unofficial and not affiliated with OpenAI. Your ChatGPT credentials stay in
  `~/.codex`. Usage counts against your ChatGPT plan.
- LTX-2.5 weights are gated and licensed by Lightricks; read the license before commercial use.
