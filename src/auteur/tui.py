"""Full-screen session: output scrolls above, the input bar is pinned at the bottom.

Layout (alternate screen, so the shell disappears while Auteur runs):

    ┌ output log ─ scrollable (wheel, PgUp/PgDn, Shift-↑/↓, End) ┐
    │ live area  ─ spinners, progress, pickers, queued messages   │
    ├─────────────────────────────────────────────────────────────┤
    │ › input bar (completion, history, Esc-Enter newline)        │
    ├─────────────────────────────────────────────────────────────┤
    └ status line                                                 ┘

Commands run on a worker thread so the screen stays live; whatever the user
submits meanwhile is queued. Rich output is captured as ANSI into the log, and
the worker's spinners/questions are routed here through :data:`ui.host`.
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import os
import re
import threading
import time
from dataclasses import dataclass, field

from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (Float, FloatContainer, HSplit, Layout, VSplit, Window)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import (ConditionalProcessor, PasswordProcessor, Processor, Transformation)
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from . import config as cfg
from . import ui

OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")  # hyperlinks etc. — not drawable here
CSI_NONSTYLE = re.compile(r"\x1b\[[0-9;?]*[ABCDEFGHJKSTfhlnsu]")  # cursor moves, clears
MAX_LINES = 6000


# ---- the captured log ---------------------------------------------------------------- #


class LogBuffer(io.TextIOBase):
    """A file Rich writes ANSI into; kept as prompt_toolkit fragments per line."""

    def __init__(self, on_change):
        self.lines: list[list[tuple[str, str]]] = [[]]
        self.plain: list[str] = [""]
        self.version = 0
        self._lock = threading.Lock()
        self._on_change = on_change

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        clean = CSI_NONSTYLE.sub("", OSC.sub("", s))
        added = 0
        with self._lock:
            for style, text, *_ in to_formatted_text(ANSI(clean)):
                parts = text.split("\n")
                for i, part in enumerate(parts):
                    if i:
                        self.lines.append([])
                        self.plain.append("")
                        added += 1
                    if part:
                        self.lines[-1].append((style, part))
                        self.plain[-1] += part
            if len(self.lines) > MAX_LINES:
                drop = len(self.lines) - MAX_LINES
                del self.lines[:drop]
                del self.plain[:drop]
            self.version += 1
        self._on_change(added)
        return len(s)

    def flush(self) -> None:
        pass

    def clear(self) -> None:
        with self._lock:
            self.lines, self.plain = [[]], [""]
            self.version += 1
        self._on_change(0)

    def text(self) -> str:
        with self._lock:
            return "\n".join(self.plain)

    def window(self, height: int, offset: int, extra: list[list[tuple[str, str]]] | None = None
               ) -> tuple[list[tuple[str, str]], int]:
        """Fragments for ``height`` lines ending ``offset`` lines above the bottom.

        Output is top-anchored: it starts at the top of the screen and, once it
        fills the window, the newest lines stay in view. ``extra`` lines (the
        live area) follow the log as if they were its last lines.
        """
        with self._lock:
            lines = self.lines[:-1] if not self.lines[-1] else self.lines  # drop trailing empty line
            lines = lines + (extra or [])
            total = len(lines)
            offset = max(0, min(offset, total - height)) if total > height else 0
            end = total - offset
            view = lines[max(0, end - height):end]
        frags: list[tuple[str, str]] = []
        for i, line in enumerate(view):
            frags.extend(line)
            if i < len(view) - 1:
                frags.append(("", "\n"))
        return frags, offset


def split_lines(frags: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text, *_ in frags:
        for i, part in enumerate(text.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append((style, part))
    return lines


def render_ansi(renderable, width: int) -> list[tuple[str, str]]:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, color_system="truecolor", width=width, theme=ui.theme,
            highlight=False).print(renderable)
    return to_formatted_text(ANSI(OSC.sub("", buf.getvalue()).rstrip("\n")))


# ---- stray output ------------------------------------------------------------------------ #


class StdCapture:
    """While the app owns the screen, nothing else may write to the terminal.

    Libraries (and C code) print to file descriptors 1/2 directly, which would
    scribble over the full-screen layout. We give prompt_toolkit its own handle on
    the terminal, point fds 1/2 at a pipe, and feed whatever arrives into the
    log as dim text.
    """

    def __init__(self, log: LogBuffer):
        self.log = log
        self.tty = None

    def start(self):
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = (os.dup(1), os.dup(2))
        self.tty = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", errors="replace")
        r, self.w = os.pipe()
        os.dup2(self.w, 1)
        os.dup2(self.w, 2)
        self.reader = threading.Thread(target=self._pump, args=(r,), daemon=True, name="auteur-stdio")
        self.reader.start()
        return self.tty

    def _pump(self, r: int) -> None:
        import codecs

        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        while True:
            try:
                chunk = os.read(r, 65536)
            except OSError:
                break
            if not chunk:
                break
            pending += dec.decode(chunk)
            *lines, pending = pending.split("\n")
            for line in lines:
                line = line.split("\r")[-1].rstrip()  # progress bars: keep the final state
                if line.strip():
                    self.log.write(f"\x1b[2m  {line}\x1b[0m\n")
        os.close(r)

    def released(self):
        """Context manager: real terminal on fds 1/2 (for interactive subprocesses)."""
        import contextlib
        import sys

        @contextlib.contextmanager
        def cm():
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(self.saved[0], 1)
            os.dup2(self.saved[1], 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(self.w, 1)
                os.dup2(self.w, 2)

        return cm()

    def stop(self) -> None:
        import sys

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        os.close(self.w)  # last write end → the reader sees EOF
        self.reader.join(timeout=2)
        for fd in self.saved:
            os.close(fd)
        self.tty.close()


# ---- input helpers ------------------------------------------------------------------ #


class Placeholder(Processor):
    def __init__(self, get):
        self.get = get

    def apply_transformation(self, ti):
        if ti.lineno == 0 and not ti.document.text:
            return Transformation(list(ti.fragments) + list(self.get()))
        return Transformation(ti.fragments)


class LogControl(FormattedTextControl):
    def __init__(self, tui: FullScreen):
        super().__init__(tui.log_fragments, focusable=False, show_cursor=False)
        self.tui = tui

    def mouse_handler(self, ev):
        if ev.event_type == MouseEventType.SCROLL_UP:
            self.tui.scroll(3)
            return None
        if ev.event_type == MouseEventType.SCROLL_DOWN:
            self.tui.scroll(-3)
            return None
        return NotImplemented


@dataclass
class Question:
    kind: str  # select | text
    prompt: str
    choices: list[str] = field(default_factory=list)
    index: int = 0
    secret: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    answer: str | None = None


# ---- the app ---------------------------------------------------------------------------- #


class FullScreen(ui.Host):
    def __init__(self, session):
        self.session = session
        self.log = LogBuffer(self._on_log)
        self.offset = 0  # lines scrolled up from the bottom
        self.unseen = 0
        self.live_stack: list = []
        self.question: Question | None = None
        self.queue: list[str] = []
        self.busy = False
        self.worker: threading.Thread | None = None
        self._last_ctrl_c = 0.0
        self.app: Application | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._log_cache: tuple | None = None
        self.capture: StdCapture | None = None
        from .repl import SlashCompleter

        self.input = Buffer(
            completer=SlashCompleter(session),
            complete_while_typing=Condition(lambda: self.question is None),
            history=self._history(),
            multiline=False,
        )

    # -- plumbing ------------------------------------------------------------------- #
    @staticmethod
    def _history():
        try:
            path = cfg.config_path().parent / "history"
            path.parent.mkdir(parents=True, exist_ok=True)
            return FileHistory(str(path))
        except OSError:
            return InMemoryHistory()

    def _on_log(self, added: int) -> None:
        if self.offset and added:
            self.offset += added  # stay where the user scrolled to
            self.unseen += added
        self.invalidate()

    def invalidate(self) -> None:
        if self.app is not None:
            self.app.invalidate()

    def width(self) -> int:
        try:
            return self.app.output.get_size().columns if self.app else 100
        except Exception:  # noqa: BLE001
            return 100

    def height(self) -> int:
        try:
            return self.app.output.get_size().rows if self.app else 40
        except Exception:  # noqa: BLE001
            return 40

    def scroll(self, lines: int) -> None:
        self.offset = max(0, self.offset + lines)
        if self.offset == 0:
            self.unseen = 0
        self.invalidate()

    # -- views ------------------------------------------------------------------------ #
    def log_fragments(self):
        info = self.log_window.render_info
        h = info.window_height if info else max(5, self.height() - 6)
        live = self.live_fragments() if self.has_live() else []
        key = (self.log.version, self.offset, h)
        if not live and self._log_cache and self._log_cache[0] == key:
            return self._log_cache[1]
        extra = [[]] + split_lines(live) if live else None  # a blank line between output and live area
        frags, self.offset = self.log.window(h, self.offset, extra)
        if not live:
            self._log_cache = (key, frags)
        return frags

    def live_fragments(self):
        width = min(self.width(), ui.MAX_WIDTH)
        out: list[tuple[str, str]] = []
        q = self.question
        if q and q.kind == "select":
            out += [("class:q", f"  ? {q.prompt}\n")]
            for i, c in enumerate(q.choices):
                cur = i == q.index
                out += [("class:q.cur" if cur else "class:q.opt", f"  {'❯' if cur else ' '} {c}\n")]
            out += [("class:hint", "    ↑↓ move · enter choose · esc cancel")]
            return out
        if q and q.kind == "text":
            return [("class:q", f"  ? {q.prompt}"), ("class:hint", "   (enter to submit · esc to cancel)")]
        if self.live_stack:
            try:
                out += render_ansi(self.live_stack[-1](), width)
            except Exception as e:  # noqa: BLE001 — a view must never kill the app
                out += [("class:hint", f"  ({type(e).__name__} in live view)")]
        for item in self.queue:
            if out:
                out.append(("", "\n"))
            out += [("class:hint", "  ⧗ queued  "), ("class:queued", item[:200])]
        return out

    def has_live(self) -> bool:
        return bool(self.question or self.live_stack or self.queue)

    def rule(self, top: bool):
        w = self.width()
        if top and self.offset:
            label = f" ↓ {self.unseen} new lines · End " if self.unseen else " ↓ scrolled up · End "
            return [("class:rule", "─" * 2), ("class:rule.note", label), ("class:rule", "─" * max(0, w - len(label) - 2))]
        if top and self.busy and not self.question:
            label = " working · ctrl-c to interrupt "
            return [("class:rule", "─" * 2), ("class:rule.busy", label), ("class:rule", "─" * max(0, w - len(label) - 2))]
        return [("class:rule", "─" * w)]

    def prompt(self):
        if self.question and self.question.kind == "text":
            return [("class:q", "? ")]
        return [("class:prompt", "› ")]

    def placeholder(self):
        q = self.question
        if q and q.kind == "text":
            return [("class:placeholder", "type here…" if not q.secret else "paste — input is hidden")]
        if q:
            return []
        return self.session.placeholder()

    # -- layout ------------------------------------------------------------------------ #
    def build(self, output=None) -> Application:
        self.log_window = Window(LogControl(self), wrap_lines=False, always_hide_cursor=True)
        secret = Condition(lambda: bool(self.question and self.question.secret))
        input_window = Window(
            BufferControl(self.input, input_processors=[
                ConditionalProcessor(PasswordProcessor(char="•"), secret),
                Placeholder(self.placeholder),
            ]),
            height=Dimension(min=1, max=8), wrap_lines=True, dont_extend_height=True,
        )
        body = HSplit([
            self.log_window,
            Window(FormattedTextControl(lambda: self.rule(True)), height=1),
            VSplit([Window(FormattedTextControl(self.prompt), width=2, dont_extend_width=True), input_window]),
            Window(FormattedTextControl(lambda: self.rule(False)), height=1),
            Window(FormattedTextControl(self.session.toolbar), height=1, style="class:status"),
        ])
        root = FloatContainer(body, floats=[
            Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=12, scroll_offset=1)),
        ])
        colorterm = os.environ.get("COLORTERM", "").lower()
        depth = ColorDepth.TRUE_COLOR if colorterm in ("truecolor", "24bit") else ColorDepth.DEPTH_8_BIT
        return Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=self.bindings(),
            style=self.style(),
            full_screen=True,
            mouse_support=Condition(lambda: self.session.settings.mouse),
            refresh_interval=0.12,
            color_depth=depth,
            input=self.session.pt_input,
            output=output or self.session.pt_output,
        )

    @staticmethod
    def style() -> Style:
        return Style.from_dict({
            "prompt": f"{ui.AMBER} bold",
            "placeholder": ui.SMOKE,
            "rule": "#3a3833",
            "rule.note": f"{ui.TEAL}",
            "rule.busy": f"{ui.AMBER}",
            "q": f"{ui.AMBER} bold",
            "q.cur": f"{ui.AMBER} bold",
            "q.opt": ui.INK,
            "hint": ui.SMOKE,
            "queued": f"{ui.INK} italic",
            # Transparent status line (no bg, no reverse) so it matches any terminal theme.
            "status": "noreverse",
            "tb": ui.SMOKE,
            "tb.sep": "#4a4741",
            "tb.film": f"{ui.AMBER} bold",
            "tb.value": ui.INK,
            "tb.on": "#7BD88F",
            "tb.off": ui.ROSE,
            "tb.hint": "#6b675f",
            "completion-menu": f"bg:{ui.NIGHT}",
            "completion-menu.completion": f"bg:{ui.NIGHT} {ui.INK}",
            "completion-menu.completion.current": f"bg:{ui.AMBER} {ui.NIGHT} bold",
            "completion-menu.meta.completion": f"bg:{ui.NIGHT} {ui.SMOKE}",
            "completion-menu.meta.completion.current": f"bg:{ui.AMBER} {ui.NIGHT}",
        })

    def bindings(self) -> KeyBindings:
        kb = KeyBindings()
        selecting = Condition(lambda: bool(self.question and self.question.kind == "select"))
        asking = Condition(lambda: self.question is not None)

        @kb.add("enter")
        def _enter(event):
            buf = self.input
            state = buf.complete_state
            if state and state.current_completion:
                buf.apply_completion(state.current_completion)
                return
            if self.question and self.question.kind == "select":
                q = self.question
                self.answer(q.choices[q.index] if q.choices else None)
                return
            text = buf.text
            if self.question is None and text.strip():
                buf.append_to_history()
            buf.reset()
            self.submit(text)

        @kb.add("escape", "enter")
        def _newline(event):
            if not self.question:
                self.input.insert_text("\n")

        @kb.add("up", filter=selecting)
        def _up(event):
            self.question.index = (self.question.index - 1) % len(self.question.choices)

        @kb.add("down", filter=selecting)
        def _down(event):
            self.question.index = (self.question.index + 1) % len(self.question.choices)

        for n in range(1, 10):
            @kb.add(str(n), filter=selecting)
            def _digit(event, n=n):
                if n <= len(self.question.choices):
                    self.answer(self.question.choices[n - 1])

        @kb.add("escape", filter=asking, eager=True)
        def _cancel(event):
            self.input.reset()
            self.answer(None)

        @kb.add("pageup")
        def _pgup(event):
            self.scroll(max(3, self.height() // 2))

        @kb.add("pagedown")
        def _pgdn(event):
            self.scroll(-max(3, self.height() // 2))

        @kb.add("s-up")
        def _sup(event):
            self.scroll(1)

        @kb.add("s-down")
        def _sdown(event):
            self.scroll(-1)

        @kb.add("end", filter=Condition(lambda: not self.input.text))
        def _end(event):
            self.scroll(-self.offset)

        @kb.add("c-l")
        def _clear(event):
            self.clear()

        @kb.add("c-c")
        def _ctrl_c(event):
            if self.question:
                self.input.reset()
                self.answer(None)
            elif self.busy:
                self.queue.clear()
                self.interrupt()
            elif self.input.text:
                self.input.reset()
            elif time.time() - self._last_ctrl_c < 2:
                event.app.exit()
            else:
                self._last_ctrl_c = time.time()
                ui.console.print(Text("  (press Ctrl-C again to exit)", style="dim"))

        @kb.add("c-d")
        def _ctrl_d(event):
            if not self.input.text and not self.busy:
                event.app.exit()

        return kb

    # -- running commands ----------------------------------------------------------------- #
    def submit(self, text: str) -> None:
        q = self.question
        if q and q.kind == "text":
            self.answer(text)
            return
        if not text.strip():
            return
        if self.offset:
            self.scroll(-self.offset)
        if self.busy:
            self.queue.append(text.strip())
            self.invalidate()
            return
        self.start(text)

    def start(self, text: str) -> None:
        self.busy = True
        self.worker = threading.Thread(target=self._work, args=(text,), daemon=True, name="auteur-command")
        self.worker.start()

    def _work(self, text: str) -> None:
        try:
            ui.console.print()
            ui.console.print(Text.assemble(("› ", f"bold {ui.AMBER}"), (text, f"bold {ui.INK}")))
            self.session.handle(text)
        except KeyboardInterrupt:
            ui.warn("cut! — interrupted")
        finally:
            try:
                self.session.refresh()
            except Exception:  # noqa: BLE001
                pass
            self.busy = False
            self.live_stack.clear()
            if not self.session.running:
                self._exit()
            elif self.queue:
                self.start(self.queue.pop(0))
            self.invalidate()

    def interrupt(self) -> None:
        """Raise KeyboardInterrupt inside the worker (lands at the next bytecode)."""
        t = self.worker
        if t and t.is_alive() and t.ident:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(t.ident), ctypes.py_object(KeyboardInterrupt))

    def _exit(self) -> None:
        if self.app and self.loop:
            self.loop.call_soon_threadsafe(lambda: self.app.exit() if self.app.is_running else None)

    # -- ui.Host ---------------------------------------------------------------------------- #
    def live_push(self, get) -> None:
        self.live_stack.append(get)
        self.invalidate()

    def live_pop(self, get) -> None:
        if get in self.live_stack:
            self.live_stack.remove(get)
        self.invalidate()

    def _ask(self, q: Question) -> str | None:
        self.question = q
        self.invalidate()
        q.done.wait()
        self.question = None
        self.invalidate()
        return q.answer

    def answer(self, value: str | None) -> None:
        q = self.question
        if q is None:
            return
        q.answer = value
        shown = "cancelled" if value is None else ("•" * min(len(value), 12) if q.secret else value)
        ui.console.print(Text.assemble(("  ? ", "accent"), (q.prompt + "  ", "dim"), (shown, "ink")))
        q.done.set()

    def select(self, question, choices, default=None):
        idx = choices.index(default) if default in choices else 0
        return self._ask(Question("select", question, list(choices), idx))

    def text(self, question, secret=False):
        return self._ask(Question("text", question, secret=secret))

    def external(self, fn):
        if not (self.app and self.loop):
            return fn()

        def with_terminal():
            if self.capture is None:
                return fn()
            with self.capture.released():
                return fn()

        async def go():
            return await run_in_terminal(with_terminal, in_executor=True)

        return asyncio.run_coroutine_threadsafe(go(), self.loop).result()

    def clear(self) -> None:
        self.log.clear()
        self.offset = self.unseen = 0

    # -- run ------------------------------------------------------------------------------- #
    def run(self) -> str:
        saved_console, saved_host = ui.console, ui.host
        ui.console = Console(file=self.log, force_terminal=True, color_system="truecolor", theme=ui.theme,
                             highlight=False, width=min(self.width(), ui.MAX_WIDTH))
        ui.host = self
        output = None
        if self.session.pt_output is None:  # a real terminal: fence off stray writes
            from prompt_toolkit.output import create_output

            self.capture = StdCapture(self.log)
            output = create_output(stdout=self.capture.start())
        self.app = self.build(output)

        async def main():
            self.loop = asyncio.get_running_loop()
            ui.fit()
            self.session.welcome()
            self.session.refresh()
            await self.app.run_async()

        try:
            asyncio.run(main())
        finally:
            if self.capture is not None:
                self.capture.stop()
            ui.console, ui.host = saved_console, saved_host
        return self.log.text()
