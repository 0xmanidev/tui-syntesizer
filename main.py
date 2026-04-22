import json
import numpy as np
import sounddevice as sd
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Static, Input, Button
from textual.containers import Horizontal, Vertical, Grid, ScrollableContainer
from textual.binding import Binding
SR        = 44100
BLOCK     = 512
MAX_POLY  = 6
NOTE_KEYS = "zsxdcvgbhnjm"

KEYS_LAYOUT = [
    ("C","z",False), ("C#","s",True),  ("D","x",False),
    ("D#","d",True), ("E","c",False),  ("F","v",False),
    ("F#","g",True), ("G","b",False),  ("G#","h",True),
    ("A","n",False), ("A#","j",True),  ("B","m",False),
]

WAVEFORMS = ["saw", "square", "sine", "triangle"]

PRESETS = {
    "Init":  {"waveform":"saw",      "detune":4.0,  "attack":0.01, "decay":0.2,  "sustain":0.7, "release":0.3,  "cutoff":2000.0, "resonance":0.3, "reverb_mix":0.0, "volume":0.7, "octave":0},
    "Bass":  {"waveform":"square",   "detune":2.0,  "attack":0.01, "decay":0.3,  "sustain":0.6, "release":0.5,  "cutoff":500.0,  "resonance":2.0, "reverb_mix":0.1, "volume":0.8, "octave":-1},
    "Pad":   {"waveform":"sine",     "detune":12.0, "attack":0.8,  "decay":0.5,  "sustain":0.9, "release":1.5,  "cutoff":2000.0, "resonance":0.3, "reverb_mix":0.4, "volume":0.7, "octave":0},
    "Pluck": {"waveform":"triangle", "detune":3.0,  "attack":0.001,"decay":0.15, "sustain":0.1, "release":0.3,  "cutoff":1500.0, "resonance":1.5, "reverb_mix":0.2, "volume":0.7, "octave":0},
}


def note_freq(semitone: int, octave: int) -> float:
    return 261.63 * (2.0 ** (semitone / 12.0)) * (2.0 ** octave)


class Voice:
    def __init__(self, freq: float):
        self.freq       = freq
        self.phase      = 0.0
        self.phase2     = 0.0
        self.age        = 0
        self.release_at: int | None = None
        self.active     = True

    def release(self):
        if self.release_at is None:
            self.release_at = self.age


class Reverb:
    DELAY_TIMES = [0.0297, 0.0371, 0.0411, 0.0437]

    def __init__(self):
        self.buffers = [np.zeros(int(SR * t)) for t in self.DELAY_TIMES]
        self.ptrs    = [0] * len(self.DELAY_TIMES)

    def process(self, x: np.ndarray, mix: float, decay: float = 0.5) -> np.ndarray:
        if mix < 0.01:
            return x
        out = np.zeros_like(x)
        n   = len(self.buffers)
        for i, sample in enumerate(x):
            wet = 0.0
            for j, (buf, p) in enumerate(zip(self.buffers, self.ptrs)):
                d = buf[p]
                wet    += d
                buf[p]  = sample + d * decay * 0.5
                self.ptrs[j] = (p + 1) % len(buf)
            out[i] = sample * (1.0 - mix) + (wet / n) * mix
        return out


class Synth:
    def __init__(self):
        self.waveform   = "saw"
        self.detune     = 4.0
        self.attack     = 0.01
        self.decay      = 0.2
        self.sustain    = 0.7
        self.release    = 0.3
        self.cutoff     = 2000.0
        self.resonance  = 0.3
        self.reverb_mix = 0.0
        self.volume     = 0.7
        self.octave     = 0

        self.voices: dict[str, Voice] = {}
        self._reverb = Reverb()
        self._z1 = 0.0
        self._z2 = 0.0

    def _osc(self, phase: np.ndarray) -> np.ndarray:
        w = self.waveform
        if w == "square":   return np.where(phase < 0.5, 1.0, -1.0)
        if w == "sine":     return np.sin(2.0 * np.pi * phase)
        if w == "triangle": return 4.0 * np.abs(phase - 0.5) - 1.0
        return 2.0 * phase - 1.0  

    def _envelope(self, voice: Voice, frames: int) -> np.ndarray:
        A   = max(1, int(self.attack  * SR))
        D   = max(1, int(self.decay   * SR))
        R   = max(1, int(self.release * SR))
        env = np.zeros(frames)
        age = voice.age
        for i in range(frames):
            if voice.release_at is None:
                if   age < A:     env[i] = age / A
                elif age < A + D: env[i] = 1.0 - (1.0 - self.sustain) * (age - A) / D
                else:             env[i] = self.sustain
            else:
                r = age - voice.release_at
                if r < R: env[i] = self.sustain * (1.0 - r / R)
                else:     voice.active = False
            age += 1
        voice.age = age
        return env

    def _lowpass(self, x: np.ndarray) -> np.ndarray:
        cutoff = float(np.clip(self.cutoff, 20.0, SR / 2.0 - 100.0))
        Q      = float(np.clip(self.resonance, 0.1, 10.0))
        w      = 2.0 * np.pi * cutoff / SR
        cosw   = np.cos(w); sinw = np.sin(w)
        alpha  = sinw / (2.0 * Q)
        b0 = (1.0 - cosw) / 2.0; b1 = 1.0 - cosw; b2 = (1.0 - cosw) / 2.0
        a0 = 1.0 + alpha;         a1 = -2.0 * cosw; a2 = 1.0 - alpha
        b0,b1,b2 = b0/a0, b1/a0, b2/a0
        a1,a2    = a1/a0, a2/a0
        y = np.zeros_like(x); z1, z2 = self._z1, self._z2
        for i in range(len(x)):
            y[i] = b0*x[i] + z1
            z1   = b1*x[i] - a1*y[i] + z2
            z2   = b2*x[i] - a2*y[i]
        self._z1, self._z2 = z1, z2
        return y

    def callback(self, outdata: np.ndarray, frames: int, time_info, status):
        try:
            if not self.voices:
                outdata[:] = 0; return
            mix = np.zeros(frames); dead = []
            for key, voice in self.voices.items():
                if not voice.active:
                    dead.append(key); continue
                p1 = (voice.phase  + np.cumsum(np.full(frames, voice.freq / SR))) % 1.0
                p2 = (voice.phase2 + np.cumsum(np.full(frames, (voice.freq + self.detune) / SR))) % 1.0
                mix += (self._osc(p1) + self._osc(p2)) * 0.5 * self._envelope(voice, frames)
                voice.phase = float(p1[-1]); voice.phase2 = float(p2[-1])
            for k in dead: self.voices.pop(k, None)
            mix /= max(len(self.voices), 1)
            mix  = self._lowpass(mix)
            mix  = self._reverb.process(mix, self.reverb_mix)
            mix  = np.tanh(mix * self.volume)
            outdata[:] = mix.reshape(-1, 1)
        except Exception:
            outdata[:] = 0

    def note_on(self, key: str):
        if key not in NOTE_KEYS: return
        if len(self.voices) >= MAX_POLY and key not in self.voices:
            self.voices.pop(min(self.voices, key=lambda k: self.voices[k].age), None)
        self.voices[key] = Voice(note_freq(NOTE_KEYS.index(key), self.octave))

    def note_off(self, key: str):
        if key in self.voices:
            self.voices[key].release()

    PARAM_KEYS = ("waveform","detune","attack","decay","sustain","release",
                  "cutoff","resonance","reverb_mix","volume","octave")

    def to_dict(self)   -> dict: return {k: getattr(self, k) for k in self.PARAM_KEYS}
    def from_dict(self, d: dict):
        for k in self.PARAM_KEYS:
            if k in d: setattr(self, k, d[k])
        self._z1 = self._z2 = 0.0


engine = Synth()


class SynthUI(App):
    CSS = """
    Screen { background: #0d0d10; color: #c8c4d8; layout: vertical; }

    #title {
        height: 1; background: #18181f; color: #7060e8;
        text-style: bold; padding: 0 2; border-bottom: solid #25253a;
    }
    #status {
        height: 1; background: #18181f; border-top: solid #25253a;
        padding: 0 2; color: #5a5870; dock: bottom;
    }
    Footer { background: #18181f; color: #5a5870; }

    /* Scroll */
    #scroll { height: 1fr; }
    #panels { height: auto; padding: 1 2; }

    /* Section headers */
    .section { color: #7060e8; text-style: bold; height: 1; margin: 1 0 0 0; }

    /* Two-column grid */
    .grid { grid-size: 2; grid-columns: 14 1fr; grid-gutter: 0 1; height: auto; }
    .lbl  { color: #5a5870; height: 3; content-align: left middle; text-style: italic; }

    /* Inputs */
    Input { background: #18181f; border: solid #25253a; color: #c8c4d8; height: 3; }
    Input:focus { border: solid #7060e8; }

    /* Waveform toggle buttons */
    .wave-btn {
        background: #18181f; border: solid #25253a; color: #5a5870;
        height: 3; min-width: 11; margin: 0 1 0 0;
    }
    .wave-btn:hover { background: #2e2a45; color: #c8c4d8; border: solid #7060e8; }
    .wave-active {
        background: #7060e8 !important; border: solid #9080ff !important;
        color: #ffffff !important; text-style: bold;
    }
    #wave-row { height: 3; align: left middle; }

    /* Octave stepper */
    #oct-row  { height: 3; align: left middle; }
    #oct-dn, #oct-up { width: 5; min-width: 5; height: 3; }
    #oct-val  { width: 5; height: 3; content-align: center middle; color: #7060e8; text-style: bold; }

    /* Generic buttons */
    Button {
        background: #18181f; border: solid #25253a; color: #c8c4d8;
        height: 3; min-width: 8; margin: 0 1 0 0;
    }
    Button:hover { background: #7060e8; color: #fff; border: solid #7060e8; }
    Button.-primary { background: #7060e8; border: solid #7060e8; color: #fff; }

    .btn-row { height: 3; align: left middle; margin-bottom: 1; }

    /* Piano */
    #keyboard {
        dock: bottom; height: 7; background: #18181f;
        border-top: solid #25253a; align: center middle; padding: 0 1;
    }
    #key-strip { align: center middle; width: auto; height: 5; }
    .wkey { background: #dedad0; color: #222; width: 6; height: 5; min-width: 6; border: solid #aaa8a0; margin: 0; }
    .bkey { background: #1a1725; color: #888; width: 5; height: 5; min-width: 5; border: solid #35324a; margin: 0; }
    .wkey:hover { background: #fff; color: #7060e8; }
    .bkey:hover { background: #2e2a45; color: #a090ff; }
    .key-active { background: #7060e8 !important; color: #fff !important; border: solid #9080ff !important; }
    """

    BINDINGS = [
        Binding("q",    "quit",      "Quit"),
        Binding("up",   "octave_up", "Oct +"),
        Binding("down", "octave_dn", "Oct -"),
    ]

    INPUTS = {
        "detune":  ("detune",     0.0,  50.0),
        "attack":  ("attack",     0.0,   4.0),
        "decay":   ("decay",      0.0,   4.0),
        "sustain": ("sustain",    0.0,   1.0),
        "release": ("release",    0.0,   8.0),
        "cutoff":  ("cutoff",    20.0, 20000.0),
        "res":     ("resonance",  0.1,  10.0),
        "reverb":  ("reverb_mix", 0.0,   1.0),
        "volume":  ("volume",     0.0,   1.0),
    }

    def compose(self) -> ComposeResult:
        yield Static("  SynthUI", id="title")

        with ScrollableContainer(id="scroll"):
            with Vertical(id="panels"):

                
                yield Static("--- OSC", classes="section")
                with Grid(classes="grid"):
                    yield Static("Waveform",  classes="lbl")
                    with Horizontal(id="wave-row"):
                        for w in WAVEFORMS:
                            b = Button(w.capitalize(), id=f"wave_{w}", classes="wave-btn")
                            if w == engine.waveform:
                                b.add_class("wave-active")
                            yield b
                    yield Static("Detune Hz", classes="lbl")
                    yield Input("4.0", id="detune")
                    yield Static("Octave",    classes="lbl")
                    with Horizontal(id="oct-row"):
                        yield Button("-", id="oct-dn")
                        yield Static(str(engine.octave), id="oct-val")
                        yield Button("+", id="oct-up")

                yield Static("--- ENV", classes="section")
                with Grid(classes="grid"):
                    yield Static("Attack s",  classes="lbl"); yield Input("0.01", id="attack")
                    yield Static("Decay s",   classes="lbl"); yield Input("0.2",  id="decay")
                    yield Static("Sustain",   classes="lbl"); yield Input("0.7",  id="sustain")
                    yield Static("Release s", classes="lbl"); yield Input("0.3",  id="release")

                yield Static("--- FILTER / FX", classes="section")
                with Grid(classes="grid"):
                    yield Static("Cutoff Hz",  classes="lbl"); yield Input("2000", id="cutoff")
                    yield Static("Resonance",  classes="lbl"); yield Input("0.3",  id="res")
                    yield Static("Reverb Mix", classes="lbl"); yield Input("0.0",  id="reverb")
                    yield Static("Volume",     classes="lbl"); yield Input("0.7",  id="volume")

                yield Static("--- PRESETS", classes="section")
                with Horizontal(classes="btn-row"):
                    for name in PRESETS:
                        yield Button(name, id=f"pre_{name}")
                    yield Button("Save", id="save", variant="primary")
                    yield Button("Load", id="load", variant="primary")

        with Horizontal(id="keyboard"):
            with Horizontal(id="key-strip"):
                for note_name, kbd_key, is_black in KEYS_LAYOUT:
                    b = Button(f"{note_name}\n{kbd_key}", id=f"key_{kbd_key}")
                    b.add_class("bkey" if is_black else "wkey")
                    yield b

        yield Static(self._status(), id="status")
        yield Footer()


    def on_mount(self):
        try:
            self._stream = sd.OutputStream(
                samplerate=SR, channels=1, blocksize=BLOCK,
                callback=engine.callback, dtype="float32",
            )
            self._stream.start()
            self.notify("Audio ready", severity="information")
        except Exception as e:
            self.notify(f"Audio error: {e}", severity="error")
        self.set_interval(0.08, self._tick)

    def on_unmount(self):
        if hasattr(self, "_stream"):
            try: self._stream.stop(); self._stream.close()
            except Exception: pass


    def _status(self) -> str:
        return (f"OCT {engine.octave:+d}  |  voices {len(engine.voices)}/{MAX_POLY}  |  "
                f"{engine.waveform.upper()}  |  cutoff {int(engine.cutoff)} Hz")

    def _tick(self):
        try:
            self.query_one("#status", Static).update(self._status())
            for _, kbd_key, _ in KEYS_LAYOUT:
                btn = self.query_one(f"#key_{kbd_key}", Button)
                if kbd_key in engine.voices: btn.add_class("key-active")
                else:                        btn.remove_class("key-active")
        except Exception:
            pass


    def on_key(self, event):
        if event.key in NOTE_KEYS:
            engine.note_on(event.key)
            self.set_timer(0.4, lambda k=event.key: engine.note_off(k))
            event.stop()


    def on_button_pressed(self, event):
        bid = event.button.id or ""

        if bid == "oct-up":
            engine.octave = min(engine.octave + 1, 4)
            self._refresh_octave()

        elif bid == "oct-dn":
            engine.octave = max(engine.octave - 1, -4)
            self._refresh_octave()

        elif bid.startswith("wave_"):
            w = bid[5:]
            engine.waveform = w
            self._refresh_waveform()

        elif bid.startswith("key_"):
            key = bid[4:]
            engine.note_on(key)
            self.set_timer(0.4, lambda k=key: engine.note_off(k))

        elif bid.startswith("pre_"):
            name = bid[4:]
            if name in PRESETS:
                engine.from_dict(PRESETS[name])
                self._sync_ui()
                self.notify(f"Loaded: {name}", severity="information")

        elif bid == "save":
            try:
                Path("preset.json").write_text(json.dumps(engine.to_dict(), indent=2))
                self.notify("Saved to preset.json", severity="information")
            except Exception as e:
                self.notify(f"Save failed: {e}", severity="error")

        elif bid == "load":
            p = Path("preset.json")
            if p.exists():
                try:
                    engine.from_dict(json.loads(p.read_text()))
                    self._sync_ui()
                    self.notify("Preset loaded", severity="information")
                except Exception as e:
                    self.notify(f"Load failed: {e}", severity="error")
            else:
                self.notify("No preset.json found", severity="warning")


    def on_input_changed(self, event):
        iid = event.input.id
        if iid not in self.INPUTS: return
        val = event.value.strip()
        if not val: return
        try:
            attr, lo, hi = self.INPUTS[iid]
            setattr(engine, attr, float(np.clip(float(val), lo, hi)))
        except (ValueError, TypeError):
            pass


    def _refresh_octave(self):
        try: self.query_one("#oct-val", Static).update(str(engine.octave))
        except Exception: pass

    def _refresh_waveform(self):
        """Highlight the active waveform button, unhighlight the rest."""
        for w in WAVEFORMS:
            try:
                btn = self.query_one(f"#wave_{w}", Button)
                if w == engine.waveform: btn.add_class("wave-active")
                else:                    btn.remove_class("wave-active")
            except Exception:
                pass

    def _sync_ui(self):
        """Push all engine values back into UI after loading a preset."""
        try:
            self._refresh_waveform()
            for iid, (attr, _, _) in self.INPUTS.items():
                self.query_one(f"#{iid}", Input).value = str(round(getattr(engine, attr), 4))
            self._refresh_octave()
        except Exception as e:
            self.notify(f"UI sync error: {e}", severity="warning")

    def action_octave_up(self):
        engine.octave = min(engine.octave + 1, 4)
        self._refresh_octave()

    def action_octave_dn(self):
        engine.octave = max(engine.octave - 1, -4)
        self._refresh_octave()


if __name__ == "__main__":
    SynthUI().run()