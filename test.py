import pytest
import numpy as np
from pathlib import Path
import json
import main
from main import Synth, Voice, note_freq, PRESETS, SynthUI
from textual.widgets import Static


def test_note_frequency():
    """Verify semitone to frequency conversion."""
    assert round(note_freq(0, 0), 2) == 261.63
    assert round(note_freq(0, 1), 2) == 523.26

def test_synth_preset_loading():
    """Test if engine correctly adopts dictionary values."""
    s = Synth()
    s.from_dict(PRESETS["Bass"])
    assert s.waveform == "square"
    assert s.octave == -1
    assert s.volume == 0.8

def test_voice_lifecycle():
    """Ensure voices activate and enter release state correctly."""
    v = Voice(440.0)
    assert v.active is True
    assert v.release_at is None
    v.age = 100
    v.release()
    assert v.release_at == 100

def test_engine_polyphony():
    """Check that the engine respects MAX_POLY (6)."""
    s = Synth()
    for k in "zsxdcvg":
        s.note_on(k)
    assert len(s.voices) == 6
    assert "z" not in s.voices

@pytest.mark.asyncio
async def test_ui_octave_increment():
    """Test that pressing the octave up button updates the engine and UI."""
    main.engine.octave = 0
    app = SynthUI()
    async with app.run_test() as pilot:
        assert app.query_one("#oct-val", Static)._Static__content == "0"
        await pilot.click("#oct-up")
        assert app.query_one("#oct-val", Static)._Static__content == "1"
        assert main.engine.octave == 1

@pytest.mark.asyncio
async def test_ui_waveform_selection():
    """Test that clicking a waveform button updates the active class."""
    app = SynthUI()
    async with app.run_test() as pilot:
        await pilot.click("#wave_sine")
        assert "wave-active" in app.query_one("#wave_sine").classes
        assert "wave-active" not in app.query_one("#wave_square").classes

@pytest.mark.asyncio
async def test_keyboard_binding():
    """Test the octave_up action updates engine and UI."""
    main.engine.octave = 0
    app = SynthUI()
    async with app.run_test() as pilot:
        await app.run_action("octave_up")
        await pilot.pause()
        assert app.query_one("#oct-val", Static)._Static__content == "1"
        assert main.engine.octave == 1

@pytest.mark.asyncio
async def test_save_functionality(tmp_path):
    """Test that save creates preset.json."""
    app = SynthUI()
    async with app.run_test() as pilot:
        save_btn = app.query_one("#save")
        await pilot.pause()
        save_btn.press()
        await pilot.pause()
        p = Path("preset.json")
        assert p.exists()
        p.unlink()

if __name__ == "__main__":
    SynthUI().run()