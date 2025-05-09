"""DNNPype/sound.py: Sound synthesis with Scipy."""

from __future__ import annotations

import numpy as np
import scipy.io.wavfile as wav

_samplerate: int = 44100  # Hz
_duration: float = 5.0  # s


def sound_from_partials(
    partials: np.ndarray,
    *,
    base_frequency: float,
    samplerate: int = _samplerate,
    duration: float = _duration,
) -> np.ndarray:
    """Generate sound from partials.

    Inputs
    ------
    partials: numpy.ndarray
        The partials (distribution) to generate sound from.
    base_frequency: float
        The base frequency of the sound.
    samplerate: int
        The sample rate of the sound.
    duration: float
        The duration of the sound in seconds.

    Returns
    -------
    sound: numpy.ndarray
        The generated sound, as a numpy array of int16 values.
    """
    n_partials: int = partials.shape[0]
    amplitude: float = np.iinfo(np.int16).max / 2

    frequencies: np.ndarray = np.array(
        [base_frequency * (i + 1) for i in range(n_partials)]
    )
    time: np.ndarray = np.linspace(
        0, duration, int(samplerate * duration), endpoint=False
    )
    sound: np.ndarray = np.zeros_like(time)

    for i in range(n_partials):
        sound += partials[i] * np.sin(2 * np.pi * frequencies[i] * time)
    sound = amplitude * sound
    return sound.astype(np.int16)


def save_file(
    sound: np.ndarray,
    filename: str,
    *,
    samplerate: int = _samplerate,
) -> None:
    """Save sound to file.

    Inputs
    ------
    sound: numpy.ndarray
        The sound to save.
    filename: str
        The name of the file to save the sound to.
    samplerate: int
        The sample rate of the sound.
    """
    wav.write(filename, samplerate, sound)
    print(f"Saved sound to {filename}")


if __name__ == "__main__":
    base_frequency = 440.0  # A4
    partials = np.array([0.5, 0.3, 0.2])  # Example partials
    sound = sound_from_partials(partials, base_frequency=base_frequency)
    save_file(sound, "a4.wav")

    base_frequency = 69.3  # C#2
    partials = np.array([0.48, 0.99, 0.56, 0.28, 0.25, 0.3, 0.0, 0.24])
    sound = sound_from_partials(partials, base_frequency=base_frequency)
    save_file(sound, "csharp2_bad.wav")

    base_frequency = 69.3  # C#2
    partials = np.array([0.99, 0.96, 0.74, 0.38, 0.27, 0.0, 0.0, 0.05])
    sound = sound_from_partials(partials, base_frequency=base_frequency)
    save_file(sound, "csharp2_good.wav")
