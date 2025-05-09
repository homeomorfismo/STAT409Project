"""DNNPype/appSound.py: Driver for sound synthesis with Scipy."""

from __future__ import annotations
from typing import Optional, List, Callable, Tuple
from enum import Enum

import os
import argparse
import rich as r
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import polars as pl
import plotly.graph_objects as go

# local imports
import sound as sound


###############################################################################
# Constants, enums, and globas
###############################################################################
class PartialsTypes(Enum):
    """Enum for partials types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOG = "log"

    def __str__(self) -> str:
        return self.value


# Lazy way to avoid magic numbers
_mul: dict[str, tuple[float, float]] = {
    "exp": (0.5, 1.5),
    "lin": (0.01, np.tan(np.pi / 2 - 0.01)),
    "log": (0.1, 0.9),
}
_add: dict[str, tuple[float, float]] = {
    "exp": (0.0, np.exp(1.0)),
    "lin": (0.0, 100.0),
    "log": (np.exp(0.5), np.exp(1.5)),
}


###############################################################################
# Auxiliary functions
###############################################################################
def _argparse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sound synthesis with Scipy.")
    # Required arguments
    parser.add_argument(
        "--frequency",
        type=float,
        default=440.0,
        help="Fundamental frequency of the sound in Hz.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Directory to save the output files.",
        required=True,
    )
    # Optional arguments
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the sound in seconds.",
        required=False,
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=44100,
        help="Sample rate of the sound in Hz.",
        required=False,
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of (random) samples to generate.",
        default=5,
        required=False,
    )
    _str_out: str = (
        "Output CSV file name (without .csv extension)."
        " If not provided, the name will be generated from previous arguments."
    )
    parser.add_argument(
        "--output",
        type=str,
        help=_str_out,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save wav files of the generated samples.",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--plot-samples",
        action="store_true",
        help="Plot the generated samples.",
        default=False,
        required=False,
    )
    return parser.parse_args()


def _handle_output_dir(
    *,
    output_dir: str,
) -> None:
    """Create the output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        r.print(
            f"[bold green]Output directory [/bold green]"
            f"[cyan]{output_dir}[/cyan] "
            f"[bold green]created.[/bold green]"
        )
    else:
        r.print(
            f"[bold yellow]Output directory [/bold yellow]"
            f"[cyan]{output_dir}[/cyan] "
            f"[bold yellow]already exists.[/bold yellow]"
        )


def _print_args(
    *,
    args: argparse.Namespace,
) -> None:
    """Print the command line arguments."""
    r.print(
        f"[bold blue]Interactive sound generation and rating\n[/bold blue]"
        f"[bold blue]=======================================\n[/bold blue]"
        f"[bold]\tBase frequency: [/bold] {args.frequency} Hz\n"
        f"[bold]\tDuration: [/bold] {args.duration} s\n"
        f"[bold]\tSample rate: [/bold] {args.samplerate} Hz\n"
        f"[bold]\tNumber of samples: [/bold] {args.samples}\n"
        f"[bold]\tOutput directory: [/bold] {args.output_dir}\n"
        f"[bold]\tOutput summary file: [/bold] {args.output}\n"
        f"[bold]\tSave samples: [/bold] {args.save_samples}\n"
        f"[bold blue]=======================================\n\n[/bold blue]"
    )


###############################################################################
# Partials distribution functions
###############################################################################
def _exponentialPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute exponential partials."""
    slicing_idx: list[int] = [1]
    mult, shift = np.split(theta, slicing_idx, axis=0)
    _fun = lambda x: (np.exp(8 - x) + shift[0]) * np.exp(freq * np.log(mult))
    partials = np.array([_fun(i) for i in range(1, 9)]).flatten()
    partials = partials / np.max(partials, axis=0, keepdims=True)
    return partials


def _linearPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute linear (relu) partials."""
    slicing_idx: list[int] = [1]
    mult, shift = np.split(theta, slicing_idx, axis=0)
    _fun = lambda x: mult * (8 * freq - freq * x) + shift[0]
    partials = np.array([_fun(i) for i in range(1, 9)]).flatten()
    partials = partials / np.max(partials, axis=0, keepdims=True)
    return partials


def _logPartials(freq: float, theta: np.ndarray) -> np.ndarray:
    """Compute log partials."""
    slicing_idx: list[int] = [1]
    mult, shift = np.split(theta, slicing_idx, axis=0)
    _fun = lambda x: np.log(9 - x) * (1 + shift[0]) + np.log(freq * mult)
    partials = np.array([_fun(i) for i in range(1, 9)]).flatten()
    partials = partials / np.max(partials, axis=0, keepdims=True)
    return partials


###############################################################################
# Random parameter generation functions
###############################################################################
def _gen_theta_row(
    *,
    n_samples: int,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Generate random parameters for exponential partials."""
    return np.random.uniform(min_val, max_val, size=(n_samples,))


def _gen_partials(
    *,
    partials_type: PartialsTypes,
    n_samples: int,
    freq: float,
) -> List[np.ndarray]:
    """Generate random partials."""
    if partials_type == PartialsTypes.EXPONENTIAL:
        theta_mul = _gen_theta_row(
            n_samples=n_samples,
            min_val=_mul["exp"][0],
            max_val=_mul["exp"][1],
        )
        theta_add = _gen_theta_row(
            n_samples=n_samples,
            min_val=_add["exp"][0],
            max_val=_add["exp"][1],
        )
        partial_gen: Callable = _exponentialPartials
    elif partials_type == PartialsTypes.LINEAR:
        theta_mul = _gen_theta_row(
            n_samples=n_samples,
            min_val=_mul["lin"][0],
            max_val=_mul["lin"][1],
        )
        theta_add = _gen_theta_row(
            n_samples=n_samples,
            min_val=_add["lin"][0],
            max_val=_add["lin"][1],
        )
        partial_gen: Callable = _linearPartials
    elif partials_type == PartialsTypes.LOG:
        theta_mul = _gen_theta_row(
            n_samples=n_samples,
            min_val=_mul["log"][0],
            max_val=_mul["log"][1],
        )
        theta_add = _gen_theta_row(
            n_samples=n_samples,
            min_val=_add["log"][0],
            max_val=_add["log"][1],
        )
        partial_gen: Callable = _logPartials
    else:
        raise ValueError(f"Unknown partials type: {partials_type}")
    partials = []
    theta = np.vstack((theta_mul, theta_add)).T
    for i in range(theta.shape[0]):
        partials.append(partial_gen(freq=freq, theta=theta[i]))
    return partials


###############################################################################
# Handler rating functions
###############################################################################
def _play_sound(
    *,
    sound: np.ndarray,
    samplerate: int,
) -> None:
    """Play sound using sounddevice."""
    sd.play(sound, samplerate)
    sd.wait()


def _get_user_rating() -> float:
    """Get user rating for the sound."""
    rating: float = -1.0
    while rating < 0.0 or rating > 100.0:
        try:
            rating = float(input("Rate the sound (0-100): "))
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 100.")
    return rating


def _append_rating(
    output_dict: dict,
    *,
    rating: float,
    partials_type: PartialsTypes,
    partials: np.ndarray,
) -> None:
    """Append rating to the output dictionary."""
    output_dict["rating"].append(rating)
    output_dict["partials_type"].append(partials_type)
    _partials = partials.tolist()
    for i, partial in enumerate(_partials):
        output_dict[f"partial{i+1}"].append(partial)


###############################################################################
# Main function
###############################################################################


def main() -> None:
    """Run with 'classify_samples' command."""
    args = _argparse()
    _handle_output_dir(output_dir=args.output_dir)
    if args.output is None:
        args.output = (
            f"sound_{args.frequency}Hz"
            f"_{args.duration}s_{args.samplerate}Hz_{args.samples}samples"
        )
        args.output = args.output.replace(".", "_")
    _print_args(args=args)

    # Generate random partials
    exp_partials = _gen_partials(
        partials_type=PartialsTypes.EXPONENTIAL,
        n_samples=args.samples,
        freq=args.frequency,
    )
    exp_partials = [(p, PartialsTypes.EXPONENTIAL) for p in exp_partials]

    lin_partials = _gen_partials(
        partials_type=PartialsTypes.LINEAR,
        n_samples=args.samples,
        freq=args.frequency,
    )
    lin_partials = [(p, PartialsTypes.LINEAR) for p in lin_partials]

    log_partials = _gen_partials(
        partials_type=PartialsTypes.LOG,
        n_samples=args.samples,
        freq=args.frequency,
    )
    log_partials = [(p, PartialsTypes.LOG) for p in log_partials]

    # Shuffle the partials
    partials = exp_partials + lin_partials + log_partials
    np.random.shuffle(partials)

    if args.save_samples:
        _get_name = (
            lambda i: f"sample{i}_{args.frequency}Hz_{args.duration}s_{args.samplerate}Hz"
        )
        _save_file = lambda x, i: sound.save_file(
            sound=x,
            filename=os.path.join(args.output_dir, f"{_get_name(i)}.wav"),
            samplerate=args.samplerate,
        )
    else:
        _save_file = lambda x, i: None

    # Create a dictionary to store the ratings
    output_dict = {
        "rating": [],
        "partials_type": [],
        "partial1": [],
        "partial2": [],
        "partial3": [],
        "partial4": [],
        "partial5": [],
        "partial6": [],
        "partial7": [],
        "partial8": [],
    }

    # Main loop
    for i, (partial_dist, partials_type) in enumerate(partials):
        # Generate soundwave
        sw = sound.sound_from_partials(
            partial_dist,
            base_frequency=args.frequency,
            samplerate=args.samplerate,
            duration=args.duration,
        )
        sw = sw / np.max(np.abs(sw))  # Normalize soundwave

        # Play sound
        _play_sound(sound=sw, samplerate=args.samplerate)

        # Get user rating
        rating = _get_user_rating()

        # Save sound to file
        _save_file(sw, i)

        # Append rating to the output dictionary
        _append_rating(
            output_dict=output_dict,
            rating=rating,
            partials_type=partials_type,
            partials=partial_dist,
        )

    # Save the output dictionary to a CSV file
    output_df = pl.DataFrame(output_dict)
    output_df.write_csv(
        os.path.join(args.output_dir, f"{args.output}.csv"),
        separator=",",
    )

    if args.plot_samples:
        freqs = np.arange(1, 9) * args.frequency
        fig = go.Figure()
        for i, (partial_dist, partials_type) in enumerate(partials):
            fig.add_trace(
                go.Bar(
                    x=freqs,
                    y=partial_dist,
                    name=f"Sample {i+1} ({partials_type})",
                )
            )
        fig.update_layout(
            title="Generated Samples",
            xaxis_title="Partial Index",
            yaxis_title="Partial Value",
        )
        fig.show()

    r.print(
        f"[bold green]Output file [/bold green]"
        f"[cyan]{args.output}.csv[/cyan] "
        f"[bold green]saved.[/bold green]"
    )


if __name__ == "__main__":
    main()
