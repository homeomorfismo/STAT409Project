"""DNNPype/appDNN.py: Wrapper for DNNPype to run the model."""

from __future__ import annotations

import argparse
from typing import Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import loss
import model
import opt
import optax
import polars as pl
import rich as r

###############################################################################
# Defaults and constants # TODO: Use a toml
###############################################################################
_lr: float = 0.01
_n_epochs: int = 100
_n_batches: int = 10


###############################################################################
# Auxiliary functions
###############################################################################
def _argsparse() -> argparse.Namespace:
    """Parse command line arguments."""
    _dsc: str = (
        "DNNPype: DNN-based Ising number prediction\n"
        "This script is a wrapper for DNNPype to run the model.\n"
    )
    parser = argparse.ArgumentParser(description=_dsc)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
        default=False,
    )
    # TODO: Add predict option
    # parser.add_argument(
    #     "--predict",
    #     action="store_true",
    #     help="Predict the model",
    #     default=False,
    # )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the model",
        default=False,
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        help="Number of hidden layers",
        default=2,
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        help="Number of hidden units",
        default=12,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=0,
    )
    # TODO: Add load model option
    # parser.add_argument(
    #     "--load",
    #     action="store_true",
    #     help="Load the model",
    #     default=False,
    # )
    return parser.parse_args()


def _print_args(
    *,
    args: argparse.Namespace,
) -> None:
    """Print the command line arguments."""
    r.print(
        f"[bold blue]DNNPype - DNN-based Ising number prediction\n[/bold blue]"
        f"[bold blue]===========================================\n[/bold blue]"
        f"[bold]\tTrain: [/bold] {args.train}\n"
        # f"[bold]\tPredict: [/bold] {args.predict}\n"
        f"[bold]\tSave: [/bold] {args.save}\n"
        # f"[bold]\tLoad: [/bold] {args.load}\n"
        f"[bold]\tNumber of hidden layers: [/bold] {args.n_hidden}\n"
        f"[bold]\tNumber of hidden units: [/bold] {args.dim_hidden}\n"
        f"[bold blue]=======================[/bold blue]"
        f"[bold blue]====================\n\n[/bold blue]"
    )


###############################################################################
# Data loading functions
###############################################################################
def _load_data() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load the data from the given path."""
    # TODO: Hardcode the path
    path = "../../../Data/allOrgan.csv"
    df = pl.read_csv(path)
    input_cols = [
        "isBourdon",
        "flueDepth",
        "frequency",
        "cutUpHeight",
        "diameterToe",
        "acousticIntensity",
    ]
    output_cols = [f"partial{i}" for i in range(1, 9)]
    inputs = df.select(input_cols).to_numpy()
    outputs = df.select(output_cols).to_numpy()
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs)
    return inputs, outputs


def _load_model() -> nnx.Module:
    """Load the model from the given path."""
    pass


###############################################################################
# Main function
###############################################################################
def main() -> None:
    """Run with 'run_model' command."""
    args = _argsparse()
    _print_args(args=args)

    # Load the data
    inputs, outputs = _load_data()
    theta = jnp.array([0.5, 0.5])  # pressure and density of air

    # Split into training and test sets
    ## Randomly shuffle the data
    rng = jax.random.PRNGKey(args.seed)
    rng, rngs = jax.random.split(rng)
    perm = jax.random.permutation(rngs, inputs.shape[0])
    inputs = inputs[perm]
    outputs = outputs[perm]
    # 80% training, 20% test
    n_train = int(0.8 * inputs.shape[0])
    train_inputs = inputs[:n_train]
    train_outputs = outputs[:n_train]
    test_inputs = inputs[n_train:]
    test_outputs = outputs[n_train:]

    # Set up the model
    rngs = nnx.Rngs(args.seed)
    dnn = model.SmallDNN(
        n_hidden=args.n_hidden,
        dim_hidden=args.dim_hidden,
        rngs=rngs,
    )

    # Set up optimizer
    optimizer = opt.get_optimizer(
        model=dnn,
        optax_optimizer=optax.adam,
        learning_rate=_lr,
    )

    # Set up loss function
    loss_fn = loss.refLoss

    # Set up metrics
    shape_param = jnp.array([1.0, 1.0])  # slope, intercept
    metrics = loss.LinearPartialsMetric(shape_param)

    # Train the model
    if args.train:
        opt.train(
            model=dnn,
            optimizer=optimizer,
            metrics=metrics,
            train_data=train_inputs,
            expected_data=train_outputs,
            param=theta,
            loss_fn=loss_fn,
            epochs=_n_epochs,
            batch_size=_n_batches,
        )

    # Verify the model
    results = dnn(test_inputs)
    isingNumber = loss._isingNumber(test_inputs, theta)
    # Stack the isingNumber and the outputs
    _test_outputs = jnp.hstack((isingNumber, test_outputs))
    print(f"Results: {jnp.linalg.norm(results - _test_outputs)}")


if __name__ == "__main__":
    main()
