"""DNNPype/dnn.py: Rewriting of the DNN"""

# Other observations
# - On one hidden layer, it is optimal to consider 16-25 hidden units
# - More than 2 hidden layers is not recommended
# - Increasing the penalization to the Ising number loss function
#   (i.e., _ising_attention) worsens the loss function, slightly improves
#   the Ising number.
# - The convex factor seems to help if larger than 0.5 and less than 1.0
# - Standard deviation for bias initialization shold be below 0.4 and
#   above 0.1
# - Batch normalization is recommended, an alternative is to use RMSNorm
# - Clipping the gradients close to 1.0 is recommended
# - Weights decay is recommended, between 1e-2 and 1e-4
# - Current reference loss below 0.05 is successful

from __future__ import annotations

import argparse
from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import polars as pl
import rich as r
import rich.console as console

###############################################################################
# Globals
###############################################################################
# Number of inputs, hidden layers, and outputs
_n_inputs: int = 6
_n_hidden: int = 2
_dim_hidden: int = 16
_n_outputs: int = 9
# Number of environment parameters and shape parameters
_n_env_parameters: int = 2
_n_shape_parameters: int = 2
# Number of epochs and batch size
_n_epochs: int = 10
_n_batch_size: int = 32
# Default environment parameters
_default_env_parameters: jnp.ndarray = jnp.array(
    [0.7354785, 1.185]
)  # Pressure (kPa) and density (kg/m^3)
# Default shape parameters, these define a metric (partial distribution)
_default_shape_parameters: jnp.ndarray = jnp.array([1.0, 1.0])
# Standard deviation for normal distribution
_std_dev: float = 0.325
# Weight for Ising number loss, greater than 1
_ising_attention: float = 2.5
# Cosine decay scheduler parameters
_decay_steps_scheduler: int = 10
_convex_factor: float = 0.875
_power_exponent: float = 1.125
# Non-zero division epsilon
_epsilon: float = 1e-8  # Used also in sqrts
# Momentum-like factors
_beta1: float = 0.95
_beta2: float = 0.9995
_grad_clip: float = 1.0
_weights_decay: float = 1e-2


###############################################################################
# nnx.Module
###############################################################################
class SmallDNN(nnx.Module):
    """Small DNN for pipe modeling.

    Inputs
    ======
    isBourdon: int
        Indicates if the pipe is a Bourdon pipe (1) or not (0).
    flueDepth: float
        Depth of the flue in mm.
    frequency: float
        Frequency of the fundamental mode in Hz.
    cutUpHeight: float
        Height of the cut-up in mm.
    diameterToe: float
        Diameter of the toe in mm.
    acousticIntensity: float
        Acoustic intensity in dB.

    Outputs
    =======
    isingNumber: float
        Ising number.
    partial<N>: float
        Intensity of the Nth partial mode.
        N = 1, ..., 8
    """

    def __init__(
        self,
        n_hidden: int = _n_hidden,
        dim_hidden: int = _dim_hidden,
        *,
        rngs: nnx.Rngs,
    ):
        self.normalizationLayer = nnx.BatchNorm(
            num_features=_n_inputs,
            epsilon=_epsilon,
            rngs=rngs,
        )
        self.inputLayer = nnx.Linear(
            _n_inputs,
            dim_hidden,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.normal(_std_dev),
            rngs=rngs,
            use_bias=True,
        )
        self.hiddenLayers = [
            nnx.Linear(
                dim_hidden,
                dim_hidden,
                kernel_init=nnx.initializers.glorot_uniform(),
                bias_init=nnx.initializers.normal(_std_dev),
                rngs=rngs,
                use_bias=True,
            )
            for _ in range(n_hidden)
        ]
        self.outputLayerPartials = nnx.Linear(
            dim_hidden,
            _n_outputs - 1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.normal(_std_dev),
            rngs=rngs,
            use_bias=True,
        )
        self.outputLayerIsing = nnx.Linear(
            dim_hidden,
            1,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.normal(_std_dev),
            rngs=rngs,
            use_bias=True,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the DNN.

        Inputs
        ======
        x: jnp.ndarray
            Input data of shape (batch_size, _n_inputs).
        rngs: nnx.Rngs
            Random number generators.

        Outputs
        =======
        y: jnp.ndarray
            Output data of shape (batch_size, _n_outputs).
        """
        # Normalization layer
        x = self.normalizationLayer(x)

        # Input layer
        x = self.inputLayer(x)
        x = nnx.tanh(x)

        # Hidden layers
        for hiddenLayer in self.hiddenLayers:
            x = hiddenLayer(x)
            x = nnx.tanh(x)

        # Output layers
        y_partials = self.outputLayerPartials(x)
        y_partials = nnx.softmax(y_partials, axis=-1)
        y_partials = y_partials / jnp.max(y_partials, axis=-1, keepdims=True)

        y_ising = self.outputLayerIsing(x)
        y_ising = jax.nn.softplus(y_ising)

        return jnp.concatenate((y_ising, y_partials), axis=-1)


###############################################################################
# Formulae
###############################################################################
def get_exact_ising_number(
    input_data: jnp.ndarray,
    env_parameters: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the exact (batched) Ising number.

    Inputs
    ======
    input_data: jnp.ndarray
        Input data of shape (batch_size, _n_inputs).
    env_parameters: jnp.ndarray
        Environment parameters of shape (batch_size, _n_env_parameters).

    Outputs
    =======
    ising: jnp.ndarray
        Ising number of shape (batch_size, 1).
    """
    flueDepth = input_data[:, 1:2]
    frequency = input_data[:, 2:3]
    cutUpHeight = input_data[:, 3:4]

    pressure = env_parameters[:, 0:1]
    density = env_parameters[:, 1:2]

    ising = (1 / frequency) * jnp.sqrt(
        (2 * pressure * flueDepth) / (density * jnp.power(cutUpHeight, 3))
    )
    return ising


def get_linear_partial_dist(
    input_data: jnp.ndarray,
    shape_parameters: jnp.ndarray,
) -> jnp.array:
    """Compute (batched) linear distribution of the partials.

    Inputs
    ======
    input_data: jnp.ndarray
        Input data of shape (batch_size, _n_inputs).
    shape_parameters: jnp.ndarray
        Shape parameters of shape (batch_size, _n_shape_parameters).

    Outputs
    =======
    opt_partials: jnp.ndarray
        Linear distribution of the partials of shape
        (batch_size, _n_outputs-1).
    """
    frequency = input_data[:, 2:3]  # (batch_size, 1)

    slope = shape_parameters[:, 0:1]
    intercept = shape_parameters[:, 1:2]

    harmonic_multipliers = jnp.arange(1, _n_outputs, dtype=jnp.float32)
    partial_frequencies = frequency * harmonic_multipliers
    batch_intercept = (intercept * _n_outputs) * frequency

    opt_partials = slope * (batch_intercept - partial_frequencies) + intercept

    max_partials = jnp.max(opt_partials, axis=1, keepdims=True)
    opt_partials = opt_partials / jnp.where(
        max_partials == 0, 1e-6, max_partials
    )
    return opt_partials


def get_exp_partial_dist(*args, **kwargs):
    """Compute (batched) exponential distribution of the partials."""
    raise NotImplementedError(
        "Exponential distribution of the partials is not implemented yet."
    )


def get_log_partial_dist():
    """Compute (batched) logarithmic distribution of the partials."""
    raise NotImplementedError(
        "Logarithmic distribution of the partials is not implemented yet."
    )


###############################################################################
# Metrics
###############################################################################
class PipeMetric(nnx.metrics.Metric, ABC):
    """Base class for pipe metrics."""

    def __init__(self, name: Optional[str] = None):
        """Initialize the metric.

        Inputs
        ======
        name: Optional[str]
            Name of the metric.
        """
        self._name = name

    def update(self, *args, **kwargs) -> None:
        """Update metric state with new predictions and inputs."""
        raise NotImplementedError("update method not implemented.")

    def compute(self) -> jnp.ndarray:
        """Compute the final metric value."""
        raise NotImplementedError("compute method not implemented.")

    def reset(self) -> None:
        """Reset the metric state."""
        raise NotImplementedError("reset method not implemented.")


class LinearPartialsMetric(PipeMetric):
    """Metric that compares the observed partials vs. linear partials (MSE)."""

    def __init__(
        self,
        shape_parameters: jnp.ndarray,
        name: Optional[str] = None,
    ):
        """Initialize the metric.

        Inputs
        ======
        shape_parameters: jnp.ndarray
            Shape parameters for the linear distribution of the partials.
        name: Optional[str]
            Name of the metric.
        """
        super().__init__(name=name)

        _param_array = jnp.asarray(shape_parameters)

        if (
            _param_array.ndim == 1
            and _param_array.shape[0] == _n_shape_parameters
        ):
            self.shape_parameters = _param_array[jnp.newaxis, :]
        elif _param_array.ndim == 2 and _param_array.shape == (
            1,
            _n_shape_parameters,
        ):
            self.shape_parameters = _param_array
        else:
            raise ValueError(
                "shape_parameters must be a 1D array of "
                f"length {_n_shape_parameters} "
                f"or a 2D array of shape (1, {_n_shape_parameters}). "
                f"Received shape: {_param_array.shape}"
            )

        self.total_error = nnx.Variable(jnp.array(0.0, dtype=jnp.float32))
        self.num_samples = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def update(
        self,
        y_pred: jnp.ndarray,
        input_data: jnp.ndarray,
        **kwargs,
    ) -> None:
        """Update metric state with new predictions and corresponding inputs.

        Inputs
        ======
        y_pred: jnp.ndarray
            Model predictions of shape (batch_size, _n_outputs).
        input_data: jnp.ndarray
            Input data of shape (batch_size, _n_inputs).
        kwargs: dict
            Additional keyword arguments (not used).
        """
        pred_partials = y_pred[:, 1:]  # Extract partials from model output
        target_partials = get_linear_partial_dist(
            input_data, self.shape_parameters
        )

        squared_errors = (pred_partials - target_partials) ** 2
        mse_per_sample = jnp.mean(squared_errors, axis=-1)

        self.total_error.value += jnp.sum(mse_per_sample)
        self.num_samples.value += pred_partials.shape[0]

    def compute(self) -> jnp.ndarray:
        """Compute the final metric value (average MSE)."""
        return jnp.where(
            self.num_samples.value > 0,
            self.total_error.value / self.num_samples.value,
            jnp.array(0.0, dtype=jnp.float32),
        )

    def reset(self) -> None:
        """Reset the metric state."""
        self.total_error.value = jnp.array(0.0, dtype=jnp.float32)
        self.num_samples.value = jnp.array(0, dtype=jnp.int32)


################################################################################
# Loss functions
################################################################################
def reference_loss(
    dnn: nnx.Module,
    input_data: jnp.ndarray,
    output_data: jnp.ndarray,
    regularization: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the loss function for the DNN.

    Inputs
    ======
    dnn: nnx.Module
        DNN model.
    input_data: jnp.ndarray
        Input data of shape (batch_size, _n_inputs).
    output_data: jnp.ndarray
        Output data of shape (batch_size, _n_outputs).
    regularization: float
        Regularization parameter for L2 regularization.

    Outputs
    =======
    loss: jnp.ndarray
        Loss value.
    """
    predicted_data = dnn(input_data)

    predicted_ising = predicted_data[:, 0:1]
    predicted_partials = predicted_data[:, 1:]

    ref_ising = output_data[:, 0:1]
    ref_partials = output_data[:, 1:]

    loss_ising = jnp.square(predicted_ising - ref_ising)
    loss_partials = jnp.mean(
        jnp.square(predicted_partials - ref_partials),
        axis=1,
        keepdims=True,
    )

    reg_weights = jnp.sum(
        jnp.square(optax.global_norm(nnx.state(dnn, nnx.Param)))
    )

    loss = jnp.mean(_ising_attention * loss_ising + loss_partials)
    loss += regularization * reg_weights

    return loss, predicted_data


################################################################################
# Training and evaluation functions
################################################################################
def train(
    dnn: nnx.Module,
    loss_fn: Callable[
        [nnx.Module, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray],
    ],
    optimizer: nnx.Optimizer,
    metrics: PipeMetric,
    train_data: jnp.ndarray,
    expected_data: jnp.ndarray,
    epochs: int = _n_epochs,
    batch_size: int = _n_batch_size,
) -> None:
    """Train the DNN model.

    Inputs
    ======
    dnn: nnx.Module
        DNN model.
    loss_fn: Callable
        Loss function.
    optimizer: nnx.Optimizer
        Optimizer for training.
    metrics: nnx.MultiMetric
        Metrics for tracking performance.
    train_data: jnp.ndarray
        Training data of shape (num_samples, _n_inputs).
    expected_data: jnp.ndarray
        Expected data of shape (num_samples, _n_outputs).
    epochs: int
        Number of epochs for training.
    batch_size: int
        Batch size for training.
    """
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, num_samples)

            x_batch = train_data[start_idx:end_idx]
            y_batch = expected_data[start_idx:end_idx]

            (loss, predicted), grads = nnx.value_and_grad(
                loss_fn,
                has_aux=True,
                argnums=(0,),
            )(
                dnn,
                x_batch,
                y_batch,
            )
            optimizer.update(grads[0])
            metrics.update(
                predicted,
                x_batch,
            )
        epoch_metrics = metrics.compute()
        r.print(
            f"[bold] Epoch: [/bold] {epoch + 1}/{epochs}\n"
            f"[bold] Loss: [/bold] {loss}\n"
            f"[bold] Metrics: [/bold] {epoch_metrics}\n"
        )
        metrics.reset()


def evaluate(
    dnn: nnx.Module,
    loss_fn: Callable[
        [nnx.Module, jnp.ndarray, jnp.ndarray],
        Tuple[jnp.ndarray, jnp.ndarray],
    ],
    metrics: PipeMetric,
    eval_data: jnp.ndarray,
    expected_eval_data: jnp.ndarray,
    env_parameters_eval: jnp.ndarray,
    shape_parameters_eval: jnp.ndarray,
    batch_size: int = _n_batch_size,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate the DNN model.

    Inputs
    ======
    dnn: nnx.Module
        DNN model.
    loss_fn: Callable
        Loss function that takes (dnn, input, output_targets)
        and returns a tuple of (loss_value, predicted_data_batch).
    metrics: nnx.MultiMetric
        Metrics for tracking performance.
    eval_data: jnp.ndarray
        Evaluation input data of shape (num_samples, _n_inputs).
    expected_eval_data: jnp.ndarray
        Expected (target) evaluation data of shape
        (num_samples, _n_outputs).
    env_parameters_eval: jnp.ndarray
        Environment parameters for eval data of shape
        (num_samples, _n_env_parameters).
    shape_parameters_eval: jnp.ndarray
        Shape parameters for eval data of shape
        (num_samples, _n_shape_parameters).
    batch_size: int
        Batch size for evaluation.

    Returns
    =======
    Tuple[float, dict]:
        Average loss on the evaluation set.
    """
    num_samples = eval_data.shape[0]
    if num_samples == 0:
        r.print("[yellow]Warning: Evaluation dataset is empty.[/yellow]")
        return 0.0, {}

    num_batches = (num_samples + batch_size - 1) // batch_size

    metrics.reset()
    total_loss_eval = 0.0
    processed_samples = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        if start_idx >= end_idx:
            continue

        x_batch = eval_data[start_idx:end_idx]
        y_true_batch = expected_eval_data[start_idx:end_idx]

        env_parameters_eval[start_idx:end_idx]
        shape_parameters_eval[start_idx:end_idx]

        loss_value, predicted_data_batch = loss_fn(
            dnn,
            x_batch,
            y_true_batch,
        )

        current_batch_size = x_batch.shape[0]
        total_loss_eval += loss_value.item() * current_batch_size
        processed_samples += current_batch_size

        metrics.update(
            predicted_data_batch,
            x_batch,
        )

    avg_loss_eval = (
        total_loss_eval / processed_samples if processed_samples > 0 else 0.0
    )
    eval_metrics_computed = metrics.compute()

    r.print(
        f"[bold green]Evaluation Results:[/bold green]\n"
        f"  [bold]Avg Loss:[/bold] {avg_loss_eval:.4f}\n"
        f"  [bold]Metrics:[/bold] {eval_metrics_computed}\n"
    )

    # Print the Ising numbers from the evaluation data
    ising_numbers = expected_eval_data[:, 0:1]
    r.print(
        "[bold green]Ising numbers from evaluation data:[/bold green]\n"
        f"{ising_numbers}"
    )
    r.print(
        "[bold yellow]Ising numbers from the model:[/bold yellow]\n"
        f"{predicted_data_batch[:, 0:1]}"
    )

    return avg_loss_eval, eval_metrics_computed


###############################################################################
# Data loading functions
###############################################################################
def load_data(
    str_path: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load the data from the given path."""
    if str_path is not None:
        path = str_path
    else:
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
    # Outputs need to be divided by a 100
    output_cols = [f"partial{i}" for i in range(1, 9)]
    inputs = df.select(input_cols).to_numpy()
    outputs = df.select(output_cols).to_numpy()
    inputs = jnp.array(inputs)
    outputs = jnp.array(outputs) / 100.0

    # Compute the exact Ising number from inputs
    batch_env_parameters = jnp.tile(
        _default_env_parameters, (inputs.shape[0], 1)
    )
    ising_number = get_exact_ising_number(
        inputs,
        batch_env_parameters,
    )
    # Pre-concatenate the Ising number with the partials
    outputs = jnp.concatenate((ising_number, outputs), axis=1)
    return inputs, outputs


def get_args():
    """Parse command line arguments.

    Returns
    =======
    argparse.Namespace
        Parsed command line arguments.
    """
    dscr = (
        "DNNPype/dnn.py: DNN model for organ pipe acoustics.\n"
        "Train or evaluate a DNN model for organ pipe acoustics."
    )
    parser = argparse.ArgumentParser(description=dscr)

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Operation mode: 'train' or 'evaluate'. Default: train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,  # Assuming _n_epochs might be 100
        help="Number of training epochs. Default: 100",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,  # Assuming _n_batch_size might be 32
        help="Batch size for training and evaluation. Default: 32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer. Default: 0.001",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../../Data/allOrgan.csv",
        help="Path to the CSV data file.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Seed for JAX RNG key generation.",
    )
    return parser.parse_args()


#################################################################################
# Main function
#################################################################################
def main():
    """Main function to run the DNNPype model."""
    # 0. Parse command line arguments
    args = get_args()
    rich_console = console.Console()

    _temp_str: str = (
        f"[bold cyan]DNNPype ({args.mode} mode)[/bold cyan]\n"
        f"\t[bold] Epochs: {args.epochs}[/bold]\n"
        f"\t[bold] Batch Size: {args.batch_size}[/bold]\n"
        f"\t[bold] LR: {args.learning_rate}[/bold]\n"
        f"\t[bold] Data Path: {args.data_path}[/bold]\n"
        f"\t[bold] RNG Seed: {args.rng_seed}[/bold]\n"
    )
    rich_console.print(_temp_str)

    # 1. Initialize model
    rngs = nnx.Rngs(args.rng_seed)
    dnn_model = SmallDNN(rngs=rngs)

    # 2. TODO: Implement model loading if args.load_path is provided

    # 3. Initialize Optimizer
    scheduler = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=_decay_steps_scheduler,
        alpha=_convex_factor,
        exponent=_power_exponent,
    )
    optax_optimizer = optax.chain(
        optax.clip_by_global_norm(_grad_clip),
        optax.scale_by_adam(
            b1=_beta1,
            b2=_beta2,
            eps=_epsilon,
            eps_root=_epsilon,
        ),
        optax.add_decayed_weights(
            weight_decay=_weights_decay,
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0),  # descending
    )
    optimizer = nnx.Optimizer(dnn_model, optax_optimizer)

    # 4. Load Data (Ising numbers pre-computed)
    all_input_data, all_expected_outputs = load_data(str_path=args.data_path)

    # 4.2 Print Ising numbers from the loaded data
    ising_numbers = all_expected_outputs[:, 0:1]
    rich_console.print(
        f"[bold green]Ising numbers from loaded data:[/bold green]\n"
        f"{ising_numbers}"
    )
    rich_console.input("[bold yellow]Press Enter to continue...[/bold yellow]")

    if all_input_data is None or all_expected_outputs is None:
        rich_console.print(
            "[bold red]Failed to load data. Exiting.[/bold red]"
        )
        raise ValueError("Data loading failed.")
    if all_input_data.shape[0] != all_expected_outputs.shape[0]:
        rich_console.print(
            "[bold red]Incompatible shapes. Exiting.[/bold red]"
        )
        raise ValueError("Incompatible shapes.")
    if all_expected_outputs.shape[1] != _n_outputs:
        rich_console.print(
            "[bold red]Outputs shape mismatch. Exiting.[/bold red]"
        )
        raise ValueError("Outputs shape mismatch.")

    # 5. Initialize Metrics
    metrics_computer = LinearPartialsMetric(
        shape_parameters=_default_shape_parameters
    )

    # 6. Split Data (Simple Shuffle and Split)
    num_total_samples = all_input_data.shape[0]
    if num_total_samples < 2:  # Need at least 1 for train, 1 for eval
        rich_console.print("[bold red]Not enough data! Exiting.[/bold red]")
        raise ValueError("Not enough data for training and evaluation.")

    indices = jax.random.permutation(rngs(), num_total_samples)
    shuffled_inputs = all_input_data[indices]
    shuffled_expected_outputs = all_expected_outputs[indices]

    split_idx = int(0.8 * num_total_samples)
    train_input = shuffled_inputs[:split_idx]
    train_expected = shuffled_expected_outputs[:split_idx]
    eval_input = shuffled_inputs[split_idx:]
    eval_expected = shuffled_expected_outputs[split_idx:]

    _temp_str = (
        f"[bold]Total Samples: {num_total_samples}[/bold]\n"
        f"[bold]Training Samples: {train_input.shape[0]}[/bold]\n"
        f"[bold]Evaluation Samples: {eval_input.shape[0]}[/bold]\n"
    )
    rich_console.print(_temp_str)

    jnp.tile(_default_env_parameters, (train_input.shape[0], 1))
    jnp.tile(_default_shape_parameters, (train_input.shape[0], 1))

    eval_env_params = (
        jnp.tile(_default_env_parameters, (eval_input.shape[0], 1))
        if eval_input.shape[0] > 0
        else jnp.empty((0, _n_env_parameters))
    )
    eval_shape_params = (
        jnp.tile(_default_shape_parameters, (eval_input.shape[0], 1))
        if eval_input.shape[0] > 0
        else jnp.empty((0, _n_shape_parameters))
    )

    # 7. Run Mode
    if args.mode == "train":
        if train_input.shape[0] == 0:
            rich_console.print(
                "[bold red]Training data is empty. Skipping...[/bold red]"
            )
        else:
            rich_console.print(
                "[bold yellow]Starting Training...[/bold yellow]"
            )
            train(
                dnn=dnn_model,
                loss_fn=reference_loss,  # This is a 3-argument function
                optimizer=optimizer,
                metrics=metrics_computer,  # Pass the initialized metric object
                train_data=train_input,
                expected_data=train_expected,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            # TODO: Implement model saving if args.save_path is provided

        # Optionally evaluate after training
        if eval_input.shape[0] > 0:
            rich_console.print(
                "[bold yellow]Evaluating after training...[/bold yellow]"
            )
            evaluate(
                dnn=dnn_model,
                loss_fn=reference_loss,  # This is a 3-argument function
                metrics=metrics_computer,
                eval_data=eval_input,
                expected_eval_data=eval_expected,
                env_parameters_eval=eval_env_params,
                shape_parameters_eval=eval_shape_params,
                batch_size=args.batch_size,
            )
        elif (
            args.mode == "train"
        ):  # Only print if mode was train and no eval data
            rich_console.print(
                "[yellow]No evaluation data to evaluate.[/yellow]"
            )

    elif args.mode == "evaluate":
        # TODO: Implement model loading if args.load_path is provided
        # if not args.load_path:
        # rich_console.print(TODO)
        # return

        if eval_input.shape[0] == 0:
            rich_console.print(
                "[bold red]Evaluation data is empty. [/bold red]"
                "[bold red] Skipping evaluation.[/bold red]"
            )
        else:
            rich_console.print(
                "[bold yellow]Starting Evaluation...[/bold yellow]"
            )
            evaluate(
                dnn=dnn_model,
                loss_fn=reference_loss,  # This is a 3-argument function
                metrics=metrics_computer,
                eval_data=eval_input,
                expected_eval_data=eval_expected,
                env_parameters_eval=eval_env_params,
                shape_parameters_eval=eval_shape_params,
                batch_size=args.batch_size,
            )

    rich_console.print(
        f"[bold cyan]DNNPype ({args.mode} mode) finished.[/bold cyan]"
    )


if __name__ == "__main__":
    main()
