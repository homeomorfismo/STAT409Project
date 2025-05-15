"""DNNPype/opt.py: Wrapper loops and utilities for training and evaluation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import flax.nnx as nnx
import jax.numpy as jnp
import optax

# import model

# Default learning rate
_DEFAULT_LR: float = 0.01


def get_optimizer(
    model: nnx.Module,
    optax_optimizer: optax.GradientTransformation,
    **kwargs: Dict[str, float],
) -> Tuple[nnx.Optimizer, Callable]:
    """
    Initialize the optimizer for the given model.

    Args:
        model (nnx.Module): The model to optimize.
        optax_optimizer (optax.GradientTransformation):
            The Optax optimizer function.
        **kwargs (Dict[str, float]):
            Additional parameters for the optimizer.

    Returns:
        Tuple[nnx.Optimizer, Callable]:
            A tuple containing the initialized optimizer
            and its update function.
    """
    if "learning_rate" not in kwargs:
        kwargs.setdefault("learning_rate", _DEFAULT_LR)
    return nnx.Optimizer(model, optax_optimizer(**kwargs))


def train(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    train_data: jnp.ndarray,
    expected_data: jnp.ndarray,
    param: jnp.ndarray,
    loss_fn: Callable[..., Tuple[float, Any]],
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """
    Train the model on the given training dataset.

    Args:
        model (nnx.Module): The model to train.
        optimizer (nnx.Optimizer): The optimizer used for training.
        metrics (nnx.MultiMetrics):
            Metrics object to track training performance.
        train_data (jnp.ndarray):
            The training dataset as a jnp.ndarray.
        expected_data (jnp.ndarray):
            The expected outputs as a jnp.ndarray.
        param (jnp.ndarray): The parameters for the model.
        loss_fn (Callable):
            The loss function to compute gradients.
        epochs (int, optional):
            The number of epochs to train for. Defaults to 10.
        batch_size (int, optional):
            The batch size for training. Defaults to 32.
    """
    num_samples = train_data.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx in range(num_batches):
            # Extract batch data
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch = train_data[start_idx:end_idx]
            expected = expected_data[start_idx:end_idx]

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=False, argnums=(0,))
            loss, grads = grad_fn(model, batch, expected, param)
            # metrics.update(loss=loss, logits=logits, labels=expected)
            optimizer.update(grads[0])

        # Compute and print epoch metrics
        # epoch_metrics = metrics.compute()
        # print(f"Epoch {epoch + 1} Metrics: {epoch_metrics}")
        # metrics.reset()  # Reset metrics for the next epoch


def evaluate(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    eval_dataset: Dict[str, jnp.ndarray],
    batch_size: int = 32,
) -> None:
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        model (nnx.Module): The model to evaluate.
        metrics (nnx.MultiMetrics):
            Metrics object to track evaluation performance.
        eval_dataset (Dict[str, jnp.ndarray]):
            The evaluation dataset as a dictionary of features and labels.
        batch_size (int, optional):
            The batch size for evaluation. Defaults to 32.
    """
    num_samples = eval_dataset["data"].shape[0]
    num_batches = num_samples // batch_size

    for batch_idx in range(num_batches):
        # Extract batch data
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = {
            key: value[start_idx:end_idx]
            for key, value in eval_dataset.items()
        }

        # Compute logits and update metrics
        logits = model(batch)
        metrics.update(logits=logits, labels=batch["label"])

    # Print evaluation metrics
    eval_metrics = metrics.compute()
    print(f"Evaluation Metrics: {eval_metrics}")


if __name__ == "__main__":
    pass
