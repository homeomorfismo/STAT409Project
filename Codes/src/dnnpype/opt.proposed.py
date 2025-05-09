"""DNNPype/opt.py: Wrapper loops and utilities for training and evaluation."""

from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

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
        optax_optimizer (optax.GradientTransformation): The Optax optimizer function.
        **kwargs (Dict[str, float]): Additional parameters for the optimizer.

    Returns:
        Tuple[nnx.Optimizer, Callable]: A tuple containing the initialized optimizer and its update function.
    """
    if "learning_rate" not in kwargs:
        kwargs["learning_rate"] = (
            _DEFAULT_LR  # Use default learning rate if not provided
        )
    return nnx.Optimizer(model, optax_optimizer(**kwargs))


def train(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetrics,
    train_dataset: Dict[str, jnp.ndarray],
    loss_fn: Callable[[nnx.Module, Dict[str, jnp.ndarray]], Tuple[float, Any]],
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """
    Train the model on the given training dataset.

    Args:
        model (nnx.Module): The model to train.
        optimizer (nnx.Optimizer): The optimizer used for training.
        metrics (nnx.MultiMetrics): Metrics object to track training performance.
        train_dataset (Dict[str, jnp.ndarray]): The training dataset as a dictionary of features and labels.
        loss_fn (Callable): The loss function to compute gradients.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 32.
    """
    num_samples = train_dataset["data"].shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx in range(num_batches):
            # Extract batch data
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = {
                key: value[start_idx:end_idx] for key, value in train_dataset.items()
            }

            # Compute gradients and update model
            grad = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad(model, batch)
            metrics.update(loss=loss, logits=logits, labels=batch["label"])
            optimizer.update(grads)

        # Compute and print epoch metrics
        epoch_metrics = metrics.compute()
        print(f"Epoch {epoch + 1} Metrics: {epoch_metrics}")
        metrics.reset()  # Reset metrics for the next epoch


def evaluate(
    model: nnx.Module,
    metrics: nnx.MultiMetrics,
    eval_dataset: Dict[str, jnp.ndarray],
    batch_size: int = 32,
) -> None:
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        model (nnx.Module): The model to evaluate.
        metrics (nnx.MultiMetrics): Metrics object to track evaluation performance.
        eval_dataset (Dict[str, jnp.ndarray]): The evaluation dataset as a dictionary of features and labels.
        batch_size (int, optional): The batch size for evaluation. Defaults to 32.
    """
    num_samples = eval_dataset["data"].shape[0]
    num_batches = num_samples // batch_size

    for batch_idx in range(num_batches):
        # Extract batch data
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = {key: value[start_idx:end_idx] for key, value in eval_dataset.items()}

        # Compute logits and update metrics
        logits = model(batch)
        metrics.update(logits=logits, labels=batch["label"])

    # Print evaluation metrics
    eval_metrics = metrics.compute()
    print(f"Evaluation Metrics: {eval_metrics}")


# Example of initializing an optimizer (can be configured dynamically)
optimizer = get_optimizer(
    model=nnx.Module(),  # Replace with your actual model
    optax_optimizer=optax.adam,
    learning_rate=_DEFAULT_LR,
)

# Example of defining metrics (can be customized)
metrics = nnx.MultiMetrics(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average(),
)
