"""DNNPype/opt.py: Wrapper loops for DNNPype."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

_lr: float = 0.01


def getOptimizer(
    model: nnx.Module,
    optax_optimizer: optax.GradientTransformation,
    **kwargs: Dict[str, float],
) -> Tuple[nnx.Module, Callable]:
    return nnx.Optimizer(model, optax_optimizer(**kwargs))


optimizer = nnx.Optimizer(model, optax.adam(learning_rate=_lr))
metrics = nnx.MultiMetrics(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average(),
)


def train(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetrics,
    train_dataset: jnp.ndarray,
) -> None:
    """Train the model on the training dataset."""
    grad = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad(model, train_dataset)
    metrics.update(loss=loss, logits=logits, labels=train_dataset["label"])
    optimizer.update(grads)


def eval(
    model: nnx.Module,
    metrics: nnx.MultiMetrics,
    eval_dataset: jnp.ndarray,
) -> None:
    """Evaluate the model on the evaluation dataset."""
    logits = model(eval_dataset)
    metrics.update(logits=logits, labels=eval_dataset["label"])
