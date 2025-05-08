"""DNNPype/opt.py: Wrapper loops for DNNPype."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# _default: Dict[str, float] = {
#     "learning_rate": 0.001,
#     "weight_decay": 0.0,
#     "momentum": 0.9,
#     "nesterov": False,
#     "clip_value": 1.0,
#     "clip_gradient": False,
#     "max_grad_norm": 1.0,
#     "eps": 1e-8,
#     "decay_steps": 10000,
#     "decay_rate": 0.96,
# }
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
):
    """Train the model on the training dataset."""
    grad = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad(model, train_dataset)
    metrics.update(loss=loss, logits=logits, labels=train_dataset["label"])
    optimizer.update(grads)
