"""DNNPype/loss.py: Loss functions for DNNPype."""

from __future__ import annotations

from typing import List, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp


###############################################################################
# Formulae
###############################################################################
@nnx.jit
def _isingNumber(x_pipe: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) Ising number."""
    _, flueDepth, frequency, cutUpHeight, _, _ = jnp.split(x_pipe, 6, axis=1)
    pressure, density = theta
    isingNumber = (1 / frequency) * jnp.sqrt(
        (2 * pressure * flueDepth) / (density * jnp.power(cutUpHeight, 3))
    )
    return isingNumber


@nnx.jit
def _exponentialPartials(freq: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) exponential partials."""
    n_part_from_freq = jnp.expand_dims(freq, axis=1) * jnp.arange(1, 9)
    slicing_idx: list[int] = [1]
    shift, intercept = jnp.split(theta, slicing_idx, axis=-1)
    partials = jnp.exp(-n_part_from_freq * intercept) + shift
    partials = partials / jnp.max(partials, axis=1, keepdims=True)
    return partials


@nnx.jit
def _linearPartials(freq: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) linear partials."""
    n_part_from_freq = jnp.expand_dims(freq, axis=1) * jnp.arange(1, 9)
    slicing_idx: list[int] = [1]
    slope, intercept = jnp.split(theta, slicing_idx, axis=-1)
    partials = n_part_from_freq * slope + intercept
    partials = partials / jnp.max(partials, axis=1, keepdims=True)
    return partials


@nnx.jit
def _logPartials(freq: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Compute the (batched) log partials."""
    n_part_from_freq = jnp.expand_dims(freq, axis=1) * jnp.arange(1, 9)
    slicing_idx: list[int] = [1]
    slope, intercept = jnp.split(theta, slicing_idx, axis=-1)
    partials = (
        jnp.log(jnp.ones_like(n_part_from_freq) + n_part_from_freq * slope)
        + intercept
    )
    partials = partials / jnp.max(partials, axis=1, keepdims=True)
    return partials


################################################################################
# Metrics
################################################################################
class ExponentialPartialsMetric(nnx.metrics.Metric):
    """Metric for exponential partials."""

    def __init__(self, theta: Optional[jnp.ndarray] = None):
        self.predicted: List[jnp.ndarray] = []
        self.target: List[jnp.ndarray] = []
        self._theta = theta

    def update(
        self,
        x_pipe: jnp.ndarray,
        predicted_output: jnp.ndarray,
        shape_params: Optional[jnp.ndarray] = None,
    ):
        """Store predicted and ideal exponential partials."""
        # Extract frequency (third part) from data; shape (batch, 1)
        _, _, frequency, _, _, _ = jnp.split(x_pipe, 6, axis=1)
        _shape_params = (
            shape_params if shape_params is not None else self._theta
        )
        ideal = _exponentialPartials(frequency, _shape_params)
        if predicted_output.shape[1] == 9:
            _, pred = jnp.split(predicted_output, [1], axis=1)
        elif predicted_output.shape[1] == 8:
            pred = predicted_output
        else:
            raise ValueError("Unexpected predicted_output shape.")
        self.predicted.append(pred)
        self.target.append(ideal)

    def compute(self) -> jnp.ndarray:
        """Return the mean L2 error."""
        if not self.predicted or not self.target:
            return jnp.array(0.0)
        pred_all = jnp.concatenate(self.predicted, axis=0)
        targ_all = jnp.concatenate(self.target, axis=0)
        # Compute the mean norm of the difference along the partials axis.
        return jnp.mean(jnp.linalg.norm(pred_all - targ_all, axis=1))

    def reset(self):
        """Clear stored values."""
        self.predicted.clear()
        self.target.clear()


class LinearPartialsMetric(nnx.metrics.Metric):
    """Metric for linear partials."""

    def __init__(self, theta: Optional[jnp.ndarray] = None):
        self.predicted: List[jnp.ndarray] = []
        self.target: List[jnp.ndarray] = []
        self._theta = theta

    def update(
        self,
        x_pipe: jnp.ndarray,
        predicted_output: jnp.ndarray,
        shape_params: Optional[jnp.ndarray] = None,
    ):
        """Store predicted and ideal linear partials."""
        _, _, frequency, _, _, _ = jnp.split(x_pipe, 6, axis=1)
        _shape_params = (
            shape_params if shape_params is not None else self._theta
        )
        ideal = _linearPartials(frequency, _shape_params)
        if predicted_output.shape[1] == 9:
            _, pred = jnp.split(predicted_output, [1], axis=1)
        elif predicted_output.shape[1] == 8:
            pred = predicted_output
        else:
            raise ValueError("Unexpected predicted_output shape.")
        self.predicted.append(pred)
        self.target.append(ideal)

    def compute(self) -> jnp.ndarray:
        """Return the mean L2 error."""
        if not self.predicted or not self.target:
            return jnp.array(0.0)
        pred_all = jnp.concatenate(self.predicted, axis=0)
        targ_all = jnp.concatenate(self.target, axis=0)
        return jnp.mean(jnp.linalg.norm(pred_all - targ_all, axis=1))

    def reset(self):
        """Clear stored values."""
        self.predicted.clear()
        self.target.clear()


class LogPartialsMetric(nnx.metrics.Metric):
    """Metric for log partials."""

    def __init__(self, theta: Optional[jnp.ndarray] = None):
        self.predicted: List[jnp.ndarray] = []
        self.target: List[jnp.ndarray] = []
        self._theta = theta

    def update(
        self,
        x_pipe: jnp.ndarray,
        predicted_output: jnp.ndarray,
        shape_params: Optional[jnp.ndarray] = None,
    ):
        """Store predicted and ideal log partials."""
        _, _, frequency, _, _, _ = jnp.split(x_pipe, 6, axis=1)
        _shape_params = (
            shape_params if shape_params is not None else self._theta
        )
        ideal = _logPartials(frequency, _shape_params)
        if predicted_output.shape[1] == 9:
            _, pred = jnp.split(predicted_output, [1], axis=1)
        elif predicted_output.shape[1] == 8:
            pred = predicted_output
        else:
            raise ValueError("Unexpected predicted_output shape.")
        self.predicted.append(pred)
        self.target.append(ideal)

    def compute(self) -> jnp.ndarray:
        """Return the mean L2 error."""
        if not self.predicted or not self.target:
            return jnp.array(0.0)
        pred_all = jnp.concatenate(self.predicted, axis=0)
        targ_all = jnp.concatenate(self.target, axis=0)
        return jnp.mean(jnp.linalg.norm(pred_all - targ_all, axis=1))

    def reset(self):
        """Clear stored values."""
        self.predicted.clear()
        self.target.clear()


################################################################################
# Loss functions
################################################################################
def expLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    theta: jnp.ndarray,
):
    """Flat partials + exact Ising number"""
    computedResults = model(inputs)  # (ising, partial1, ..., partial8)
    slicing_idx: list[int] = [1]
    computedIsingNumber, computedPartials = jnp.split(
        computedResults, slicing_idx, axis=1
    )
    _, _, frequency, _, _, _ = jnp.split(inputs, 6, axis=1)
    refIsingNumber = _isingNumber(inputs, theta)
    refPartials = _exponentialPartials(frequency, theta)

    loss = jnp.mean(
        jnp.square(computedIsingNumber - refIsingNumber)
        + jnp.sum(jnp.square(computedPartials - refPartials), axis=1)
    )
    return loss


def linearLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    theta: jnp.ndarray,
):
    """Linear partials + exact Ising number"""
    computedResults = model(inputs)  # (ising, partial1, ..., partial8)
    slicing_idx: list[int] = [1]
    computedIsingNumber, computedPartials = jnp.split(
        computedResults, slicing_idx, axis=1
    )
    _, _, frequency, _, _, _ = jnp.split(inputs, 6, axis=1)
    refIsingNumber = _isingNumber(inputs, theta)
    refPartials = _linearPartials(frequency, theta)

    loss = jnp.mean(
        jnp.square(computedIsingNumber - refIsingNumber)
        + jnp.sum(jnp.square(computedPartials - refPartials), axis=1)
    )
    return loss


def logLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    theta: jnp.ndarray,
):
    """Log partials + exact Ising number"""
    computedResults = model(inputs)  # (ising, partial1, ..., partial8)
    slicing_idx: list[int] = [1]
    computedIsingNumber, computedPartials = jnp.split(
        computedResults, slicing_idx, axis=1
    )
    _, _, frequency, _, _, _ = jnp.split(inputs, 6, axis=1)
    refIsingNumber = _isingNumber(inputs, theta)
    refPartials = _logPartials(frequency, theta)

    loss = jnp.mean(
        jnp.square(computedIsingNumber - refIsingNumber)
        + jnp.sum(jnp.square(computedPartials - refPartials), axis=1)
    )
    return loss


def refLoss(
    model: nnx.Module,
    inputs: jnp.ndarray,
    refPartials: jnp.ndarray,
    theta: jnp.ndarray,
):
    """Reference loss function for testing"""
    computedResults = model(inputs)  # (ising, partial1, ..., partial8)
    slicing_idx: list[int] = [1]
    computedIsingNumber, computedPartials = jnp.split(
        computedResults, slicing_idx, axis=1
    )
    refIsingNumber = _isingNumber(inputs, theta)
    loss = jnp.mean(
        jnp.square(computedIsingNumber - refIsingNumber)
        + jnp.sum(jnp.square(computedPartials - refPartials), axis=1)
    )
    return loss


####################################################################################
# Tests
####################################################################################
if __name__ == "__main__":
    # Test the loss function
    fakeData = jnp.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=jnp.float32
    )
    fakeTheta = jnp.array([1, 2], dtype=jnp.float32)
    print(_isingNumber(fakeData, fakeTheta))
    fakeFreq = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.float32)
    print(_exponentialPartials(fakeFreq, fakeTheta))
    print(_linearPartials(fakeFreq, fakeTheta))
    print(_logPartials(fakeFreq, fakeTheta))

    # Test the loss function
    fakeModel = nnx.Linear(
        6, 9, kernel_init=nnx.initializers.xavier_uniform(), rngs=nnx.Rngs(0)
    )
    fakeInputs = jnp.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=jnp.float32
    )
    fakePartials = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=jnp.float32,
    )
    fakeTheta = jnp.array([1, 2], dtype=jnp.float32)

    print("===== Losses =====")
    print(expLoss(fakeModel, fakeInputs, fakeTheta))
    print(linearLoss(fakeModel, fakeInputs, fakeTheta))
    print(logLoss(fakeModel, fakeInputs, fakeTheta))
    print(refLoss(fakeModel, fakeInputs, fakePartials, fakeTheta))

    # Differentiate the loss function

    grad_expLoss = jax.grad(expLoss, argnums=1)
    grad_linearLoss = jax.grad(linearLoss, argnums=1)
    grad_logLoss = jax.grad(logLoss, argnums=1)
    grad_refLoss = jax.grad(refLoss, argnums=1)

    print("===== Gradients =====")
    print(grad_expLoss(fakeModel, fakeInputs, fakeTheta))
    print(grad_linearLoss(fakeModel, fakeInputs, fakeTheta))
    print(grad_logLoss(fakeModel, fakeInputs, fakeTheta))
    print(grad_refLoss(fakeModel, fakeInputs, fakePartials, fakeTheta))

    # Test the metrics
    exp_metric = ExponentialPartialsMetric()
    linear_metric = LinearPartialsMetric()
    log_metric = LogPartialsMetric()
    exp_metric.update(fakeData, fakePartials, fakeTheta)
    linear_metric.update(fakeData, fakePartials, fakeTheta)
    log_metric.update(fakeData, fakePartials, fakeTheta)
    print("===== Before reset =====")
    print(exp_metric.compute())
    print(linear_metric.compute())
    print(log_metric.compute())

    # Reset the metrics
    exp_metric.reset()
    linear_metric.reset()
    log_metric.reset()
    print("===== After reset =====")
    print(exp_metric.compute())
    print(linear_metric.compute())
    print(log_metric.compute())
