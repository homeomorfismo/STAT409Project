"""DNNPype/loss.py: Loss functions for DNNPype."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.nnx as nnx


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
        jnp.log(jnp.ones_like(n_part_from_freq) + n_part_from_freq * slope) + intercept
    )
    partials = partials / jnp.max(partials, axis=1, keepdims=True)
    return partials


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
    print(
        f"{computedPartials.shape=}, {computedIsingNumber.shape=}, {computedResults.shape=}"
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


####################################################################################
# Tests
####################################################################################
if __name__ == "__main__":
    # Test the loss function
    fakeData = jnp.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=jnp.float32)
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
    fakeTheta = jnp.array([1, 2], dtype=jnp.float32)
    print(expLoss(fakeModel, fakeInputs, fakeTheta))
    print(linearLoss(fakeModel, fakeInputs, fakeTheta))
    print(logLoss(fakeModel, fakeInputs, fakeTheta))
