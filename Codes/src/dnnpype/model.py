"""DNNPype/model.py: DNN models for pipe modeling."""

from __future__ import annotations
from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx


_n_inputs: int = 6
_n_outputs: int = 9


class SmallDNN(nnx.Module):
    """Small DNN for pipe modeling.

    This model takes:
        - isBourdon, flueDepth, frequency, cutUpHeight, diameterToe, acousticIntensity
    and returns:
        - isingNumber, partial1, ..., partial8
    """

    def __init__(
        self,
        n_hidden: int = 2,
        dim_hidden: int = 12,
        rngs: Optional[nnx.Rngs] = None,
    ):
        assert rngs is not None, "RNGs must be provided for the model."
        self.inputLayer = nnx.Linear(
            _n_inputs,
            dim_hidden,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs,
        )
        self.hiddenLayers = [
            nnx.Linear(
                dim_hidden,
                dim_hidden,
                kernel_init=nnx.initializers.glorot_uniform(),
                rngs=rngs,
            )
            for _ in range(n_hidden)
        ]
        self.normalizationLayer = nnx.RMSNorm(
            num_features=dim_hidden,
            rngs=rngs,
        )
        self.outputLayerPartials = nnx.Linear(
            dim_hidden,
            _n_outputs - 1,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs,
        )
        self.outputLayerIsing = nnx.Linear(
            dim_hidden,
            1,
            kernel_init=nnx.initializers.glorot_uniform(),
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the DNN.

        Args:
            x (jnp.ndarray): Input data.
            rngs (nnx.Rngs): Random number generators.

        Returns:
            jnp.ndarray: Output data.
        """
        # Input layer
        x = self.inputLayer(x)
        x = nnx.relu(x)

        # Hidden layers
        for hiddenLayer in self.hiddenLayers:
            x = hiddenLayer(x)
            x = nnx.relu(x)

        # Normalization layer
        x = self.normalizationLayer(x)

        # Output layers
        y_partials = self.outputLayerPartials(x)
        y_partials = jax.nn.softmax(y_partials)

        y_ising = self.outputLayerIsing(x)
        y_ising = jax.nn.softplus(y_ising)

        return jnp.concatenate((y_ising, y_partials), axis=-1)


if __name__ == "__main__":
    model = SmallDNN(n_hidden=2, dim_hidden=12, rngs=nnx.Rngs(0))
    x = jnp.ones((1, _n_inputs))
    print(f"{x=} and {model(x)=}")
