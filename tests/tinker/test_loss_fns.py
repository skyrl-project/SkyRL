import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tinker.loss_fns import (
    LOSS_FUNCTION_MAP,
    LossFnConfig,
    cispo_loss,
)


def test_cispo_loss_clipping_and_masking():
    target_logprobs = jnp.array([0.0, 5.0, -5.0], dtype=jnp.float32)
    sampling_logprobs = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    advantages = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    loss_mask = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
    )

    actual = np.asarray(cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config))

    ratios = np.exp(np.asarray(target_logprobs - sampling_logprobs))
    clipped_ratios = np.clip(ratios, 0.8, 1.2)
    expected = -(clipped_ratios * np.asarray(target_logprobs) * np.asarray(advantages))
    expected[2] = 0.0

    assert np.allclose(actual, expected)


def test_cispo_stops_gradient_through_clipped_ratio():
    sampling_logprobs = jnp.array([0.1, -0.2], dtype=jnp.float32)
    advantages = jnp.array([1.3, -0.7], dtype=jnp.float32)
    loss_mask = jnp.ones((2,), dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
    )

    def total_loss(target_logprobs):
        return cispo_loss(target_logprobs, loss_mask, sampling_logprobs, advantages, loss_fn_config).sum()

    target_logprobs = jnp.array([0.0, 0.2], dtype=jnp.float32)
    grad = np.asarray(jax.grad(total_loss)(target_logprobs))

    clipped_ratio = np.clip(
        np.exp(np.asarray(target_logprobs - sampling_logprobs)),
        0.8,
        1.2,
    )
    expected_grad = -(clipped_ratio * np.asarray(advantages))

    assert np.allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "importance_sampling", "ppo", "cispo"])
def test_loss_fns_masked_extreme_and_nonfinite_inputs_keep_gradients_finite(loss_name: str):
    """Masked positions (loss_mask == 0) must not produce NaN gradients under jax.grad."""
    loss_fn = LOSS_FUNCTION_MAP[loss_name]
    target_logprobs = jnp.array([-1.0, -5.0, -10.0], dtype=jnp.float32)
    sampling_logprobs = jnp.array([-1.2, -200.0, -jnp.inf], dtype=jnp.float32)
    advantages = jnp.array([1.5, 0.0, jnp.nan], dtype=jnp.float32)
    loss_mask = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jnp.array(0.8, dtype=jnp.float32),
        clip_high_threshold=jnp.array(1.2, dtype=jnp.float32),
    )

    def total_loss(lp):
        return loss_fn(lp, loss_mask, sampling_logprobs, advantages, loss_fn_config).sum()

    loss_val, grad = jax.value_and_grad(total_loss)(target_logprobs)
    assert np.isfinite(np.asarray(loss_val))
    grad_np = np.asarray(grad)
    assert np.isfinite(grad_np).all(), f"{loss_name}: non-finite grad {grad_np}"
    assert grad_np[1] == 0.0
    assert grad_np[2] == 0.0

    # Active position 0 must match the unmasked single-token loss and gradient exactly.
    ref_val, ref_grad = jax.value_and_grad(
        lambda lp0: loss_fn(
            lp0,
            loss_mask[:1],
            sampling_logprobs[:1],
            advantages[:1],
            loss_fn_config,
        ).sum()
    )(target_logprobs[:1])
    assert np.allclose(np.asarray(loss_val), np.asarray(ref_val), rtol=1e-6, atol=1e-6)
    assert np.allclose(grad_np[:1], np.asarray(ref_grad), rtol=1e-6, atol=1e-6)
