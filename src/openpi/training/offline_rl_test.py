"""CPU tests for the PT4FM JAX offline-RL loss (no GPU / no real model needed).

Run: JAX_PLATFORMS=cpu pytest src/openpi/training/offline_rl_test.py
"""

import dataclasses

import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.offline_rl as orl


def _orl_config(**over):
    base = _config.OfflineRLConfig()
    return dataclasses.replace(base, **over)


def test_awr_weight_beta_inf_is_uniform():
    a = jnp.array([-2.0, 0.0, 1.0, 3.0])
    m = jnp.ones(4, bool)
    w = orl.awr_weight(a, beta=1e12, w_max=1e12, mask=m)
    assert jnp.allclose(w, 1.0, atol=1e-3)


def test_awr_weight_mean_one_and_monotonic():
    a = jnp.array([-2.0, 0.0, 1.0, 3.0])
    m = jnp.ones(4, bool)
    w = orl.awr_weight(a, beta=1.0, w_max=1e12, mask=m)
    assert abs(float(w.mean()) - 1.0) < 1e-4
    assert bool(jnp.all(jnp.diff(w) >= 0))


def test_awr_weight_mask_zeros_outside():
    a = jnp.array([5.0, 0.0, 0.0])
    m = jnp.array([False, True, True])
    w = orl.awr_weight(a, beta=1.0, w_max=1e12, mask=m)
    assert float(w[0]) == 0.0
    assert abs(float((w[1] + w[2]) / 2) - 1.0) < 1e-4


def test_cfg_routing_positive_only():
    adv = jnp.array([True, True, False, False])
    r = orl.cfg_routing_masks(adv, positive_only_conditional=True, uncond_prob=0.0, rng=jax.random.key(0))
    assert bool(jnp.all(r["pos_cond"] == adv))
    assert bool(jnp.all(~r["neg_cond"]))
    assert bool(jnp.all(r["conditional"] == adv))


class _MockModel:
    def train(self):
        pass

    def compute_loss(self, rng, obs, actions, train=False):
        b, ah = actions.shape[0], actions.shape[1]
        return jnp.arange(b * ah, dtype=jnp.float32).reshape(b, ah)


def _obs(b, ad, L):
    return _model.Observation.from_dict({
        "image": {"base_0_rgb": jnp.zeros((b, 4, 4, 3))},
        "image_mask": {"base_0_rgb": jnp.ones((b,), bool)},
        "state": jnp.zeros((b, ad)),
        "tokenized_prompt": jnp.zeros((b, L), int),
        "tokenized_prompt_mask": jnp.ones((b, L), bool),
    })


def test_recap_parity():
    """awr off + reuse_unconditional + lambda=1 + cfg off == jnp.mean(compute_loss)."""
    b, ah, ad, L = 6, 4, 7, 8
    obs, actions = _obs(b, ad, L), jnp.zeros((b, ah, ad))
    rl_info = {
        "advantage": jnp.array([1, 1, 1, 0, 0, 0]).astype(bool),
        "advantage_weight": jnp.ones(b),
        "is_demo": jnp.ones(b, bool),
    }
    cfg = _orl_config(
        awr=_config.AWRConfig(enabled=False),
        sft_aux=_config.SFTAuxConfig(mode="reuse_unconditional", weight=1.0),
        cfg=_config.CFGConfig(enabled=False),
    )
    loss, m = orl.compute_offline_rl_loss(_MockModel(), jax.random.key(0), obs, actions, rl_info, cfg)
    ref = jnp.mean(_MockModel().compute_loss(None, obs, actions))
    assert jnp.allclose(loss, ref, atol=1e-4)
    assert jnp.allclose(m["rl_loss"] + m["sft_loss"], loss, atol=1e-4)


def test_awr_and_separate_forward_finite():
    b, ah, ad, L = 6, 4, 7, 8
    obs, actions = _obs(b, ad, L), jnp.zeros((b, ah, ad))
    rl_info = {
        "advantage": jnp.array([1, 1, 1, 0, 0, 0]).astype(bool),
        "advantage_weight": jnp.array([2.0, 1.0, 0.5, 0.0, 0.0, 0.0]),
        "is_demo": jnp.ones(b, bool),
    }
    awr_on = _orl_config(awr=_config.AWRConfig(enabled=True, beta=0.7),
                         sft_aux=_config.SFTAuxConfig(mode="reuse_unconditional", weight=1.0),
                         cfg=_config.CFGConfig(enabled=False))
    l1, _ = orl.compute_offline_rl_loss(_MockModel(), jax.random.key(1), obs, actions, rl_info, awr_on)
    sep = dataclasses.replace(awr_on, sft_aux=_config.SFTAuxConfig(mode="separate_forward", weight=0.5))
    l2, _ = orl.compute_offline_rl_loss(_MockModel(), jax.random.key(2), obs, actions, rl_info, sep)
    assert bool(jnp.isfinite(l1)) and bool(jnp.isfinite(l2))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
