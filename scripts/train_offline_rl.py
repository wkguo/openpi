"""PT4FM offline-RL trainer (JAX) — AWR-CFG + SFT-aux flow loss for pi0.5.

A thin fork of ``scripts/train.py`` that swaps in the PT4FM offline-RL data loader
(carries per-frame advantage + guidance tokens) and loss
(:func:`openpi.training.offline_rl.compute_offline_rl_loss`). Everything else —
``init_train_state``, checkpoint/restore, sharding, wandb — is reused from
``train.py``, so this resumes directly from a finetuned JAX pi0.5 checkpoint via
``config.weight_loader`` (no JAX->PyTorch conversion).

Usage:
    uv run scripts/train_offline_rl.py pine_foundry_rl --exp-name pine_rl \
        --offline-rl.advantage-tag v1_N10_q30
"""

import dataclasses
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.offline_rl as offline_rl
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Reuse the (non-trivial) train-state init / logging / wandb helpers verbatim.
from train import init_logging, init_train_state, init_wandb


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
) -> tuple[training_utils.TrainState, dict]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions, rl_info = batch

    def loss_fn(model, rng, observation, actions, rl_info):
        loss, metrics = offline_rl.compute_offline_rl_loss(
            model, rng, observation, actions, rl_info, config.offline_rl
        )
        return loss, metrics

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, rl_info
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    metrics = {**metrics, "grad_norm": optax.global_norm(grads)}
    return new_state, metrics


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running OFFLINE-RL on: {platform.node()}")
    if config.offline_rl is None:
        raise ValueError(f"Config '{config.name}' has no offline_rl block; use e.g. pine_foundry_rl.")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must divide device count {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = offline_rl.create_offline_rl_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Offline-RL batch:\n{training_utils.array_tree_to_info(batch)}")
    obs0 = batch[0]
    wandb.log(
        {"camera_views": [
            wandb.Image(np.concatenate([np.array(img[i]) for img in obs0.images.values()], axis=1))
            for i in range(min(5, len(next(iter(obs0.images.values())))))
        ]},
        step=0,
    )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                     total=config.num_train_steps, dynamic_ncols=True)
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
