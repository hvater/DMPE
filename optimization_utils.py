from functools import partial

from tqdm.notebook import tqdm

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from model_utils import simulate_ahead
from density_estimation import update_kde_grid_multiple_observations
from metrics import JSDLoss


@jax.jit
def soft_penalty(a, a_max=1):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    penalty = jnp.sum(jax.nn.relu(jnp.abs(a) - a_max), axis=(-2, -1))
    return jnp.squeeze(penalty)


@partial(jax.jit, static_argnums=(1, 4))
def loss_function(
        actions,
        model,
        init_obs,
        init_state,
        n_steps,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        target_distribution
    ):

    observations = simulate_ahead(
        model=model,
        n_steps=n_steps,
        obs=init_obs,
        state=init_state,
        actions=actions,
        env_state_normalizer=model.env_state_normalizer[0, :],
        action_normalizer=model.action_normalizer[0, :],
        static_params={key: value[0, :] for (key, value) in model.static_params.items()}
    )

    p_est = update_kde_grid_multiple_observations(p_est, x, observations, start_n_measurments, bandwidth)    
    
    loss = JSDLoss(
        p=p_est,
        q=target_distribution
    )

    # TODO: pull this automatically, maybe penalty_kwargs or something
    rho_obs = 1e4
    rho_act = 1e4
    penalty_terms = rho_obs * soft_penalty(a=observations, a_max=1) + rho_act * soft_penalty(a=actions, a_max=1)

    return loss + penalty_terms


def optimize(
        grad_loss_function,
        proposed_actions,
        model,
        init_obs,
        init_state,
        n_steps,
        p_est,
        x,
        start_n_measurments,
        bandwidth,
        target_distribution
    ):

    solver = optax.adabelief(learning_rate=1e-1)
    opt_state = solver.init(proposed_actions)

    # _grad_loss_function = partial(
    #     grad_loss_function,
    #     model=model,
    #     n_steps=n_steps,
    #     x=x,
    #     start_n_measurments=start_n_measurments,
    #     bandwidth=bandwidth,
    #     target_distribution=target_distribution
    # )

    for iter in range(5):
        grad = jax.vmap(
            grad_loss_function,
            in_axes=(0, None, 0, 0, None, 0, None, None, None, None)
        )(
            proposed_actions,
            model,
            init_obs,
            init_state,
            n_steps,
            p_est,
            x,
            start_n_measurments,
            bandwidth,
            target_distribution
        )
        updates, opt_state = solver.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

    # final_loss = jax.vmap(
    #     loss_function,
    #     in_axes=(0, None, 0, 0, None, 0, None, None, None, None)
    # )(
    #     proposed_actions,
    #     model,
    #     init_obs,
    #     init_state,
    #     n_steps,
    #     p_est,
    #     x,
    #     start_n_measurments,
    #     bandwidth,
    #     target_distribution
    # )

    return proposed_actions, 0