from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from exciting_environments.core_env import CoreEnvironment

from dmpe.models.model_utils import simulate_ahead, simulate_ahead_with_env
from dmpe.utils.density_estimation import (
    DensityEstimate,
    update_density_estimate_single_observation,
    update_density_estimate_multiple_observations,
)
from dmpe.utils.metrics import JSDLoss


def soft_penalty(a, a_max=1, penalty_order=2):
    """Computes penalty for the given input. Assumes symmetry in all dimensions."""
    penalties = jax.nn.relu(jnp.abs(a) - a_max)

    penalties = penalties**penalty_order

    penalty = jnp.sum(penalties, axis=(-2, -1))
    return jnp.squeeze(penalty)


@eqx.filter_jit
def loss_function(
    model,
    init_obs: jnp.ndarray,
    init_state,
    actions: jnp.ndarray,
    density_estimate: DensityEstimate,
    tau: float,
    consider_action_distribution: bool,
    target_distribution: jnp.ndarray,
    penalty_function: Callable,
) -> jnp.ndarray:
    """Predicts a trajectory based on the given actions and the model and computes the
    corresponding loss value.

    Args:
        model: The model to use for the prediction
        init_obs: The initial observation from which to start the simulation
        actions: The actions to apply in each step of the simulation, the length
            of the first dimension of this array determine the lenght of the
            output.
        density_estimate: The current estimate of the data density
        tau: The sampling time for the model
        target_distribution: The goal distribution of the data. The JSD loss is computed
            w.r.t. this distribution
    """
    observations = simulate_ahead(model=model, init_obs=init_obs, actions=actions, tau=tau)

    if consider_action_distribution:
        predicted_density_estimate = update_density_estimate_multiple_observations(
            density_estimate, jnp.concatenate([observations[0:-1, :], actions], axis=-1)  # observations
        )
    else:
        predicted_density_estimate = update_density_estimate_multiple_observations(density_estimate, observations)

    loss = JSDLoss(
        p=predicted_density_estimate.p / jnp.sum(predicted_density_estimate.p),
        q=target_distribution / jnp.sum(target_distribution),
    )
    penalty_terms = penalty_function(observations, actions)
    return loss + penalty_terms


@eqx.filter_jit
def optimize_actions(
    loss_function,
    grad_loss_function,
    proposed_actions,
    model,
    optimizer,
    init_obs,
    init_state,
    density_estimate,
    n_opt_steps,
    tau,
    consider_action_distribution,
    target_distribution,
    penalty_function,
):
    """Uses the model to compute the effect of actions onto the observation trajectory to
    optimize the actions w.r.t. the given (gradient of the) loss function.
    """
    opt_state = optimizer.init(proposed_actions)

    def body_fun(i, carry):
        proposed_actions, opt_state = carry
        value, grad = grad_loss_function(
            model,
            init_obs,
            init_state,
            proposed_actions,
            density_estimate,
            tau,
            consider_action_distribution,
            target_distribution,
            penalty_function,
        )
        updates, opt_state = optimizer.update(grad, opt_state, proposed_actions)
        proposed_actions = optax.apply_updates(proposed_actions, updates)

        # proposed_actions = proposed_actions - lr * grad

        return (proposed_actions, opt_state)

    proposed_actions, _ = jax.lax.fori_loop(0, n_opt_steps, body_fun, (proposed_actions, opt_state))

    loss = loss_function(
        model,
        init_obs,
        init_state,
        proposed_actions,
        density_estimate,
        tau,
        consider_action_distribution,
        target_distribution,
        penalty_function,
    )

    return proposed_actions, loss


def optimize_actions_multistart(
    loss_function,
    grad_loss_function,
    all_proposed_actions,
    model,
    optimizer,
    init_obs,
    init_state,
    density_estimate,
    n_opt_steps,
    tau,
    consider_action_distribution,
    target_distribution,
    penalty_function,
):
    """Uses the model to compute the effect of actions onto the observation trajectory to
    optimize the actions w.r.t. the given (gradient of the) loss function. This is done for
    multiple sets of proposed actions and the best set is returned.

    """
    assert all_proposed_actions.ndim == 3, "proposed_actions must have shape (n_starts, n_opt_steps, action_dim)"

    all_optimized_actions, all_losses = jax.vmap(
        optimize_actions,
        in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None, None),
    )(
        loss_function,
        grad_loss_function,
        all_proposed_actions,
        model,
        optimizer,
        init_obs,
        init_state,
        density_estimate,
        n_opt_steps,
        tau,
        consider_action_distribution,
        target_distribution,
        penalty_function,
    )

    best_idx = jnp.argmin(all_losses)
    return all_optimized_actions[best_idx], all_losses[best_idx]


class Exciter(eqx.Module):
    """A class that carries the necessary tools for excitation input computations.

    Args:
        loss_function:
        grad_loss_function: The gradient of the loss function w.r.t. the actions as
            a callable function
        excitiation_optimizer: The optimizer for the excitation input computation
        tau: The time step length of the simulation
        n_opt_steps: Number of SGD steps per iteration
        consider_action_distribution: Whether to consider the action as part of the
            feature vector
        target_distribution: The targeted distribution for the data density
        penalty_function:
        clip_action: Whether to clip the actions to the action space limits
        n_starts: Number of random parallel starts for the optimization
        reuse_proposed_actions: Whether to reuse the proposed actions from the previous
    """

    loss_function: Callable
    grad_loss_function: Callable
    excitation_optimizer: optax._src.base.GradientTransformationExtraArgs
    tau: float
    n_opt_steps: int
    consider_action_distribution: bool
    target_distribution: jnp.ndarray
    penalty_function: Callable
    clip_action: bool
    n_starts: int
    reuse_proposed_actions: bool

    @eqx.filter_jit
    def choose_action(
        self,
        obs: jnp.ndarray,
        state,
        model,
        density_estimate: DensityEstimate,
        proposed_actions: jnp.ndarray,
        expl_key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray, DensityEstimate]:
        """Chooses the next action to take, updates the density estimate and
        proposes future actions.

        Args:
            obs: The current observations from which to start
            model: The current model of the environment used for the prediction
            density_estimate: The current estimate of the data density without
                the current step k
            propsed_actions: An initial proposition of actions to take

        Returns:
            action: The chosen action
            next_proposed_actions: An initial proposition for future actions
            density_estimate: The updated density estimate now incorporating
                the current step k
        """

        if self.reuse_proposed_actions:
            n_random_starts = self.n_starts - 1

        expl_key, new_proposed_actions_key, expl_action_key, _ = jax.random.split(expl_key, 4)

        if n_random_starts > 0:
            random_proposed_actions = jax.random.uniform(
                key=new_proposed_actions_key, shape=(n_random_starts, *proposed_actions.shape), minval=-1, maxval=1
            )

            if self.reuse_proposed_actions:
                all_proposed_actions = jnp.concatenate([proposed_actions[None, :], random_proposed_actions], axis=0)
            else:
                all_proposed_actions = random_proposed_actions
        else:
            all_proposed_actions = proposed_actions[None, :]

        proposed_actions, loss = optimize_actions_multistart(
            loss_function=self.loss_function,
            grad_loss_function=self.grad_loss_function,
            all_proposed_actions=all_proposed_actions,
            model=model,
            optimizer=self.excitation_optimizer,
            init_obs=obs,
            init_state=state,
            density_estimate=density_estimate,
            n_opt_steps=self.n_opt_steps,
            tau=self.tau,
            consider_action_distribution=self.consider_action_distribution,
            target_distribution=self.target_distribution,
            penalty_function=self.penalty_function,
        )

        action = proposed_actions[0, :]
        if self.clip_action:
            action = jnp.clip(action, -1, 1)

        next_proposed_actions = proposed_actions.at[:-1, :].set(proposed_actions[1:, :])

        new_proposed_action = jax.random.uniform(key=expl_action_key, minval=-1, maxval=1)
        next_proposed_actions = next_proposed_actions.at[-1, :].set(new_proposed_action)

        # update grid KDE with x_k and u_k
        if self.consider_action_distribution:
            density_estimate = update_density_estimate_single_observation(
                density_estimate, jnp.concatenate([obs, action], axis=-1)
            )
        else:
            density_estimate = update_density_estimate_single_observation(density_estimate, obs)

        return action, next_proposed_actions, density_estimate, loss, expl_key
