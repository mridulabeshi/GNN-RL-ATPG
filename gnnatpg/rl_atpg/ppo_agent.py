"""
ppo_agent.py - Proximal Policy Optimization (PPO) Agent for ATPG

Implements PPO with GNN-based policy and value networks for learning
ATPG search strategies.

The agent uses:
    Real circuit graph
    +
    Dynamic ATPG state
    +
    PPO

Training history is saved for later Matplotlib visualization.
"""

import logging
import os
from typing import Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from gnnatpg.gnn.circuit_gnn import CircuitGNN
from gnnatpg.rl_atpg.atpg_env import (
    ATPGEnvironment,
    ATPGAction,
)


logger = logging.getLogger(__name__)


# ============================================================
# ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):

    def __init__(
        self,
        gnn: CircuitGNN,
        num_actions: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.gnn = gnn

        gnn_out_dim = gnn.out_dim

        # ----------------------------------------------------
        # ATPG state encoder
        # ----------------------------------------------------

        self.state_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Combined representation
        # ----------------------------------------------------

        combined_dim = (
            gnn_out_dim * 2
            + hidden_dim
        )

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------

        self.actor = nn.Sequential(
            nn.Linear(
                combined_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                num_actions,
            ),
        )

        # ----------------------------------------------------
        # Critic
        # ----------------------------------------------------

        self.critic = nn.Sequential(
            nn.Linear(
                combined_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    # ========================================================
    # STATE ENCODER
    # ========================================================

    def _encode_state(
        self,
        state_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        if state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(0)

        if state_tensor.dim() != 3:
            raise ValueError(
                "Expected ATPG state tensor "
                f"[B, N, 5], got "
                f"{tuple(state_tensor.shape)}"
            )

        if state_tensor.size(0) != batch_size:
            raise ValueError(
                f"State batch size "
                f"{state_tensor.size(0)} does not match "
                f"graph batch size {batch_size}"
            )

        if state_tensor.size(-1) != 5:
            raise ValueError(
                f"Expected 5 state features, "
                f"got {state_tensor.size(-1)}"
            )

        state_mean = state_tensor.mean(
            dim=1
        )

        state_max = state_tensor.max(
            dim=1
        ).values

        state_summary = torch.cat(
            [
                state_mean,
                state_max,
            ],
            dim=-1,
        )

        return self.state_encoder(
            state_summary
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        graph_data: Data,
        state_tensor: torch.Tensor,
    ):

        if graph_data.x is None:
            raise ValueError(
                "graph_data.x is missing"
            )

        if graph_data.edge_index is None:
            raise ValueError(
                "graph_data.edge_index is missing"
            )

        # ----------------------------------------------------
        # GNN
        # ----------------------------------------------------

        node_emb, graph_emb = self.gnn(
            graph_data
        )

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        if (
            hasattr(graph_data, "batch")
            and graph_data.batch is not None
        ):

            batch = graph_data.batch

        else:

            batch = torch.zeros(
                node_emb.size(0),
                dtype=torch.long,
                device=node_emb.device,
            )

        # ----------------------------------------------------
        # Global node pooling
        # ----------------------------------------------------

        node_emb_global = (
            global_mean_pool(
                node_emb,
                batch,
            )
        )

        batch_size = graph_emb.size(0)

        # ----------------------------------------------------
        # State representation
        # ----------------------------------------------------

        state_emb = self._encode_state(
            state_tensor,
            batch_size,
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        combined = torch.cat(
            [
                node_emb_global,
                graph_emb,
                state_emb,
            ],
            dim=-1,
        )

        action_logits = self.actor(
            combined
        )

        state_value = self.critic(
            combined
        )

        return (
            action_logits,
            state_value,
        )

    # ========================================================
    # ACTION
    # ========================================================

    def get_action(
        self,
        graph_data: Data,
        state_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ):

        with torch.no_grad():

            action_logits, value = (
                self.forward(
                    graph_data,
                    state_tensor,
                )
            )

            if action_mask is not None:

                action_mask = (
                    action_mask.to(
                        action_logits.device
                    )
                )

                action_logits = (
                    action_logits
                    + (
                        1.0 - action_mask
                    ) * (-1e8)
                )

            probs = F.softmax(
                action_logits,
                dim=-1,
            )

            dist = Categorical(probs)

            action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).item(),
            value.squeeze(-1).item(),
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        graph_data: Data,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):

        action_logits, values = (
            self.forward(
                graph_data,
                states,
            )
        )

        probs = F.softmax(
            action_logits,
            dim=-1,
        )

        dist = Categorical(probs)

        log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        return (
            log_probs,
            entropy,
            values.squeeze(-1),
        )


# ============================================================
# ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:

    def __init__(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        self.advantages = []
        self.returns = []

    def add(
        self,
        state,
        action,
        reward,
        value,
        log_prob,
        done,
    ):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):

        advantages = []
        returns = []

        gae = 0.0

        values = (
            self.values
            + [last_value]
        )

        for t in reversed(
            range(len(self.rewards))
        ):

            if self.dones[t]:

                delta = (
                    self.rewards[t]
                    - values[t]
                )

                gae = delta

            else:

                delta = (
                    self.rewards[t]
                    + gamma
                    * values[t + 1]
                    - values[t]
                )

                gae = (
                    delta
                    + gamma
                    * gae_lambda
                    * gae
                )

            advantages.insert(
                0,
                gae,
            )

            returns.insert(
                0,
                gae + values[t],
            )

        self.advantages = advantages
        self.returns = returns

    def get_batches(
        self,
        batch_size: int,
    ):

        n = len(self.states)

        indices = np.arange(n)

        np.random.shuffle(indices)

        for start in range(
            0,
            n,
            batch_size,
        ):

            end = start + batch_size

            batch_indices = indices[
                start:end
            ]

            yield {
                "states": [
                    self.states[i]
                    for i in batch_indices
                ],

                "actions": torch.tensor(
                    [
                        self.actions[i]
                        for i in batch_indices
                    ],
                    dtype=torch.long,
                ),

                "old_log_probs": torch.tensor(
                    [
                        self.log_probs[i]
                        for i in batch_indices
                    ],
                    dtype=torch.float,
                ),

                "advantages": torch.tensor(
                    [
                        self.advantages[i]
                        for i in batch_indices
                    ],
                    dtype=torch.float,
                ),

                "returns": torch.tensor(
                    [
                        self.returns[i]
                        for i in batch_indices
                    ],
                    dtype=torch.float,
                ),
            }

    def clear(self):

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()

    @property
    def size(self):

        return len(self.states)


# ============================================================
# PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(
        self,
        gnn: CircuitGNN,
        env: ATPGEnvironment,
        config: dict,
        device: str = "cpu",
        graph_data: Optional[Data] = None,
    ):

        self.device = torch.device(
            device
        )

        self.env = env

        self.config = config.get(
            "rl",
            {},
        )

        self.training_config = config.get(
            "training",
            {},
        )

        # ----------------------------------------------------
        # Real graph
        # ----------------------------------------------------

        if graph_data is None:

            raise ValueError(
                "graph_data is required."
            )

        self.graph_data = graph_data.to(
            self.device
        )

        if self.graph_data.x is None:
            raise ValueError(
                "graph_data.x is missing."
            )

        if self.graph_data.edge_index is None:
            raise ValueError(
                "graph_data.edge_index is missing."
            )

        if self.graph_data.x.dim() != 2:
            raise ValueError(
                "Expected graph features [N,F]"
            )

        if (
            self.graph_data.x.size(1)
            != gnn.in_dim
        ):

            raise ValueError(
                f"GNN expects {gnn.in_dim} "
                f"features but graph has "
                f"{self.graph_data.x.size(1)}"
            )

        # ----------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------

        self.learning_rate = self.config.get(
            "learning_rate",
            3e-4,
        )

        self.gamma = self.config.get(
            "gamma",
            0.99,
        )

        self.gae_lambda = self.config.get(
            "gae_lambda",
            0.95,
        )

        self.clip_epsilon = self.config.get(
            "clip_epsilon",
            0.2,
        )

        self.entropy_coef = self.config.get(
            "entropy_coef",
            0.01,
        )

        self.value_loss_coef = self.config.get(
            "value_loss_coef",
            0.5,
        )

        self.max_grad_norm = self.config.get(
            "max_grad_norm",
            0.5,
        )

        self.num_episodes = self.config.get(
            "num_episodes",
            10000,
        )

        self.max_steps = self.config.get(
            "max_steps_per_episode",
            500,
        )

        self.batch_size = self.config.get(
            "batch_size",
            64,
        )

        self.update_interval = self.config.get(
            "update_interval",
            2048,
        )

        # ----------------------------------------------------
        # Actor critic
        # ----------------------------------------------------

        num_actions = len(
            ATPGAction
        )

        self.actor_critic = ActorCritic(
            gnn,
            num_actions=num_actions,
        ).to(self.device)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        self.buffer = RolloutBuffer()

        # ----------------------------------------------------
        # Rolling statistics
        # ----------------------------------------------------

        self.episode_rewards = deque(
            maxlen=100
        )

        self.episode_fault_coverage = deque(
            maxlen=100
        )

        # ----------------------------------------------------
        # COMPLETE HISTORY FOR PLOTTING
        # ----------------------------------------------------

        self.reward_history = []

        self.coverage_history = []

        self.detected_fault_history = []

        self.fault_index_history = []

        self.fault_type_history = []

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self) -> dict:

        logger.info(
            "Starting PPO training for ATPG..."
        )

        total_steps = 0

        best_coverage = 0.0

        for episode in range(
            self.num_episodes
        ):

            episode_reward, episode_info = (
                self._collect_rollout()
            )

            self.episode_rewards.append(
                episode_reward
            )

            total_steps += (
                self.buffer.size
            )

            # ------------------------------------------------
            # PPO update
            # ------------------------------------------------

            if (
                total_steps
                >= self.update_interval
            ):

                update_stats = (
                    self._update_policy()
                )

                self.buffer.clear()

                total_steps = 0

                avg_reward = np.mean(
                    self.episode_rewards
                )

                avg_coverage = (
                    np.mean(
                        self.episode_fault_coverage
                    )
                    if self.episode_fault_coverage
                    else 0.0
                )

                if episode % 10 == 0:

                    logger.info(
                        f"Episode {episode}: "
                        f"avg_reward="
                        f"{avg_reward:.3f}, "
                        f"coverage="
                        f"{avg_coverage:.3f}, "
                        f"value_loss="
                        f"{update_stats.get('value_loss', 0):.4f}, "
                        f"policy_loss="
                        f"{update_stats.get('policy_loss', 0):.4f}"
                    )

                # ------------------------------------------------
                # Best model
                # ------------------------------------------------

                if (
                    avg_coverage
                    > best_coverage
                ):

                    best_coverage = (
                        avg_coverage
                    )

                    self._save_checkpoint(
                        "best_model"
                    )

            # ------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------

            save_interval = (
                self.training_config.get(
                    "save_interval",
                    10,
                )
            )

            if (
                episode % save_interval
                == 0
            ):

                self._save_checkpoint(
                    f"episode_{episode}"
                )

        # ====================================================
        # SAVE COMPLETE TRAINING HISTORY
        # ====================================================

        history_dir = (
            self.training_config.get(
                "log_dir",
                "data/models/logs",
            )
        )

        os.makedirs(
            history_dir,
            exist_ok=True,
        )

        history_path = os.path.join(
            history_dir,
            "ppo_training_history.pt",
        )

        torch.save(
            {
                "rewards":
                    self.reward_history,

                "coverage":
                    self.coverage_history,

                "detected_fault":
                    self.detected_fault_history,

                "fault_index":
                    self.fault_index_history,

                "fault_type":
                    self.fault_type_history,
            },
            history_path,
        )

        logger.info(
            f"Saved PPO training history: "
            f"{history_path}"
        )

        return {
            "final_reward": (
                np.mean(
                    self.episode_rewards
                )
                if self.episode_rewards
                else 0.0
            ),

            "best_coverage":
                best_coverage,

            "total_episodes":
                self.num_episodes,

            "history_file":
                history_path,
        }

    # ========================================================
    # COLLECT ROLLOUT
    # ========================================================

    def _collect_rollout(
        self,
    ) -> tuple[float, dict]:

        state_tensor, info = (
            self.env.reset()
        )

        state_tensor = (
            state_tensor.to(
                self.device
            )
        )

        episode_reward = 0.0

        done = False

        step = 0

        while (
            not done
            and step < self.max_steps
        ):

            graph_data = (
                self._build_graph_for_state(
                    state_tensor
                )
            )

            action, log_prob, value = (
                self.actor_critic.get_action(
                    graph_data,
                    state_tensor.unsqueeze(0),
                )
            )

            next_state, reward, done, info = (
                self.env.step(action)
            )

            next_state = (
                next_state.to(
                    self.device
                )
            )

            self.buffer.add(
                state_tensor,
                action,
                reward,
                value,
                log_prob,
                done,
            )

            episode_reward += reward

            state_tensor = next_state

            step += 1

        # ----------------------------------------------------
        # Final value
        # ----------------------------------------------------

        with torch.no_grad():

            graph_data = (
                self._build_graph_for_state(
                    state_tensor
                )
            )

            _, last_value = (
                self.actor_critic(
                    graph_data,
                    state_tensor.unsqueeze(0),
                )
            )

            last_value = (
                last_value
                .squeeze(-1)
                .item()
            )

        # ----------------------------------------------------
        # GAE
        # ----------------------------------------------------

        self.buffer.compute_gae(
            last_value,
            self.gamma,
            self.gae_lambda,
        )

        # ====================================================
        # SAVE EPISODE HISTORY
        # ====================================================

        detected = bool(
            info.get(
                "fault_detected",
                False,
            )
        )

        fault_index = int(
            info.get(
                "fault_index",
                self.env.current_fault_idx,
            )
        )

        fault_type = int(
            info.get(
                "fault_type",
                -1,
            )
        )

        # Per-episode coverage
        episode_coverage = (
            1.0
            if detected
            else 0.0
        )

        self.reward_history.append(
            float(episode_reward)
        )

        self.coverage_history.append(
            episode_coverage
        )

        self.detected_fault_history.append(
            int(detected)
        )

        self.fault_index_history.append(
            fault_index
        )

        self.fault_type_history.append(
            fault_type
        )

        # Rolling coverage used by PPO
        self.episode_fault_coverage.append(
            episode_coverage
        )

        return (
            episode_reward,
            info,
        )

    # ========================================================
    # PPO UPDATE
    # ========================================================

    def _update_policy(self) -> dict:

        if self.buffer.size == 0:
            return {}

        self.actor_critic.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        num_updates = 0

        for batch in self.buffer.get_batches(
            self.batch_size
        ):

            graph_data = (
                self._build_batch_graph(
                    batch["states"]
                )
            )

            states = torch.stack(
                batch["states"]
            ).to(self.device)

            actions = (
                batch["actions"]
                .to(self.device)
            )

            old_log_probs = (
                batch["old_log_probs"]
                .to(self.device)
            )

            advantages = (
                batch["advantages"]
                .to(self.device)
            )

            returns = (
                batch["returns"]
                .to(self.device)
            )

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

            (
                new_log_probs,
                entropy,
                values,
            ) = self.actor_critic.evaluate(
                graph_data,
                states,
                actions,
            )

            ratio = torch.exp(
                new_log_probs
                - old_log_probs
            )

            surr1 = (
                ratio
                * advantages
            )

            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                )
                * advantages
            )

            policy_loss = (
                -torch.min(
                    surr1,
                    surr2,
                ).mean()
            )

            value_loss = F.mse_loss(
                values,
                returns,
            )

            loss = (
                policy_loss
                + self.value_loss_coef
                * value_loss
                - self.entropy_coef
                * entropy.mean()
            )

            self.optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

            total_policy_loss += (
                policy_loss.item()
            )

            total_value_loss += (
                value_loss.item()
            )

            total_entropy += (
                entropy.mean().item()
            )

            num_updates += 1

        self.actor_critic.eval()

        return {
            "policy_loss":
                total_policy_loss
                / max(num_updates, 1),

            "value_loss":
                total_value_loss
                / max(num_updates, 1),

            "entropy":
                total_entropy
                / max(num_updates, 1),
        }

    # ========================================================
    # REAL CIRCUIT GRAPH
    # ========================================================

    def _build_graph_for_state(
        self,
        state_tensor: torch.Tensor,
    ) -> Data:

        if self.graph_data is None:

            raise RuntimeError(
                "Real circuit graph has not been provided."
            )

        if (
            state_tensor.size(0)
            != self.graph_data.x.size(0)
        ):

            raise ValueError(
                "ATPG state node count does not "
                "match circuit graph node count."
            )

        return self.graph_data

    # ========================================================
    # BATCH GRAPH
    # ========================================================

    def _build_batch_graph(
        self,
        states: list,
    ) -> Batch:

        if not states:

            raise ValueError(
                "Cannot build graph batch "
                "from empty states."
            )

        graphs = [
            self._build_graph_for_state(
                state
            )
            for state in states
        ]

        batch = Batch.from_data_list(
            graphs
        )

        return batch.to(
            self.device
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def _save_checkpoint(
        self,
        name: str,
    ):

        save_dir = (
            self.training_config.get(
                "checkpoint_dir",
                "data/models/checkpoints",
            )
        )

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        path = os.path.join(
            save_dir,
            f"{name}.pt",
        )

        torch.save(
            {
                "model_state_dict":
                    self.actor_critic.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),
            },
            path,
        )

        logger.info(
            f"Saved checkpoint: {path}"
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        path: str,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor_critic.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        logger.info(
            f"Loaded checkpoint: {path}"
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        graph_data: Data,
        state: torch.Tensor,
    ) -> int:

        self.actor_critic.eval()

        graph_data = graph_data.to(
            self.device
        )

        state = state.to(
            self.device
        )

        with torch.no_grad():

            action, _, _ = (
                self.actor_critic.get_action(
                    graph_data,
                    state.unsqueeze(0),
                )
            )

        return action