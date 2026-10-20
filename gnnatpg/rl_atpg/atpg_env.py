"""
ATPG Reinforcement Learning Environment

FAST / FAULT-ORIENTED VERSION

This environment uses a simplified ATPG abstraction:

    State:
        Circuit node values
        Target stuck-at fault
        Fault indicator
        D/J frontier information

    Action:
        SET_0
        SET_1
        BACKTRACE_PI_0
        BACKTRACE_PI_1
        JUSTIFY_0
        JUSTIFY_1
        ADVANCE
        NO_OP

    Fast-mode behavior:
        SET_0 / SET_1 directly assign the current fault node.

    SA0:
        Fault is detected when the fault node is assigned 1.

    SA1:
        Fault is detected when the fault node is assigned 0.

This is intentionally a simplified fault-oriented RL environment.
It does NOT perform complete primary-input justification or
fault propagation through the circuit.

The purpose is to train and demonstrate the GNN + PPO decision
component within the available project timeframe.
"""

import logging
from typing import Optional
from enum import IntEnum

import numpy as np
import torch


logger = logging.getLogger(__name__)


class ATPGAction(IntEnum):
    """Possible ATPG decisions the agent can make."""

    SET_0 = 0
    SET_1 = 1

    BACKTRACE_PI_0 = 2
    BACKTRACE_PI_1 = 3

    JUSTIFY_0 = 4
    JUSTIFY_1 = 5

    ADVANCE = 6
    NO_OP = 7


class ATPGState:
    """Represents the current state of the ATPG process."""

    def __init__(
        self,
        num_nodes: int,
        num_pis: int,
        fault_node_id: int = -1,
        fault_type: int = 0,
    ):
        self.num_nodes = num_nodes
        self.num_pis = num_pis

        # ---------------------------------------------------------
        # Circuit values
        #
        # X = -1
        # 0 = logic 0
        # 1 = logic 1
        # ---------------------------------------------------------
        self.node_values = np.full(
            num_nodes,
            -1.0,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Fault information
        #
        # fault_type:
        #   0 = SA0
        #   1 = SA1
        # ---------------------------------------------------------
        self.fault_node_id = fault_node_id
        self.fault_type = fault_type

        # ---------------------------------------------------------
        # ATPG frontier information
        # ---------------------------------------------------------
        self.d_frontier: list[int] = []
        self.j_frontier: list[int] = []

        # Primary-input assignments
        self.pi_assignments: dict[int, int] = {}

        # ---------------------------------------------------------
        # Tracking
        # ---------------------------------------------------------
        self.num_backtracks = 0
        self.num_conflicts = 0

        self.fault_detected = False
        self.is_done = False

        self.step_count = 0

    def copy(self) -> "ATPGState":
        """Create a copy of the current state."""

        new_state = ATPGState(
            num_nodes=self.num_nodes,
            num_pis=self.num_pis,
            fault_node_id=self.fault_node_id,
            fault_type=self.fault_type,
        )

        new_state.node_values = self.node_values.copy()

        new_state.d_frontier = self.d_frontier.copy()
        new_state.j_frontier = self.j_frontier.copy()

        new_state.pi_assignments = self.pi_assignments.copy()

        new_state.num_backtracks = self.num_backtracks
        new_state.num_conflicts = self.num_conflicts

        new_state.fault_detected = self.fault_detected
        new_state.is_done = self.is_done

        new_state.step_count = self.step_count

        return new_state

    def to_tensor(self) -> torch.Tensor:
        """
        Convert state to node-feature tensor.

        Features per node:

        0 -> current node value
        1 -> target fault indicator
        2 -> D-frontier indicator
        3 -> J-frontier indicator
        4 -> fault detected indicator
        """

        features = np.zeros(
            (self.num_nodes, 5),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Node values
        #
        # -1 -> 0
        #  0 -> 0.5
        #  1 -> 1
        # ---------------------------------------------------------
        features[:, 0] = (
            (self.node_values + 1.0) / 2.0
        )

        # ---------------------------------------------------------
        # Fault node
        # ---------------------------------------------------------
        if 0 <= self.fault_node_id < self.num_nodes:
            features[self.fault_node_id, 1] = 1.0

        # ---------------------------------------------------------
        # D-frontier
        # ---------------------------------------------------------
        for idx in self.d_frontier:
            if 0 <= idx < self.num_nodes:
                features[idx, 2] = 1.0

        # ---------------------------------------------------------
        # J-frontier
        # ---------------------------------------------------------
        for idx in self.j_frontier:
            if 0 <= idx < self.num_nodes:
                features[idx, 3] = 1.0

        # ---------------------------------------------------------
        # Fault detected
        # ---------------------------------------------------------
        if self.fault_detected:
            features[:, 4] = 1.0

        return torch.tensor(
            features,
            dtype=torch.float32,
        )


class ATPGEnvironment:
    """
    RL environment for fault-oriented ATPG.

    FAST MODE:

        The PPO agent directly excites the target fault node.

        SA0:
            SET_1 -> fault detected

        SA1:
            SET_0 -> fault detected

    This is a simplified abstraction used for rapid validation
    of the GNN + PPO architecture.
    """

    def __init__(
        self,
        adj_matrix: np.ndarray,
        gate_types: np.ndarray,
        primary_inputs: list[int],
        primary_outputs: list[int],
        fault_list: list[tuple[int, int]],
        max_steps: int = 500,
        config: Optional[dict] = None,
    ):

        # ---------------------------------------------------------
        # Circuit information
        # ---------------------------------------------------------
        self.adj_matrix = adj_matrix
        self.gate_types = gate_types

        self.num_nodes = adj_matrix.shape[0]

        self.primary_inputs = primary_inputs
        self.primary_outputs = primary_outputs

        # ---------------------------------------------------------
        # Fault list
        #
        # [(node_id, fault_type), ...]
        #
        # fault_type:
        #   0 = SA0
        #   1 = SA1
        # ---------------------------------------------------------
        self.fault_list = fault_list

        self.max_steps = max_steps

        self.config = config or {}

        # ---------------------------------------------------------
        # Reward parameters
        # ---------------------------------------------------------
        rl_config = self.config.get("rl", {})

        self.reward_detected = rl_config.get(
            "reward_fault_detected",
            10.0,
        )

        self.reward_no_conflict = rl_config.get(
            "reward_no_conflict",
            0.1,
        )

        self.penalty_conflict = rl_config.get(
            "penalty_conflict",
            -1.0,
        )

        self.penalty_backtrack = rl_config.get(
            "penalty_backtrack",
            -0.5,
        )

        self.penalty_timeout = rl_config.get(
            "penalty_timeout",
            -5.0,
        )

        # ---------------------------------------------------------
        # Current environment state
        # ---------------------------------------------------------
        self.state: Optional[ATPGState] = None

        # Index of the next fault to process
        self.current_fault_idx = 0

        # Statistics
        self.faults_detected = 0
        self.total_faults = len(fault_list)

        self.total_backtracks = 0

    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        fault_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Reset environment.

        If fault_idx is supplied:
            train/evaluate on that specific fault.

        Otherwise:
            process faults sequentially.

        Example:

            Episode 1 -> fault 0
            Episode 2 -> fault 1
            Episode 3 -> fault 2
            ...
        """

        if self.total_faults == 0:
            raise RuntimeError(
                "Fault list is empty. "
                "Generate stuck-at faults first."
            )

        # ---------------------------------------------------------
        # Select fault
        # ---------------------------------------------------------
        if fault_idx is not None:

            selected_fault_idx = (
                fault_idx % self.total_faults
            )

        else:

            selected_fault_idx = (
                self.current_fault_idx
                % self.total_faults
            )

        # ---------------------------------------------------------
        # Get fault
        # ---------------------------------------------------------
        node_id, fault_type = self.fault_list[
            selected_fault_idx
        ]

        # Convert numpy integer types to Python integers
        node_id = int(node_id)
        fault_type = int(fault_type)

        # ---------------------------------------------------------
        # Create fresh state
        # ---------------------------------------------------------
        self.state = ATPGState(
            num_nodes=self.num_nodes,
            num_pis=len(self.primary_inputs),
            fault_node_id=node_id,
            fault_type=fault_type,
        )

        # ---------------------------------------------------------
        # Reset per-episode counters
        # ---------------------------------------------------------
        self.total_backtracks = 0

        # ---------------------------------------------------------
        # Prepare next fault
        #
        # Only advance automatically when fault_idx wasn't
        # explicitly supplied.
        # ---------------------------------------------------------
        if fault_idx is None:

            self.current_fault_idx = (
                selected_fault_idx + 1
            ) % self.total_faults

        else:

            self.current_fault_idx = selected_fault_idx

        # ---------------------------------------------------------
        # Information returned to PPO/training loop
        # ---------------------------------------------------------
        info = {
            "fault_node": node_id,
            "fault_type": fault_type,
            "fault_index": selected_fault_idx,
            "total_faults": self.total_faults,
        }

        return (
            self.state.to_tensor(),
            info,
        )

    # =============================================================
    # STEP
    # =============================================================

    def step(
        self,
        action: int,
        node_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float, bool, dict]:
        """
        Execute one RL action.

        FAST DIRECT-FAULT MODE:

            SET_0
                directly assigns the target fault node to 0.

            SET_1
                directly assigns the target fault node to 1.

        Detection:

            SA0 + node=1 -> detected
            SA1 + node=0 -> detected
        """

        if self.state is None:
            raise RuntimeError(
                "Environment not reset. "
                "Call reset() first."
            )

        # ---------------------------------------------------------
        # Increment step counter
        # ---------------------------------------------------------
        self.state.step_count += 1

        reward = 0.0

        info = {
            "action": int(action),
            "step": self.state.step_count,
        }

        # =========================================================
        # ACTION 0 / 1
        # =========================================================

        if (
            action == ATPGAction.SET_0
            or action == ATPGAction.SET_1
        ):

            value = (
                1
                if action == ATPGAction.SET_1
                else 0
            )

            # -----------------------------------------------------
            # FAST FAULT-ORIENTED MODE
            #
            # Directly assign the current fault node.
            #
            # This removes the previous heuristic where the GNN
            # embedding selected an arbitrary unassigned node.
            # -----------------------------------------------------

            node_idx = self.state.fault_node_id

            if (
                node_idx < 0
                or node_idx >= self.num_nodes
            ):

                reward = self.penalty_conflict

                self.state.num_conflicts += 1

            else:

                self.state.node_values[
                    node_idx
                ] = value

                reward = self.reward_no_conflict

        # =========================================================
        # BACKTRACE PI
        # =========================================================

        elif (
            action == ATPGAction.BACKTRACE_PI_0
            or action == ATPGAction.BACKTRACE_PI_1
        ):

            value = (
                1
                if action == ATPGAction.BACKTRACE_PI_1
                else 0
            )

            pi_idx = self._select_unassigned_pi()

            if pi_idx is not None:

                self.state.pi_assignments[
                    pi_idx
                ] = value

                self.state.node_values[
                    pi_idx
                ] = value

                reward = self.reward_no_conflict

            else:

                reward = self.penalty_conflict

                self.state.num_conflicts += 1

        # =========================================================
        # JUSTIFICATION
        # =========================================================

        elif (
            action == ATPGAction.JUSTIFY_0
            or action == ATPGAction.JUSTIFY_1
        ):

            value = (
                1
                if action == ATPGAction.JUSTIFY_1
                else 0
            )

            if self.state.j_frontier:

                target = self.state.j_frontier.pop(0)

                self.state.node_values[
                    target
                ] = value

                reward = self.reward_no_conflict

            else:

                reward = (
                    self.penalty_conflict * 0.5
                )

        # =========================================================
        # ADVANCE
        # =========================================================

        elif action == ATPGAction.ADVANCE:

            self.state.is_done = True

            reward = 0.0

        # =========================================================
        # NO OP
        # =========================================================

        elif action == ATPGAction.NO_OP:

            reward = -0.01

        # =========================================================
        # UNKNOWN ACTION
        # =========================================================

        else:

            reward = self.penalty_conflict

            self.state.num_conflicts += 1

        # =========================================================
        # CHECK FAULT DETECTION
        # =========================================================

        if not self.state.fault_detected:

            detected = self._check_fault_detection()

            if detected:

                self.state.fault_detected = True

                reward += self.reward_detected

                self.faults_detected += 1

        # =========================================================
        # CONFLICT CHECK
        # =========================================================

        conflict = self._check_conflict()

        if conflict:

            self.state.num_backtracks += 1

            self.total_backtracks += 1

            reward += self.penalty_backtrack

        # =========================================================
        # UPDATE FRONTIERS
        # =========================================================

        self._update_frontiers()

        # =========================================================
        # TERMINATION
        # =========================================================

        done = (
            self.state.fault_detected
            or self.state.is_done
            or (
                self.state.step_count
                >= self.max_steps
            )
            or (
                self.state.num_conflicts
                >= 10
            )
        )

        # Timeout penalty
        if (
            self.state.step_count >= self.max_steps
            and not self.state.fault_detected
        ):

            reward += self.penalty_timeout

        # =========================================================
        # INFO
        # =========================================================

        info.update(
            {
                "fault_detected":
                    self.state.fault_detected,

                "fault_node":
                    self.state.fault_node_id,

                "fault_type":
                    self.state.fault_type,

                "backtracks":
                    self.state.num_backtracks,

                "conflicts":
                    self.state.num_conflicts,

                "fault_coverage":
                    (
                        1.0
                        if self.state.fault_detected
                        else 0.0
                    ),
            }
        )

        return (
            self.state.to_tensor(),
            float(reward),
            bool(done),
            info,
        )

    # =============================================================
    # FAULT DETECTION
    # =============================================================

    def _check_fault_detection(self) -> bool:
        """
        Fast direct-fault detection.

        SA0:
            node must be 1.

        SA1:
            node must be 0.

        No propagation check is performed.

        This is the intentional simplified abstraction for
        the rapid GNN + PPO experiment.
        """

        node_id = self.state.fault_node_id

        fault_type = self.state.fault_type

        # Invalid node
        if (
            node_id < 0
            or node_id >= self.num_nodes
        ):
            return False

        node_val = self.state.node_values[
            node_id
        ]

        # Still unknown
        if node_val == -1:
            return False

        # ---------------------------------------------------------
        # SA0
        #
        # Stuck-at-0 fault is excited when actual value is 1.
        # ---------------------------------------------------------
        if fault_type == 0:

            return node_val == 1

        # ---------------------------------------------------------
        # SA1
        #
        # Stuck-at-1 fault is excited when actual value is 0.
        # ---------------------------------------------------------
        if fault_type == 1:

            return node_val == 0

        return False

    # =============================================================
    # NODE SELECTION
    # =============================================================

    def _select_node_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> int:
        """
        Select a node based on GNN embeddings.

        Kept for compatibility with the existing project.

        NOTE:
            SET_0/SET_1 no longer use this method in fast mode.
            They directly target the fault node.
        """

        if embeddings.dim() > 1:

            norms = embeddings.norm(
                dim=-1
            )

        else:

            norms = embeddings

        mask = (
            self.state.node_values == -1
        )

        if not mask.any():

            return 0

        masked_norms = norms.clone()

        masked_norms[~torch.tensor(
            mask,
            device=masked_norms.device,
        )] = -float("inf")

        return int(
            masked_norms.argmax().item()
        )

    def _select_unassigned_node(self) -> int:
        """Select a random unassigned node."""

        unassigned = np.where(
            self.state.node_values == -1
        )[0]

        if len(unassigned) == 0:

            return 0

        return int(
            np.random.choice(unassigned)
        )

    # =============================================================
    # PRIMARY INPUT SELECTION
    # =============================================================

    def _select_unassigned_pi(
        self,
    ) -> Optional[int]:
        """Select an unassigned primary input."""

        unassigned = [
            pi
            for pi in self.primary_inputs
            if pi not in self.state.pi_assignments
        ]

        if not unassigned:

            return None

        return int(
            np.random.choice(unassigned)
        )

    # =============================================================
    # PATH CHECK
    # =============================================================

    def _path_to_po_exists(
        self,
        start_node: int,
    ) -> bool:
        """
        Original path-to-output check.

        Kept for compatibility/reference.

        NOTE:
            The fast direct-fault detection does not use this
            function.
        """

        visited = set()

        queue = [start_node]

        while queue:

            node = queue.pop(0)

            if node in visited:
                continue

            visited.add(node)

            # Check PO
            if node in self.primary_outputs:

                return True

            # Follow forward connections
            for neighbor in range(
                self.num_nodes
            ):

                if (
                    self.adj_matrix[
                        node,
                        neighbor
                    ] > 0
                ):

                    if (
                        self.state.node_values[
                            neighbor
                        ] != -1
                    ):

                        queue.append(neighbor)

        return False

    # =============================================================
    # CONFLICT CHECK
    # =============================================================

    def _check_conflict(self) -> bool:
        """
        Simplified conflict check.

        The current fast environment does not model gate-level
        consistency conflicts.
        """

        return False

    # =============================================================
    # FRONTIER UPDATE
    # =============================================================

    def _update_frontiers(self):
        """
        Update D-frontier and J-frontier.

        This is retained from the original environment for
        compatibility with the GNN state representation.
        """

        self.state.d_frontier = []
        self.state.j_frontier = []

        for i in range(self.num_nodes):

            # Already assigned
            if (
                self.state.node_values[i]
                != -1
            ):
                continue

            # Check whether any predecessor has a value
            has_input_value = False

            for j in range(self.num_nodes):

                if (
                    self.adj_matrix[j, i] > 0
                    and
                    self.state.node_values[j]
                    != -1
                ):

                    has_input_value = True

                    break

            if has_input_value:

                self.state.j_frontier.append(i)

    # =============================================================
    # STATISTICS
    # =============================================================

    def get_statistics(self) -> dict:
        """Get environment statistics."""

        return {
            "total_faults":
                self.total_faults,

            "faults_detected":
                self.faults_detected,

            "fault_coverage":
                (
                    self.faults_detected
                    / self.total_faults
                    if self.total_faults > 0
                    else 0.0
                ),

            "total_backtracks":
                self.total_backtracks,
        }