"""
scoap.py - SCOAP Controllability/Observability Analysis

Implements the Sandia Controllability/Observability Analysis Program (SCOAP)
for computing testability measures on gate-level netlists.

SCOAP measures:
    - CC0: Controllability-0 (effort to set output to 0)
    - CC1: Controllability-1 (effort to set output to 1)
    - CO:  Observability (effort to observe the output at a primary output)

Reference:
    L. H. Goldstein and E. L. Thigpen, "SCOAP: Sandia Controllability/
    Observability Analysis Program," DAC 1980.
"""

import logging
from collections import defaultdict
from typing import Optional

from gnnatpg.graph_construction.netlist_parser import (
    Netlist, Gate, GateType, Wire
)

logger = logging.getLogger(__name__)


class SCOAPAnalyzer:
    """
    Computes SCOAP controllability and observability measures.
    
    CC0/CC1 are computed forward from primary inputs.
    CO is computed backward from primary outputs.
    """

    def __init__(self):
        self.cc0: dict[int, float] = {}  # gate_id -> CC0
        self.cc1: dict[int, float] = {}  # gate_id -> CC1
        self.co: dict[int, float] = {}   # gate_id -> CO

    def analyze(self, netlist: Netlist) -> dict[int, dict[str, float]]:
        """
        Run full SCOAP analysis on a netlist.
        
        Returns:
            Dict mapping gate_id -> {cc0, cc1, co}
        """
        self._compute_controllability(netlist)
        self._compute_observability(netlist)

        results = {}
        for gate in netlist.gates:
            results[gate.gate_id] = {
                "cc0": self.cc0.get(gate.gate_id, 0.0),
                "cc1": self.cc1.get(gate.gate_id, 0.0),
                "co": self.co.get(gate.gate_id, 0.0),
            }

        logger.info(
            f"SCOAP analysis complete: {len(results)} gates analyzed"
        )
        return results

    def _compute_controllability(self, netlist: Netlist):
        """Forward pass: compute CC0 and CC1 from primary inputs to outputs."""
        # Build gate_id to gate mapping
        gate_map = {g.gate_id: g for g in netlist.gates}

        # Initialize primary inputs
        for gate in netlist.gates:
            if gate.gate_type in (GateType.DFF, GateType.DFFE):
                # Flip-flops: CC0 = CC1 = 1 (register state)
                self.cc0[gate.gate_id] = 1.0
                self.cc1[gate.gate_id] = 1.0
            elif any(net in netlist.primary_inputs for net in gate.input_nets):
                # Gates driven by primary inputs
                self.cc0[gate.gate_id] = 1.0
                self.cc1[gate.gate_id] = 1.0
            else:
                self.cc0[gate.gate_id] = float("inf")
                self.cc1[gate.gate_id] = float("inf")

        # Set primary input controllability directly on input wires
        # (PIs have CC0=CC1=0 conceptually, but we count gate cost)

        # Iterative fixed-point computation
        max_iterations = 100
        for iteration in range(max_iterations):
            updated = False
            for gate in netlist.gates:
                if gate.gate_type in (GateType.DFF, GateType.DFFE):
                    continue

                # Get input controllabilities
                input_cc0 = []
                input_cc1 = []
                all_finite = True

                for net in gate.input_nets:
                    wire = netlist.wires.get(net)
                    if wire and wire.driven_by is not None:
                        driver_cc0 = self.cc0.get(wire.driven_by, float("inf"))
                        driver_cc1 = self.cc1.get(wire.driven_by, float("inf"))
                    elif net in netlist.primary_inputs:
                        # Primary input: CC0=0, CC1=0
                        driver_cc0 = 0.0
                        driver_cc1 = 0.0
                    else:
                        all_finite = False
                        break

                    input_cc0.append(driver_cc0)
                    input_cc1.append(driver_cc1)

                if not all_finite:
                    continue

                # Compute output controllability based on gate type
                new_cc0, new_cc1 = self._gate_controllability(
                    gate.gate_type, input_cc0, input_cc1
                )

                # Add 1 for the gate itself
                new_cc0 += 1.0
                new_cc1 += 1.0

                # Update if changed
                if (new_cc0 < self.cc0.get(gate.gate_id, float("inf")) or
                    new_cc1 < self.cc1.get(gate.gate_id, float("inf"))):
                    self.cc0[gate.gate_id] = new_cc0
                    self.cc1[gate.gate_id] = new_cc1
                    updated = True

            if not updated:
                break

        # Handle unreachable gates
        for gate in netlist.gates:
            if self.cc0.get(gate.gate_id, float("inf")) == float("inf"):
                self.cc0[gate.gate_id] = 100.0  # Large default
            if self.cc1.get(gate.gate_id, float("inf")) == float("inf"):
                self.cc1[gate.gate_id] = 100.0

    def _gate_controllability(
        self,
        gate_type: GateType,
        input_cc0: list[float],
        input_cc1: list[float],
    ) -> tuple[float, float]:
        """Compute output controllability based on gate type."""
        if gate_type == GateType.NOT:
            return input_cc1[0], input_cc0[0]

        if gate_type == GateType.BUF:
            return input_cc0[0], input_cc1[0]

        if gate_type in (GateType.AND2, GateType.AND3, GateType.AND4):
            # AND: output=0 if any input=0 (min CC0), output=1 if all inputs=1 (sum CC1)
            return min(input_cc0), sum(input_cc1)

        if gate_type in (GateType.OR2, GateType.OR3, GateType.OR4):
            # OR: output=1 if any input=1 (min CC1), output=0 if all inputs=0 (sum CC0)
            return sum(input_cc0), min(input_cc1)

        if gate_type in (GateType.NAND2, GateType.NAND3, GateType.NAND4):
            return sum(input_cc1), min(input_cc0)

        if gate_type in (GateType.NOR2, GateType.NOR3, GateType.NOR4):
            return min(input_cc0), sum(input_cc1)

        if gate_type == GateType.XOR2:
            return (
                min(input_cc0[0] + input_cc1[1], input_cc1[0] + input_cc0[1]),
                min(input_cc0[0] + input_cc0[1], input_cc1[0] + input_cc1[1]),
            )

        if gate_type == GateType.XNOR2:
            return (
                min(input_cc0[0] + input_cc0[1], input_cc1[0] + input_cc1[1]),
                min(input_cc0[0] + input_cc1[1], input_cc1[0] + input_cc0[1]),
            )

        if gate_type == GateType.MUX2:
            # MUX(S, A, B): output = S ? A : B
            if len(input_cc0) >= 3:
                return (
                    min(
                        input_cc0[0] + input_cc0[1],
                        input_cc1[0] + input_cc0[2],
                    ),
                    min(
                        input_cc0[0] + input_cc1[1],
                        input_cc1[0] + input_cc1[2],
                    ),
                )

        # Default: sum for CC0, sum for CC1
        return sum(input_cc0), sum(input_cc1)

    def _compute_observability(self, netlist: Netlist):
        """Backward pass: compute CO from primary outputs to inputs."""
        # Initialize primary outputs
        for gate in netlist.gates:
            if gate.output_net in netlist.primary_outputs:
                self.co[gate.gate_id] = 0.0
            else:
                self.co[gate.gate_id] = float("inf")

        # Build reverse mapping: gate_id -> list of (gate_id, input_net) that drive it
        # For each gate, find which gates observe its output
        gate_map = {g.gate_id: g for g in netlist.gates}
        observer_map = defaultdict(list)  # driver_id -> [(observer_id, input_net)]

        for gate in netlist.gates:
            for input_net in gate.input_nets:
                wire = netlist.wires.get(input_net)
                if wire and wire.driven_by is not None:
                    observer_map[wire.driven_by].append(
                        (gate.gate_id, input_net)
                    )

        # Iterative backward pass
        max_iterations = 100
        for iteration in range(max_iterations):
            updated = False
            for driver_id, observers in observer_map.items():
                driver_gate = gate_map.get(driver_id)
                if not driver_gate:
                    continue

                for observer_id, input_net in observers:
                    observer_gate = gate_map.get(observer_id)
                    if not observer_gate:
                        continue

                    observer_co = self.co.get(observer_id, float("inf"))
                    if observer_co == float("inf"):
                        continue

                    # Compute observability contribution
                    co_value = observer_co + 1.0  # Add cost of observing through this gate

                    if co_value < self.co.get(driver_id, float("inf")):
                        self.co[driver_id] = co_value
                        updated = True

            if not updated:
                break

        # Handle unreachable gates
        for gate in netlist.gates:
            if self.co.get(gate.gate_id, float("inf")) == float("inf"):
                self.co[gate.gate_id] = 100.0

    def get_normalized_features(
        self, netlist: Netlist
    ) -> dict[int, dict[str, float]]:
        """
        Get SCOAP features normalized to [0, 1] range.
        """
        raw = self.analyze(netlist)

        if not raw:
            return raw

        # Find max values for normalization
        max_cc0 = max(r["cc0"] for r in raw.values()) or 1.0
        max_cc1 = max(r["cc1"] for r in raw.values()) or 1.0
        max_co = max(r["co"] for r in raw.values()) or 1.0

        normalized = {}
        for gate_id, values in raw.items():
            normalized[gate_id] = {
                "cc0": values["cc0"] / max_cc0,
                "cc1": values["cc1"] / max_cc1,
                "co": values["co"] / max_co,
            }

        return normalized
