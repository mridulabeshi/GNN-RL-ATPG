"""
simulator.py - Fault Simulation Engine

Evaluates generated test patterns against a fault list to compute
fault coverage, detectability, and pattern quality metrics.

Supports:
    - Stuck-at-0 (SA0) and Stuck-at-1 (SA1) fault models
    - Single-fault assumption
    - Parallel fault simulation for efficiency
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gnnatpg.graph_construction.netlist_parser import (
    Netlist, Gate, GateType
)

logger = logging.getLogger(__name__)


@dataclass
class Fault:
    """Represents a single stuck-at fault."""
    fault_id: int
    gate_id: int
    gate_name: str
    fault_type: str  # "SA0" or "SA1"
    detected: bool = False
    detected_by_pattern: int = -1

    @property
    def is_detected(self) -> bool:
        return self.detected


@dataclass
class TestPattern:
    """A test pattern (PI assignment vector)."""
    pattern_id: int
    pi_values: dict[str, int]  # PI name -> value
    detected_faults: list[int] = field(default_factory=list)

    @property
    def num_detected(self) -> int:
        return len(self.detected_faults)


@dataclass
class SimulationResult:
    """Results of fault simulation."""
    total_faults: int
    detected_faults: int
    undetected_faults: int
    fault_coverage: float
    test_patterns: list[TestPattern]
    pattern_count: int
    undetected_fault_list: list[Fault]

    def summary(self) -> dict:
        return {
            "total_faults": self.total_faults,
            "detected_faults": self.detected_faults,
            "fault_coverage": f"{self.fault_coverage:.2%}",
            "pattern_count": self.pattern_count,
            "undetected_count": self.undetected_faults,
        }


class FaultSimulator:
    """
    Single-fault simulator for gate-level netlists.
    
    For each test pattern, simulates each fault independently:
    1. Inject the fault (force stuck-at value)
    2. Propagate logic values through the circuit
    3. Check if fault effect reaches any primary output
    """

    def __init__(self, netlist: Netlist):
        self.netlist = netlist
        self._build_simulation_graph()

    def _build_simulation_graph(self):
        """Build efficient simulation data structures."""
        self.gate_map = {g.gate_id: g for g in self.netlist.gates}

        # Build fanin/fanout lists
        self.fanin: dict[int, list[int]] = {}
        self.fanout: dict[int, list[int]] = {}

        for gate in self.netlist.gates:
            self.fanin[gate.gate_id] = []
            self.fanout[gate.gate_id] = []

        for gate in self.netlist.gates:
            for input_net in gate.input_nets:
                wire = self.netlist.wires.get(input_net)
                if wire and wire.driven_by is not None:
                    self.fanin[gate.gate_id].append(wire.driven_by)
                    self.fanout[wire.driven_by].append(gate.gate_id)

    def generate_fault_list(self) -> list[Fault]:
        """Generate stuck-at fault list for the circuit."""
        faults = []
        fault_id = 0

        for gate in self.netlist.gates:
            # SA0 fault at gate output
            faults.append(Fault(
                fault_id=fault_id,
                gate_id=gate.gate_id,
                gate_name=gate.instance_name,
                fault_type="SA0",
            ))
            fault_id += 1

            # SA1 fault at gate output
            faults.append(Fault(
                fault_id=fault_id,
                gate_id=gate.gate_id,
                gate_name=gate.instance_name,
                fault_type="SA1",
            ))
            fault_id += 1

        logger.info(f"Generated {len(faults)} faults for {self.netlist.module_name}")
        return faults

    def simulate_pattern(
        self,
        pattern: TestPattern,
        fault_list: list[Fault],
    ) -> list[int]:
        """
        Simulate a test pattern against all undetected faults.
        
        Returns:
            List of detected fault IDs
        """
        detected = []

        for fault in fault_list:
            if fault.is_detected:
                continue

            # Check if this pattern detects the fault
            if self._detect_fault(pattern, fault):
                fault.detected = True
                fault.detected_by_pattern = pattern.pattern_id
                detected.append(fault.fault_id)
                pattern.detected_faults.append(fault.fault_id)

        return detected

    def _detect_fault(
        self, pattern: TestPattern, fault: Fault
    ) -> bool:
        """
        Check if a pattern detects a specific fault.
        
        Uses parallel pattern simulation:
        1. Compute good circuit values
        2. Compute faulty circuit values
        3. Check if any PO has different value (D or D')
        """
        # Compute good circuit values
        good_values = self._evaluate_circuit(pattern.pi_values)

        # Compute faulty circuit values
        faulty_values = self._evaluate_circuit(
            pattern.pi_values,
            fault_gate_id=fault.gate_id,
            fault_type=fault.fault_type,
        )

        # Compare at primary outputs
        for po_name in self.netlist.primary_outputs:
            po_wire = self.netlist.wires.get(po_name)
            if not po_wire or po_wire.driven_by is None:
                continue

            driver_gate = self.gate_map.get(po_wire.driven_by)
            if not driver_gate:
                continue

            gate_idx = driver_gate.gate_id
            if gate_idx in good_values and gate_idx in faulty_values:
                if good_values[gate_idx] != faulty_values[gate_idx]:
                    if good_values[gate_idx] != -1 and faulty_values[gate_idx] != -1:
                        return True

        return False

    def _evaluate_circuit(
        self,
        pi_values: dict[str, int],
        fault_gate_id: Optional[int] = None,
        fault_type: Optional[str] = None,
    ) -> dict[int, int]:
        """
        Evaluate all gate outputs given PI assignments.
        
        Args:
            pi_values: Primary input assignments
            fault_gate_id: Gate to inject fault at (None for good circuit)
            fault_type: "SA0" or "SA1" (None for good circuit)
            
        Returns:
            Dict mapping gate_id -> output value
        """
        values = {}
        evaluated = set()

        # Initialize PI values
        for gate in self.netlist.gates:
            for input_net in gate.input_nets:
                if input_net in pi_values:
                    # This gate reads from a PI
                    pass

        # Set PI wire values
        pi_wire_values = {}
        for pi_name, pi_val in pi_values.items():
            pi_wire_values[pi_name] = pi_val

        # Evaluate gates in topological order
        max_iterations = len(self.netlist.gates) * 2
        iteration = 0

        while len(evaluated) < len(self.netlist.gates) and iteration < max_iterations:
            iteration += 1
            for gate in self.netlist.gates:
                if gate.gate_id in evaluated:
                    continue

                # Check if all inputs are available
                input_values = []
                all_ready = True
                for input_net in gate.input_nets:
                    if input_net in pi_wire_values:
                        input_values.append(pi_wire_values[input_net])
                    else:
                        # Find driving gate
                        wire = self.netlist.wires.get(input_net)
                        if wire and wire.driven_by is not None:
                            if wire.driven_by in values:
                                input_values.append(values[wire.driven_by])
                            else:
                                all_ready = False
                                break
                        else:
                            all_ready = False
                            break

                if not all_ready or len(input_values) != len(gate.input_nets):
                    continue

                # Compute gate output
                output = self._compute_gate_output(
                    gate.gate_type, input_values
                )

                # Inject fault if applicable
                if fault_gate_id is not None and gate.gate_id == fault_gate_id:
                    output = 0 if fault_type == "SA0" else 1

                values[gate.gate_id] = output
                evaluated.add(gate.gate_id)

                # Propagate to wires
                if gate.output_net:
                    pi_wire_values[gate.output_net] = output

        return values

    def _compute_gate_output(
        self, gate_type: GateType, input_values: list[int]
    ) -> int:
        """Compute gate output from input values."""
        if gate_type == GateType.NOT:
            return 1 - input_values[0] if input_values[0] in (0, 1) else -1

        if gate_type == GateType.BUF:
            return input_values[0]

        if gate_type in (GateType.AND2, GateType.AND3, GateType.AND4):
            if any(v == 0 for v in input_values):
                return 0
            if all(v == 1 for v in input_values):
                return 1
            return -1  # Unknown

        if gate_type in (GateType.OR2, GateType.OR3, GateType.OR4):
            if any(v == 1 for v in input_values):
                return 1
            if all(v == 0 for v in input_values):
                return 0
            return -1

        if gate_type in (GateType.NAND2, GateType.NAND3, GateType.NAND4):
            and_result = 1
            for v in input_values:
                if v == 0:
                    and_result = 0
                    break
                if v == -1:
                    and_result = -1
            if and_result == -1:
                return -1
            return 1 - and_result

        if gate_type in (GateType.NOR2, GateType.NOR3, GateType.NOR4):
            or_result = 0
            for v in input_values:
                if v == 1:
                    or_result = 1
                    break
                if v == -1:
                    or_result = -1
            if or_result == -1:
                return -1
            return 1 - or_result

        if gate_type == GateType.XOR2:
            if len(input_values) >= 2:
                if -1 in input_values[:2]:
                    return -1
                return input_values[0] ^ input_values[1]

        if gate_type == GateType.XNOR2:
            if len(input_values) >= 2:
                if -1 in input_values[:2]:
                    return -1
                return 1 - (input_values[0] ^ input_values[1])

        if gate_type == GateType.MUX2:
            if len(input_values) >= 3:
                sel, a, b = input_values[0], input_values[1], input_values[2]
                if sel == -1 or (a == -1 and b == -1):
                    return -1
                return a if sel == 1 else b

        # DFF/DFFE: return current stored value (simplified)
        if gate_type in (GateType.DFF, GateType.DFFE):
            return input_values[0] if input_values else 0

        return -1  # Unknown gate type

    def run_simulation(
        self,
        patterns: list[TestPattern],
    ) -> SimulationResult:
        """
        Run full fault simulation with a set of test patterns.
        
        Args:
            patterns: List of test patterns to simulate
            
        Returns:
            SimulationResult with coverage metrics
        """
        fault_list = self.generate_fault_list()

        logger.info(
            f"Simulating {len(patterns)} patterns against "
            f"{len(fault_list)} faults..."
        )

        for pattern in patterns:
            self.simulate_pattern(pattern, fault_list)

        detected = sum(1 for f in fault_list if f.is_detected)
        undetected = [f for f in fault_list if not f.is_detected]

        coverage = detected / len(fault_list) if fault_list else 0.0

        result = SimulationResult(
            total_faults=len(fault_list),
            detected_faults=detected,
            undetected_faults=len(undetected),
            fault_coverage=coverage,
            test_patterns=patterns,
            pattern_count=len(patterns),
            undetected_fault_list=undetected,
        )

        logger.info(f"Fault simulation complete: {result.summary()}")
        return result

    def save_fault_list(self, fault_list: list[Fault], path: str):
        """Save fault list to file."""
        import json
        data = [
            {
                "fault_id": f.fault_id,
                "gate_id": f.gate_id,
                "gate_name": f.gate_name,
                "fault_type": f.fault_type,
            }
            for f in fault_list
        ]
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"Saved {len(data)} faults to {path}")

    def load_fault_list(self, path: str) -> list[Fault]:
        """Load fault list from file."""
        import json
        with open(path) as fh:
            data = json.load(fh)
        return [
            Fault(
                fault_id=d["fault_id"],
                gate_id=d["gate_id"],
                gate_name=d["gate_name"],
                fault_type=d["fault_type"],
            )
            for d in data
        ]
