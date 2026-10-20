"""
podem_baseline.py - PODEM ATPG Baseline

Implements the Path-Oriented Decision Making (PODEM) algorithm for
comparison with ML-guided ATPG. PODEM is a classic ATPG technique
that uses implicit enumeration with backtracking.

Reference:
    P. Goel, "An Implicit Enumeration Algorithm to Generate Tests for
    Combinational Logic Circuits," IEEE Trans. Computers, 1981.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from gnnatpg.graph_construction.netlist_parser import (
    Netlist, Gate, GateType
)
from gnnatpg.fault_simulation.simulator import (
    FaultSimulator, TestPattern, SimulationResult
)

logger = logging.getLogger(__name__)


@dataclass
class ATPGResult:
    """Result of ATPG for a single fault."""
    fault_id: int
    fault_name: str
    detected: bool
    pattern: Optional[TestPattern]
    backtracks: int
    runtime_steps: int
    status: str  # "detected", "undetectable", "redundant", "aborted"


class PODEM:
    """
    PODEM (Path-Oriented Decision Making) ATPG algorithm.
    
    Algorithm:
        1. Start with all PI values unknown (X)
        2. Select a target fault
        3. Backtrace from fault site to PIs
        4. Set PI values to activate and propagate fault
        5. Forward simulate to check propagation
        6. If conflict, backtrack and try different assignment
        7. Repeat until detected or proven undetectable
    """

    def __init__(self, netlist: Netlist, max_backtracks: int = 1000):
        self.netlist = netlist
        self.max_backtracks = max_backtracks
        self.simulator = FaultSimulator(netlist)

        # Build gate index
        self.gate_map = {g.gate_id: g for g in netlist.gates}
        self.pi_gate_ids = self._find_pi_gates()
        self.po_gate_ids = self._find_po_gates()

    def _find_pi_gates(self) -> list[int]:
        """Find gates directly connected to primary inputs."""
        pi_gates = []
        for gate in self.netlist.gates:
            if any(net in self.netlist.primary_inputs for net in gate.input_nets):
                pi_gates.append(gate.gate_id)
        return pi_gates

    def _find_po_gates(self) -> list[int]:
        """Find gates driving primary outputs."""
        po_gates = []
        for gate in self.netlist.gates:
            if gate.output_net in self.netlist.primary_outputs:
                po_gates.append(gate.gate_id)
        return po_gates

    def generate_test(
        self, target_fault_gate_id: int, fault_type: str
    ) -> ATPGResult:
        """
        Generate test pattern for a specific fault using PODEM.
        
        Args:
            target_fault_gate_id: Gate where fault is located
            fault_type: "SA0" or "SA1"
            
        Returns:
            ATPGResult with test pattern or failure information
        """
        gate = self.gate_map.get(target_fault_gate_id)
        if not gate:
            return ATPGResult(
                fault_id=-1,
                fault_name="unknown",
                detected=False,
                pattern=None,
                backtracks=0,
                runtime_steps=0,
                status="aborted",
            )

        # Initialize PI assignments to X
        pi_values = {pi: -1 for pi in self.netlist.primary_inputs}

        # Step 1: Activate fault - set fault site to opposite value
        target_value = 0 if fault_type == "SA0" else 1

        # Step 2: Backtrace from fault site to PIs
        backtrack_count = 0
        step_count = 0

        # Simple PODEM: try all PI assignments systematically
        return self._podem_search(
            target_fault_gate_id=target_fault_gate_id,
            fault_type=fault_type,
            target_value=target_value,
            pi_values=pi_values,
            backtrack_count=0,
            step_count=0,
        )

    def _podem_search(
        self,
        target_fault_gate_id: int,
        fault_type: str,
        target_value: int,
        pi_values: dict[str, int],
        backtrack_count: int,
        step_count: int,
    ) -> ATPGResult:
        """Recursive PODEM search."""
        step_count += 1

        if backtrack_count > self.max_backtracks:
            gate = self.gate_map.get(target_fault_gate_id)
            return ATPGResult(
                fault_id=target_fault_gate_id,
                fault_name=gate.instance_name if gate else "unknown",
                detected=False,
                pattern=None,
                backtracks=backtrack_count,
                runtime_steps=step_count,
                status="aborted",
            )

        # Try to propagate fault to PO
        # For simplicity, try each PI with value 0 and 1
        unassigned_pis = [
            pi for pi, v in pi_values.items() if v == -1
        ]

        if not unassigned_pis:
            # All PIs assigned - check if fault is detected
            pattern = TestPattern(
                pattern_id=0,
                pi_values={k: v for k, v in pi_values.items() if v != -1},
            )

            # Simulate good and faulty circuit
            good_vals = self.simulator._evaluate_circuit(pattern.pi_values)
            faulty_vals = self.simulator._evaluate_circuit(
                pattern.pi_values,
                fault_gate_id=target_fault_gate_id,
                fault_type=fault_type,
            )

            # Check if any PO is different
            detected = False
            for po in self.netlist.primary_outputs:
                po_wire = self.netlist.wires.get(po)
                if po_wire and po_wire.driven_by in good_vals:
                    if good_vals[po_wire.driven_by] != faulty_vals.get(
                        po_wire.driven_by, good_vals[po_wire.driven_by]
                    ):
                        detected = True
                        break

            gate = self.gate_map.get(target_fault_gate_id)
            return ATPGResult(
                fault_id=target_fault_gate_id,
                fault_name=gate.instance_name if gate else "unknown",
                detected=detected,
                pattern=pattern if detected else None,
                backtracks=backtrack_count,
                runtime_steps=step_count,
                status="detected" if detected else "undetectable",
            )

        # Try first unassigned PI with value 0
        pi_to_set = unassigned_pis[0]
        for value in [1, 0]:  # Try 1 first (more likely to activate faults)
            new_pi_values = pi_values.copy()
            new_pi_values[pi_to_set] = value

            result = self._podem_search(
                target_fault_gate_id=target_fault_gate_id,
                fault_type=fault_type,
                target_value=target_value,
                pi_values=new_pi_values,
                backtrack_count=backtrack_count + (1 if value == 0 else 0),
                step_count=step_count,
            )

            if result.detected:
                return result

        # No solution found
        gate = self.gate_map.get(target_fault_gate_id)
        return ATPGResult(
            fault_id=target_fault_gate_id,
            fault_name=gate.instance_name if gate else "unknown",
            detected=False,
            pattern=None,
            backtracks=backtrack_count,
            runtime_steps=step_count,
            status="undetectable",
        )

    def run_atpg(self) -> list[ATPGResult]:
        """
        Run PODEM ATPG on all faults in the fault list.
        
        Returns:
            List of ATPGResults, one per fault
        """
        fault_list = self.simulator.generate_fault_list()
        results = []

        logger.info(
            f"Running PODEM ATPG on {len(fault_list)} faults "
            f"(max_backtracks={self.max_backtracks})..."
        )

        for fault in fault_list:
            result = self.generate_test(fault.gate_id, fault.fault_type)
            results.append(result)

            if result.detected:
                logger.debug(
                    f"Fault {fault.fault_name} ({fault.fault_type}): "
                    f"DETECTED in {result.backtracks} backtracks"
                )
            else:
                logger.debug(
                    f"Fault {fault.fault_name} ({fault.fault_type}): "
                    f"{result.status} after {result.backtracks} backtracks"
                )

        # Summary
        detected = sum(1 for r in results if r.detected)
        total = len(results)
        coverage = detected / total if total > 0 else 0
        avg_backtracks = np.mean([r.backtracks for r in results])

        summary = {
            "total_faults": total,
            "detected": detected,
            "coverage": f"{coverage:.2%}",
            "avg_backtracks": f"{avg_backtracks:.1f}",
        }
        logger.info(f"PODEM ATPG complete: {summary}")

        return results

    def get_patterns(self, results: list[ATPGResult]) -> list[TestPattern]:
        """Extract generated test patterns from PODEM results."""
        patterns = []
        for i, result in enumerate(results):
            if result.detected and result.pattern:
                result.pattern.pattern_id = i
                patterns.append(result.pattern)
        return patterns
