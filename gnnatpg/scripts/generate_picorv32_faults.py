"""
Generate the PicoRV32 stuck-at fault list from the existing
gate-level netlist.

This skips synthesis and only performs fault generation.
"""

from pathlib import Path

from gnnatpg.graph_construction.netlist_parser import NetlistParser
from gnnatpg.fault_simulation.simulator import FaultSimulator


def main():

    netlist_path = Path("data/netlists/picorv32.v")
    output_path = Path("data/fault_list/picorv32_faults.json")

    if not netlist_path.exists():
        raise FileNotFoundError(
            f"Netlist not found: {netlist_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PicoRV32 Stuck-at Fault Generation")
    print("=" * 60)

    print(f"Netlist : {netlist_path}")
    print(f"Output  : {output_path}")
    print()

    # Parse existing gate-level netlist
    parser = NetlistParser()

    print("Parsing netlist...")
    netlist = parser.parse_file(netlist_path)

    print("Generating stuck-at faults...")

    simulator = FaultSimulator(netlist)
    fault_list = simulator.generate_fault_list()

    print()
    print(f"Generated faults: {len(fault_list)}")

    # Save using the project's existing format
    simulator.save_fault_list(
        fault_list,
        output_path,
    )

    print(f"Saved: {output_path}")

    print()
    print("First few faults:")

    for fault in fault_list[:10]:
        print(f"  {fault}")

    print()
    print("=" * 60)
    print("Fault generation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()