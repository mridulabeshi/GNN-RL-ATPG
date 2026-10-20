"""
synthesize.py - RISC-V Core Synthesis via Yosys

Converts open-source RISC-V core RTL into technology-mapped
gate-level netlists using Yosys open-source synthesis tool.

Steps:
    1. Clone/download RISC-V core RTL
    2. Generate Yosys synthesis script
    3. Run Yosys to produce gate-level Verilog netlist
    4. Validate output netlist
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional
import yaml
import shutil

logger = logging.getLogger(__name__)


class RISCVSynthesizer:
    """Synthesizes RISC-V RTL to gate-level netlists using Yosys."""

    # ---------------------------------------------------------
    # Standard cell gates expected in the output
    # ---------------------------------------------------------
    VALID_GATE_TYPES = {
        "AND2", "AND3", "AND4",
        "OR2", "OR3", "OR4",
        "NAND2", "NAND3", "NAND4",
        "NOR2", "NOR3", "NOR4",
        "XOR2", "XOR3", "XOR4",
        "XNOR2", "XNOR3",
        "NOT", "BUF",
        "MUX2", "MUX4",
        "DFF", "DFFE",
        "_DFFE_PP0_", "_DFFE_PP1_",
    }

    # ---------------------------------------------------------
    # Yosys synthesis commands for each target
    # ---------------------------------------------------------
    SYNTH_COMMANDS = {
        "generic": """
            proc
            flatten
            opt
            opt_expr
            opt_clean
            fsm
            techmap
            opt
            opt_expr
            opt_clean
            flatten
            clean
            opt
        """,

        "sky130": """
            read_liberty -lib {lib_file}
            proc
            flatten
            opt
            synth -flatten -top {top_module}
            abc -liberty {lib_file}
            opt_clean
            flatten
            clean
        """,
    }

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------
    def __init__(
        self,
        config_path: str = "config.yaml"
    ):
        """Initialize synthesizer from configuration."""

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.synth_config = self.config["synthesis"]

        self.netlist_dir = Path("data/netlists")
        self.netlist_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Clone RISC-V core
    # ---------------------------------------------------------
    def clone_core(
        self,
        core_name: str,
        dest_dir: str = "data/rtl"
    ) -> Path:
        """Clone a RISC-V core repository."""

        core_info = self.config["riscv_cores"][core_name]

        repo_url = core_info["repo"]

        dest = Path(dest_dir) / core_name

        if dest.exists():
            logger.info(
                f"Core {core_name} already cloned at {dest}"
            )
            return dest

        logger.info(
            f"Cloning {core_name} from {repo_url}..."
        )

        dest.mkdir(
            parents=True,
            exist_ok=True
        )

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                repo_url,
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(
            f"Cloned to {dest}"
        )

        return dest

    # ---------------------------------------------------------
    # Find RTL files
    # ---------------------------------------------------------
    def find_rtl_files(
        self,
        core_name: str,
        rtl_dir: Path
    ) -> list[Path]:
        """Find synthesizable RTL files for the selected RISC-V core."""

        # ---------------------------------------------------------
        # PicoRV32
        # ---------------------------------------------------------
        if core_name == "picorv32":

            # The PicoRV32 repository contains many additional files:
            # FPGA examples, SoC wrappers, formal verification files,
            # simulation files, etc.
            #
            # For ATPG, synthesize only the actual CPU core.
            rtl_files = [
                rtl_dir / "picorv32.v"
            ]

        # ---------------------------------------------------------
        # Ibex
        # ---------------------------------------------------------
        elif core_name == "ibex":

            # Ibex is primarily SystemVerilog.
            rtl_files = [
                f
                for f in rtl_dir.rglob("*.sv")
                if "dv" not in str(f).lower()
                and "test" not in str(f).lower()
                and "tb" not in f.name.lower()
            ]

            # Header files are intentionally not passed independently
            # to Yosys. They are included by the SystemVerilog sources.
            #
            # rtl_files.extend(
            #     rtl_dir.rglob("*.svh")
            # )

        # ---------------------------------------------------------
        # Generic RISC-V core
        # ---------------------------------------------------------
        else:

            extensions = {
                ".v",
                ".sv",
            }

            rtl_files = []

            for ext in extensions:

                rtl_files.extend(
                    rtl_dir.rglob(
                        f"*{ext}"
                    )
                )

            # Remove obvious non-synthesizable files.
            rtl_files = [
                f
                for f in rtl_files
                if "tb" not in f.name.lower()
                and "testbench" not in f.name.lower()
                and "sim" not in str(f).lower()
                and "doc" not in str(f).lower()
                and "formal" not in str(f).lower()
            ]

        # ---------------------------------------------------------
        # Keep only files that actually exist
        # ---------------------------------------------------------
        rtl_files = [
            f
            for f in rtl_files
            if f.exists()
        ]

        logger.info(
            f"Found {len(rtl_files)} RTL files "
            f"for {core_name}"
        )

        for f in rtl_files:
            logger.debug(
                f"  RTL: {f}"
            )

        return rtl_files

    # ---------------------------------------------------------
    # Generate Yosys synthesis script
    # ---------------------------------------------------------
    def generate_yosys_script(
        self,
        core_name: str,
        rtl_files: list[Path],
        top_module: str
    ) -> str:
        """Generate a Yosys synthesis script."""

        synth_target = self.synth_config["target"]

        commands = self.SYNTH_COMMANDS.get(
            synth_target,
            self.SYNTH_COMMANDS["generic"]
        )

        script_lines = [
            f"# Auto-generated synthesis script for {core_name}",
            f"# Target: {synth_target}",
        ]

        # ---------------------------------------------------------
        # Read RTL files
        # ---------------------------------------------------------
        for rtl_file in rtl_files:

            # Use SystemVerilog mode for .sv files.
            # Do not read .svh files independently.
            if rtl_file.suffix == ".sv":

                script_lines.append(
                    f"read_verilog -sv {rtl_file}"
                )

            elif rtl_file.suffix == ".v":

                script_lines.append(
                    f"read_verilog {rtl_file}"
                )

        # ---------------------------------------------------------
        # Set hierarchy
        # ---------------------------------------------------------
        script_lines.append(
            f"hierarchy -top {top_module}"
        )

        # ---------------------------------------------------------
        # Flatten hierarchy
        # ---------------------------------------------------------
        script_lines.append(
            "flatten"
        )

        # ---------------------------------------------------------
        # Original synthesis commands
        # ---------------------------------------------------------
        for cmd in commands.strip().split("\n"):

            cmd = cmd.strip()

            if cmd and not cmd.startswith("#"):

                # For sky130, substitute configuration values.
                if synth_target == "sky130":

                    lib_file = self.synth_config.get(
                        "lib_file",
                        ""
                    )

                    cmd = cmd.format(
                        lib_file=lib_file,
                        top_module=top_module
                    )

                script_lines.append(cmd)

        # ---------------------------------------------------------
        # Final technology mapping
        # ---------------------------------------------------------
        #
        # The generic synthesis sequence above leaves Yosys
        # internal logic cells. The ABC pass maps the combinational
        # logic to a simple gate set.
        #
        # -g simple:
        #     AND, OR, XOR, MUX
        #
        # -dff:
        #     Preserve/map flip-flop cells.
        #
        script_lines.append(
            "memory"
        )

        script_lines.append(
            "opt"
        )

        script_lines.append(
            "abc -g simple -dff"
        )

        script_lines.append(
            "opt"
        )

        script_lines.append(
            "clean"
        )

        # ---------------------------------------------------------
        # Output gate-level netlist
        # ---------------------------------------------------------
        output_path = (
            self.netlist_dir /
            f"{core_name}.v"
        )

        script_lines.append(
            f"write_verilog -noattr -noexpr {output_path}"
        )

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------
        script_lines.append(
            "stat"
        )

        return "\n".join(
            script_lines
        )

    # ---------------------------------------------------------
    # Run synthesis
    # ---------------------------------------------------------
    def run_synthesis(
        self,
        core_name: str,
        top_module: Optional[str] = None,
        rtl_dir: Optional[str] = None,
    ) -> Path:
        """
        Run full synthesis pipeline for a RISC-V core.

        Args:
            core_name:
                Name of the RISC-V core
                (ibex, picorv32, etc.)

            top_module:
                Top-level module name.
                Auto-detected from config if None.

            rtl_dir:
                Path to RTL files.
                Cloned if None.

        Returns:
            Path to generated gate-level netlist.
        """

        # ---------------------------------------------------------
        # Core configuration
        # ---------------------------------------------------------
        core_config = (
            self.config["riscv_cores"]
            .get(core_name, {})
        )

        if top_module is None:

            top_module = core_config.get(
                "top_module",
                core_name
            )

        # ---------------------------------------------------------
        # Locate RTL
        # ---------------------------------------------------------
        if rtl_dir is None:

            rtl_path = self.clone_core(
                core_name
            )

            rtl_path = (
                rtl_path /
                core_config.get(
                    "rtl_dir",
                    "."
                )
            )

        else:

            rtl_path = Path(
                rtl_dir
            )

        # ---------------------------------------------------------
        # Find RTL files
        # ---------------------------------------------------------
        rtl_files = self.find_rtl_files(
            core_name,
            rtl_path
        )

        if not rtl_files:

            raise FileNotFoundError(
                f"No RTL files found for "
                f"{core_name} in {rtl_path}"
            )

        # ---------------------------------------------------------
        # Generate Yosys script
        # ---------------------------------------------------------
        script_content = (
            self.generate_yosys_script(
                core_name,
                rtl_files,
                top_module
            )
        )

        script_path = Path(
            "data/netlists/"
            f"{core_name}_synth.ys"
        )

        script_path.write_text(
            script_content
        )

        logger.info(
            f"Generated Yosys script: "
            f"{script_path}"
        )

        # ---------------------------------------------------------
        # Run Yosys
        # ---------------------------------------------------------
        logger.info(
            f"Running Yosys synthesis "
            f"for {core_name}..."
        )

        output_netlist = (
            self.netlist_dir /
            f"{core_name}.v"
        )

        try:

            result = subprocess.run(
                [
                    "yosys",
                    "-s",
                    str(script_path)
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            # -----------------------------------------------------
            # Check Yosys result
            # -----------------------------------------------------
            if result.returncode != 0:

                logger.error(
                    f"Yosys failed:\n"
                    f"{result.stderr}"
                )

                raise RuntimeError(
                    "Yosys synthesis failed: "
                    f"{result.stderr}"
                )

            # -----------------------------------------------------
            # Parse statistics
            # -----------------------------------------------------
            self._parse_yosys_stats(
                result.stdout,
                core_name
            )

        except FileNotFoundError:

            logger.error(
                "Yosys not found. "
                "Install with: sudo apt install yosys"
            )

            raise

        # ---------------------------------------------------------
        # Validate output exists
        # ---------------------------------------------------------
        if not output_netlist.exists():

            raise FileNotFoundError(
                f"Netlist not generated at "
                f"{output_netlist}"
            )

        logger.info(
            f"Synthesis complete: "
            f"{output_netlist}"
        )

        return output_netlist

    # ---------------------------------------------------------
    # Parse Yosys statistics
    # ---------------------------------------------------------
    def _parse_yosys_stats(
        self,
        yosys_output: str,
        core_name: str
    ):
        """Parse and log Yosys synthesis statistics."""

        stats_lines = []

        capture = False

        for line in yosys_output.split("\n"):

            if (
                "Number of wires:" in line
                or
                "Number of wire bits:" in line
            ):
                capture = True

            if (
                "Number of cells:" in line
                or
                "Estimated" in line
            ):
                capture = True

            if capture:
                stats_lines.append(
                    line
                )

            if (
                line.strip() == ""
                and capture
            ):
                capture = False

        if stats_lines:

            logger.info(
                f"=== Synthesis Stats for "
                f"{core_name} ==="
            )

            for line in stats_lines:

                logger.info(
                    f"  {line}"
                )

    # ---------------------------------------------------------
    # Validate generated netlist
    # ---------------------------------------------------------
    def validate_netlist(
        self,
        netlist_path: Path
    ) -> dict:
        """Validate the generated netlist."""

        content = netlist_path.read_text()

        stats = {
            "total_lines": len(
                content.split("\n")
            ),
            "has_module": "module" in content,
            "has_endmodule": "endmodule" in content,
            "gate_types_found": set(),
        }

        # ---------------------------------------------------------
        # Check for standard cells
        # ---------------------------------------------------------
        for gate in self.VALID_GATE_TYPES:

            if gate in content:

                stats["gate_types_found"].add(
                    gate
                )

        logger.info(
            f"Netlist validation: "
            f"{stats['total_lines']} lines, "
            f"{len(stats['gate_types_found'])} "
            f"gate types found"
        )

        return stats


# ----------------------------------------------------------------------
# Synthesize all configured RISC-V cores
# ----------------------------------------------------------------------
def synthesize_all(
    config_path: str = "config.yaml"
) -> dict:
    """Synthesize all configured RISC-V cores."""

    synthesizer = RISCVSynthesizer(
        config_path
    )

    results = {}

    for core_name in (
        synthesizer.config["riscv_cores"]
    ):

        try:

            netlist_path = (
                synthesizer.run_synthesis(
                    core_name
                )
            )

            stats = (
                synthesizer.validate_netlist(
                    netlist_path
                )
            )

            results[core_name] = {
                "netlist": str(
                    netlist_path
                ),
                "stats": stats,
                "success": True,
            }

        except Exception as e:

            logger.error(
                f"Failed to synthesize "
                f"{core_name}: {e}"
            )

            results[core_name] = {
                "success": False,
                "error": str(e),
            }

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    results = synthesize_all()

    for core, result in results.items():

        status = (
            "✓"
            if result["success"]
            else "✗"
        )

        print(
            f"  {status} "
            f"{core}: {result}"
        )