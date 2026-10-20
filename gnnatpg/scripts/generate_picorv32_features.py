"""
Generate PicoRV32 graph/features from the existing gate-level netlist.

This skips synthesis and Ibex.
It uses the already available:

    data/netlists/picorv32.v

and generates:

    data/features/picorv32_features.pt
"""

from pathlib import Path

from gnnatpg.features.extract import extract_all_features


def main():

    # Existing PicoRV32 gate-level netlist
    netlist_path = Path("data/netlists/picorv32.v")

    # Where Phase 2 expects the feature file
    features_dir = Path("data/features")

    if not netlist_path.exists():
        raise FileNotFoundError(
            f"PicoRV32 netlist not found: {netlist_path}"
        )

    features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("PicoRV32 Feature Extraction")
    print("=" * 60)

    print(f"Netlist : {netlist_path}")
    print(f"Output  : {features_dir}")
    print()

    # extract_all_features expects:
    # {
    #     "core_name": netlist_path
    # }
    netlist_paths = {
        "picorv32": netlist_path
    }

    extracted_features = extract_all_features(
        netlist_paths,
        features_dir,
    )

    print()
    print("=" * 60)
    print("Feature extraction complete")
    print("=" * 60)

    for core_name, graph in extracted_features.items():

        print(
            f"{core_name}: "
            f"{graph.num_nodes} nodes, "
            f"{graph.x.shape[1]} features"
        )

        output_file = (
            features_dir
            / f"{core_name}_features.pt"
        )

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()