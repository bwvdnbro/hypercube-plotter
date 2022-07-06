import argparse

from objects import Hypercube
from plot import create_sweep_plots, create_validation_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""DESCRIPTION:
          Given a set of data from a simulation hypercube in a swift-emulator-like format, 
          creates plots with parameter sweeps."""
    )

    parser.add_argument(
        "-c",
        "--param-config",
        required=True,
        type=str,
        nargs="?",
        action="store",
        help="Path to config with parameters",
    )

    parser.add_argument(
        "-p",
        "--params",
        required=True,
        type=str,
        nargs="?",
        action="store",
        help="Path to yml files containing simulation parameters",
    )

    parser.add_argument(
        "-i",
        "--plot-config",
        required=True,
        type=str,
        nargs="?",
        action="store",
        help="Path to config with plots",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        nargs="?",
        action="store",
        help="Path to output directory",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=str,
        nargs="?",
        action="store",
        help="Path to yml files with data",
    )

    parser.add_argument(
        "-s",
        "--single",
        type=str,
        default=None,
        action="store",
        help="Only emulate the plot with the given name.",
    )

    config = parser.parse_args()

    print(f"Path to config with params: {config.param_config}")
    print(f"Path to config with plots: {config.plot_config}")
    print(f"Path to simulation params: {config.params}")
    print(f"Path to simulation data: {config.data}")
    print(f"Path to output dir: {config.output}")

    # Create instance with metadata
    hypercube = Hypercube(
        path_to_param_config=config.param_config,
        path_to_plot_config=config.plot_config,
        path_to_params=config.params,
        path_to_data=config.data,
        path_to_output=config.output,
        plot_name=config.single,
    )

    # Create and train emulators
    hypercube.create_emulators()
    create_sweep_plots(hypercube)
    create_validation_plots(hypercube)
