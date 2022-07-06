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
        "-l",
        "--plot-list",
        type=str,
        default=None,
        nargs="+",
        action="store",
        help="Only emulate the plots with the given name.",
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
        plot_names=config.plot_list,
    )

    # Create and train emulators
    hypercube.create_emulators()
    sweep_pages = create_sweep_plots(hypercube)
    validation_pages = create_validation_plots(hypercube)

    webpage = """<!DOCTYPE html>
    <head>
    <title>Parameter sweeps</title>
    </head> 
    </body><h1>Parameter sweeps for COLIBRE HyperCube wave 2</h1><h2>Parameter sweeps</h2>"""
    webpage += "<h3>Per plot</h3>"
    for page in sweep_pages:
        if "plot_" in page:
            webpage += f'<p><a href="{page}.html">{sweep_pages[page]}</a></p>'
    webpage += "<h3>Per parameter</h3>"
    for page in sweep_pages:
        if "param_" in page:
            webpage += f'<p><a href="{page}.html">{sweep_pages[page]}</a></p>'
    webpage += "<h2>Emulated simulations (for validation)</h2>"
    for page in validation_pages:
        webpage += f'<p><a href="{page}.html">{validation_pages[page]}</a></p>'
    webpage += "</body>"

    with open(f"{config.output}/index.html", "w") as handle:
        handle.write(webpage)
