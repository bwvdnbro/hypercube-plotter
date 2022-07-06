import matplotlib.pylab as plt
import numpy as np
from swiftemulator.mocking import mock_sweep
from objects import Hypercube
from velociraptor.observations import load_observations
import unyt


def create_plot(
    ax,
    mock_values,
    mock_parameters,
    parameter,
    parameter_name,
    x_range,
    y_range,
    observations,
    redshift_range,
    log_x=True,
    log_y=True,
    fitting_limits=None,
):

    with unyt.matplotlib_support:
        for observation in observations:
            if observation.endswith(".hdf5"):
                for obs in load_observations(
                    observation, redshift_bracket=redshift_range
                ):
                    obs.plot_on_axes(ax, {"color": "k", "alpha": 0.5})
            elif observation.endswith(".py"):
                with open(observation, "r") as handle:
                    exec(handle.read())
            else:
                raise AttributeError(f"Unknown observational data type: {observation}!")

    col_index = np.linspace(0, 1, len(mock_values.keys()))

    for index, mock_name in enumerate(mock_values.keys()):
        col = plt.cm.viridis(col_index[index])

        if log_x:
            ax.set_xscale("log")
            x = 10.0 ** mock_values[mock_name]["independent"]
        else:
            x = mock_values[mock_name]["independent"]

        if log_y:
            ax.set_yscale("log")
            y = 10.0 ** mock_values[mock_name]["dependent"]
        else:
            y = mock_values[mock_name]["dependent"]

        ax.plot(
            x,
            y,
            color=col,
            label=f"{parameter_name}={np.round(mock_parameters[mock_name][parameter], 3)}",
        )

    plt.legend(loc="best")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    # Indicate emulator fitting rate
    if log_y:
        ax.axvline(x=10.0 ** fitting_limits[0], color="grey", lw=3, dashes=(3, 3))
        ax.axvline(x=10.0 ** fitting_limits[1], color="grey", lw=3, dashes=(3, 3))
    else:
        ax.axvline(x=fitting_limits[0], color="grey", lw=3, dashes=(3, 3))
        ax.axvline(x=fitting_limits[1], color="grey", lw=3, dashes=(3, 3))

    return


html_header = """<!DOCTYPE html>
<head>
<title>Parameter sweeps</title>
</head> 
</body>"""


def create_validation_plots(data: Hypercube):

    webpages = {}
    for plot in data.plots:
        webpages[f"mock_{plot.name}"] = html_header + f"<h1>Plot: {plot.name}</h1><p>"
    for plot, gpe in zip(data.plots, data.emulators):
        if gpe is None:
            print(f"No emulator for plot {plot.name}.")
            continue

        x_range = np.linspace(*plot.fitting_limits, 100)

        for run in data.model_parameters.model_parameters:
            model = data.model_parameters.model_parameters[run]
            pred, pred_var = gpe.predict_values(x_range, model)

            mock_values = {
                "sim": data.scaling_relations[plot.name][run],
                "mock": {"independent": x_range, "dependent": pred},
            }
            mock_params = {"sim": model, "mock": model}

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            create_plot(
                ax,
                mock_values,
                mock_params,
                "COLIBREFeedback:SNII_energy_erg",
                f"run_{run}_mock",
                [plot.x_min, plot.x_max],
                [plot.y_min, plot.y_max],
                plot.observations,
                plot.redshift_range,
                log_x=plot.log_x,
                log_y=plot.log_y,
                fitting_limits=plot.fitting_limits,
            )
            plt.savefig(f"{data.path_to_output}/run_{run}_{plot.name}.png")
            fig.clf()
            plt.close()
            webpages[f"mock_{plot.name}"] += f'<img src="run_{run}_{plot.name}.png">'

    for webpage in webpages:
        webpages[webpage] += "</p></body>"
        with open(f"{data.path_to_output}/{webpage}.html", "w") as handle:
            handle.write(webpages[webpage])


def create_sweep_plots(data: Hypercube, num_of_lines: int = 6):
    """
    Creates parameter sweeps for each plot
    """

    webpages = {}
    for plot in data.plots:
        webpages[f"plot_{plot.name}"] = html_header + f"<h1>Plot: {plot.name}</h1><p>"
    for parameter in data.parameter_names:
        pname = parameter.replace(":", "_")
        webpages[f"param_{pname}"] = html_header + f"<h1>Parameter: {parameter}</h1><p>"

    for plot, gpe in zip(data.plots, data.emulators):
        if gpe is None:
            print(f"No emulator for plot {plot.name}.")
            continue

        for count, (parameter, name) in enumerate(
            zip(data.parameter_names, data.parameter_printable_names)
        ):

            mock_values, mock_parameters = mock_sweep(
                gpe,
                data.model_specification,
                num_of_lines,
                parameter,
                data.parameter_name_default_values,
            )

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            create_plot(
                ax,
                mock_values,
                mock_parameters,
                parameter,
                name,
                [plot.x_min, plot.x_max],
                [plot.y_min, plot.y_max],
                observations=plot.observations,
                redshift_range=plot.redshift_range,
                log_x=plot.log_x,
                log_y=plot.log_y,
                fitting_limits=plot.fitting_limits,
            )

            plt.savefig(f"{data.path_to_output}/{plot.name}_{count}.png")
            fig.clf()
            plt.close()
            webpages[f"plot_{plot.name}"] += f'<img src="{plot.name}_{count}.png">'
            pname = parameter.replace(":", "_")
            webpages[f"param_{pname}"] += f'<img src="{plot.name}_{count}.png">'

    for webpage in webpages:
        webpages[webpage] += "</p></body>"
        with open(f"{data.path_to_output}/{webpage}.html", "w") as handle:
            handle.write(webpages[webpage])

    return
