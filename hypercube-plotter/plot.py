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
    x_units=None,
    y_units=None,
    labels=None,
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

            y = mock_values[mock_name]["dependent"]

            yerr_min = None
            yerr_max = None
            if "dependent_error" in mock_values[mock_name]:
                yerr = mock_values[mock_name]["dependent_error"]
                yerr_min = y - yerr
                yerr_max = y + yerr

            if log_y:
                ax.set_yscale("log")
                y = 10.0 ** y
                if not yerr_min is None:
                    yerr_min = 10.0 ** yerr_min
                    yerr_max = 10.0 ** yerr_max

            if not x_units is None:
                x = unyt.unyt_array(x, units=x_units)
            if not y_units is None:
                y = unyt.unyt_array(y, units=y_units)
                if not yerr_min is None:
                    yerr_min = unyt.unyt_array(yerr_min, units=y_units)
                    yerr_max = unyt.unyt_array(yerr_max, units=y_units)

            if not yerr_min is None:
                ax.fill_between(x, yerr_min, yerr_max, color=col, alpha=0.2)

            if labels is None:
                label = f"{parameter_name}={np.round(mock_parameters[mock_name][parameter], 3)}"
            else:
                label = labels[mock_name]
            ax.plot(
                x,
                y,
                color=col,
                label=label,
            )

        plt.legend(loc="best")

        if log_x:
            fitting_limits = 10.0 ** np.array(fitting_limits)
        if not x_units is None:
            x_range = unyt.unyt_array(x_range, units=x_units)
            fitting_limits = unyt.unyt_array(fitting_limits, units=x_units)
        if not y_units is None:
            y_range = unyt.unyt_array(y_range, units=y_units)

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        # Indicate emulator fitting rate
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
    pages = {}
    for plot in data.plots:
        webpages[f"mock_{plot.name}"] = html_header + f"<h1>Plot: {plot.title}</h1><p>"
        pages[
            f"mock_{plot.name}"
        ] = f"Emulated simulation output (validation) for {plot.title}"
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
            mock_labels = {"sim": "simulation", "mock": "emulator same params"}

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
                x_units=plot.x_units,
                y_units=plot.y_units,
                labels=mock_labels,
            )
            plt.savefig(f"{data.path_to_output}/run_{run}_{plot.name}.png")
            fig.clf()
            plt.close()
            webpages[f"mock_{plot.name}"] += f'<img src="run_{run}_{plot.name}.png">'

    for webpage in webpages:
        webpages[webpage] += "</p></body>"
        with open(f"{data.path_to_output}/{webpage}.html", "w") as handle:
            handle.write(webpages[webpage])

    return pages


def create_sweep_plots(data: Hypercube, num_of_lines: int = 6):
    """
    Creates parameter sweeps for each plot
    """

    webpages = {}
    pages = {}
    for plot in data.plots:
        webpages[f"plot_{plot.name}"] = html_header + f"<h1>Plot: {plot.title}</h1><p>"
        pages[f"plot_{plot.name}"] = f"Parameter sweeps for {plot.title}"
    for parameter in data.parameter_names:
        pname = parameter.replace(":", "_")
        webpages[f"param_{pname}"] = html_header + f"<h1>Parameter: {parameter}</h1><p>"
        pages[f"param_{pname}"] = f"All plots for sweep of parameter {parameter}"

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
                x_units=plot.x_units,
                y_units=plot.y_units,
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

    return pages
