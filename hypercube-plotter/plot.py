import matplotlib.pylab as plt
import numpy as np
from swiftemulator.mocking import mock_sweep
from objects import Hypercube


def create_sweep_plots(data: Hypercube, num_of_lines: int = 6):
    """
    Creates parameter sweeps for each plot
    """

    for plot, gpe in zip(data.plots, data.emulators):
        for count, (parameter, name) in enumerate(zip(data.parameter_names,
                                                      data.parameter_printable_names)):

            mock_values, mock_parameters = mock_sweep(gpe, data.model_specification,
                                                      num_of_lines,
                                                      parameter,
                                                      data.parameter_name_default_values)

            col_index = np.linspace(0, 1, len(mock_values.keys()))
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            for index, mock_name in enumerate(mock_values.keys()):
                col = plt.cm.viridis(col_index[index])

                if plot.log_x:
                    ax.set_xscale("log")
                    x = 10.0 ** mock_values[mock_name]["independent"]
                else:
                    x = mock_values[mock_name]["independent"]

                if plot.log_y:
                    ax.set_yscale("log")
                    y = 10.0 ** mock_values[mock_name]["dependent"]
                else:
                    y = mock_values[mock_name]["dependent"]

                ax.plot(x, y, color=col,
                        label=f"{name}={np.round(mock_parameters[mock_name][parameter], 3)}")

            plt.legend(loc="best")
            ax.set_xlim(plot.x_min, plot.x_max)
            ax.set_ylim(plot.y_min, plot.y_max)

            # Indicate emulator fitting rate
            if plot.log_y:
                ax.axvline(x=10.0 ** plot.fitting_limits[0], color='grey', lw=3, dashes=(3, 3))
                ax.axvline(x=10.0 ** plot.fitting_limits[1], color='grey', lw=3, dashes=(3, 3))
            else:
                ax.axvline(x=plot.fitting_limits[0], color='grey', lw=3, dashes=(3, 3))
                ax.axvline(x=plot.fitting_limits[1], color='grey', lw=3, dashes=(3, 3))

            plt.savefig(f"{data.path_to_output}/{plot.name}_{count}.png")
            fig.clf()
            plt.close()

    return
