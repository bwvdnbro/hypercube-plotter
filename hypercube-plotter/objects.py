from typing import List, Union
import yaml
from swiftemulator.io.swift import load_parameter_files, load_pipeline_outputs
from swiftemulator.emulators.gaussian_process import GaussianProcessEmulator
from swiftemulator.backend.model_values import ModelValues

from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl


class Plot(object):
    def __init__(
        self,
        name: str,
        fitting_range: List[float],
        x_lim: List[float],
        y_lim: List[float],
        observations: List[str],
        redshift_range: List[float],
        log_x: bool = False,
        log_y: bool = False,
        x_units: Union[str, None] = None,
        y_units: Union[str, None] = None,
    ):
        self.name = name
        self.x_min = float(x_lim[0])
        self.x_max = float(x_lim[1])
        self.y_min = float(y_lim[0])
        self.y_max = float(y_lim[1])

        self.observations = observations
        self.redshift_range = redshift_range
        self.x_units = x_units
        self.y_units = y_units

        self.log_x = log_x
        self.log_y = log_y

        if self.log_x:
            self.fitting_limits = [
                np.log10(float(fitting_range[0])),
                np.log10(float(fitting_range[1])),
            ]
        else:
            self.fitting_limits = [float(fitting_range[0]), float(fitting_range[1])]


# Class with metadata
class Hypercube(object):
    def __init__(
        self,
        path_to_param_config: str,
        path_to_plot_config: str,
        path_to_output: str,
        path_to_data: str,
        path_to_params: str,
        plot_names: Union[List[str], None] = None,
    ):

        self.path_to_param_config = path_to_param_config
        self.path_to_plot_config = path_to_plot_config
        self.path_to_data = path_to_data
        self.path_to_params = path_to_params
        self.path_to_output = path_to_output
        self.plot_names = plot_names

        self.plots: List[Plot] = []
        self.emulators: List[GaussianProcessEmulator] = []

        self._load_params()
        self._load_plots()

    def _load_params(self):
        with open(self.path_to_param_config, "r") as handler:
            param_data = yaml.safe_load(handler)
            self.parameters = param_data["parameters"]

            self.number_of_params = len(self.parameters)

            self.parameter_names = [param["name"] for param in self.parameters]
            self.parameter_printable_names = [
                param["printname"] for param in self.parameters
            ]
            self.log_parameter_names = [
                param["name"] for param in self.parameters if param["log"]
            ]
            self.parameter_limits = [param["limits"] for param in self.parameters]

            self.parameter_name_default_values = {
                param["name"]: param["default"] for param in self.parameters
            }

            param_files = [Path(x) for x in glob(f"{self.path_to_params}/*.yml")]
            value_files = [Path(x) for x in glob(f"{self.path_to_data}/*.yml")]

            print(f"Number of simulations: {len(param_files)}")

            assert len(param_files) == len(value_files), (
                "The number of files with params must be the "
                "same as the number of files with the data"
            )

            self.filenames_params = {
                filename.stem: filename for filename in param_files
            }
            self.filenames_data = {filename.stem: filename for filename in value_files}

            model_spec, model_params = load_parameter_files(
                filenames=self.filenames_params,
                parameters=self.parameter_names,
                log_parameters=self.log_parameter_names,
                parameter_printable_names=self.parameter_printable_names,
                parameter_limits=self.parameter_limits,
            )

            self.model_specification = model_spec
            self.model_parameters = model_params

            for run in self.model_parameters.model_parameters:
                self.model_parameters.model_parameters[run][
                    "COLIBREFeedback:SNII_energy_erg"
                ] *= 1.0e-51

            print("Hypercube parameters:")
            print("---------------------")
            for c, param in enumerate(self.parameters, start=1):
                print(c, param)
            print(f"Number of parameters in the hypercube: {self.number_of_params} \n")

        return

    def _load_plots(self):
        with open(self.path_to_plot_config, "r") as handler:
            plots_data = yaml.safe_load(handler)
            plot_names = list(plots_data.keys())
            if not self.plot_names is None:
                for plot_name in self.plot_names:
                    if not plot_name in plot_names:
                        raise AttributeError(
                            f"Invalid plot name: {plot_name}! Possible names: {plot_names}."
                        )

            for plot_name in plot_names:
                if plot_name in self.plot_names:
                    plot_data = plots_data[plot_name]
                    if "redshift_range" in plot_data:
                        redshift_range = plot_data["redshift_range"]
                    else:
                        redshift_range = [0.0, 1000.0]
                    x_units = None
                    if "x_units" in plot_data:
                        x_units = plot_data["x_units"]
                    y_units = None
                    if "y_units" in plot_data:
                        y_units = plot_data["y_units"]
                    self.plots.append(
                        Plot(
                            name=plot_name,
                            x_lim=plot_data["x_range"],
                            y_lim=plot_data["y_range"],
                            fitting_range=plot_data["fitting_range"],
                            observations=plot_data["observations"],
                            redshift_range=redshift_range,
                            log_x=plot_data["x_log"],
                            log_y=plot_data["y_log"],
                            x_units=x_units,
                            y_units=y_units,
                        )
                    )
        return

    def create_emulators(self):
        """
        Creates Gaussian emulators for all plots provided in the plot config
        """

        scaling_relations = [plot.name for plot in self.plots]
        log_dependent = [plot.name for plot in self.plots if plot.log_y]
        log_independent = [plot.name for plot in self.plots if plot.log_x]
        fitting_limits = [plot.fitting_limits for plot in self.plots]

        values, units = load_pipeline_outputs(
            filenames=self.filenames_data,
            scaling_relations=scaling_relations,
            log_independent=log_independent,
            log_dependent=log_dependent,
        )
        self.scaling_relations = {}

        # Build Gaussian emulator for each plot specified in the plot config
        for relation_name, fitting_lims in tqdm(zip(scaling_relations, fitting_limits)):

            print(f"Relation: {relation_name}")
            print(f"Fitting lims: [{fitting_lims[0]}, {fitting_lims[1]}]")

            relation = values[relation_name]

            # Dict to be filled
            relation_masked = {}

            # Apply fitting mask to each run's data separately
            for run in relation.keys():

                x = relation.model_values[run]["independent"]
                y = relation.model_values[run]["dependent"]
                e = relation.model_values[run]["dependent_error"]

                # Create mask (select only values within fitting range
                # and remove any NaNs or infs)
                mask_fit = np.logical_and(x > fitting_lims[0], x < fitting_lims[1])
                mask_finite_val = np.logical_and(~np.isnan(y), ~np.isinf(y))
                mask_finite_err = np.logical_and(~np.isnan(e), ~np.isinf(e))
                mask_finite = np.logical_and(mask_finite_val, mask_finite_err)
                mask_total = np.logical_and(mask_fit, mask_finite)

                relation_masked_single_run = {
                    "independent": x[mask_total],
                    "dependent": y[mask_total],
                    "dependent_error": e[mask_total],
                }
                """
                pl.fill_between(x[mask_total], y[mask_total]-e[mask_total], y[mask_total]+e[mask_total], color="C0", alpha=0.4)
                pl.plot(x[mask_total], y[mask_total])
                pl.tight_layout()
                pl.savefig(f"input_{run}.png", dpi=300)
                pl.close()
                """
                pl.plot(x[mask_total], y[mask_total])
                relation_masked[run] = relation_masked_single_run

            self.scaling_relations[relation_name] = relation_masked

            pl.tight_layout()
            pl.savefig(f"input.png", dpi=300)
            pl.close()

            try:
                model = ModelValues(relation_masked)
                gpe = GaussianProcessEmulator()
                gpe.fit_model(
                    model_specification=self.model_specification,
                    model_parameters=self.model_parameters,
                    model_values=model,
                )
                self.emulators.append(gpe)
            except:
                self.emulators.append(None)

        return
