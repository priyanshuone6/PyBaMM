#
# Simulate drive cycle loaded from csv file
#
# import pybamm
# import pandas as pd
# import os

# os.chdir(pybamm.__path__[0] + "/..")


# # load model and update parameters so the input current is the US06 drive cycle
# model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
# param = model.default_parameter_values


# # import drive cycle from file
# drive_cycle = pd.read_csv(
#     "pybamm/input/drive_cycles/WLTC.csv", comment="#", header=None
# ).to_numpy()

# # create interpolant
# timescale = param.evaluate(model.timescale)
# current_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)

# # set drive cycle
# param["Current function [A]"] = current_interpolant


# # create and run simulation using the CasadiSolver in "fast" mode, remembering to
# # pass in the updated parameters
# sim = pybamm.Simulation(
#     model, parameter_values=param, solver=pybamm.CasadiSolver(mode="fast")
# )
# sim.solve()
# sim.plot(
#     [
#         "Negative particle surface concentration [mol.m-3]",
#         "Electrolyte concentration [mol.m-3]",
#         "Positive particle surface concentration [mol.m-3]",
#         "Current [A]",
#         "Negative electrode potential [V]",
#         "Electrolyte potential [V]",
#         "Positive electrode potential [V]",
#         "Terminal voltage [V]",
#         "X-averaged cell temperature",
#     ]
# )
import pybamm
import pandas as pd  # needed to read the csv data file

pybamm.set_logging_level("INFO")
chemistry = pybamm.parameter_sets.Chen2020
parameter_values = pybamm.ParameterValues(chemistry=chemistry)
model = pybamm.lithium_ion.SPM()
# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/WLTC.csv", comment="#", header=None
).to_numpy()
drive_cycle[:, 1] = -drive_cycle[:, 1]
timescale = parameter_values.evaluate(model.timescale)
current_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)

parameter_values["Current function [A]"] = current_interpolant
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sim.solve(t_eval=None)
sim.plot(
    ["Current [A]", "Terminal voltage [V]"],
    time_unit="seconds",
    variable_limits="tight",
)
