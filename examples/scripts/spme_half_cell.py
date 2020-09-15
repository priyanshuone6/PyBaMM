import pybamm
import numpy as np

# Start with a Base Model
model = pybamm.BaseModel(name="half cell")

# Define parameters
L_s = pybamm.Parameter("Separator thickness [m]")
L_n = pybamm.Parameter("Electrode thickness [m]")
R = pybamm.Parameter("Particle radius [m]")
L = L_s / L_n  # dimensionless separator thickness
tau_e = pybamm.Parameter("Electrolyte diffusion timescale [s]")
tau = pybamm.Parameter("Electrode diffusion timescale [s]")
D = tau_e / tau  # ratio of diffusion timescales

eps = pybamm.Parameter("Electrolyte volume fraction")
c_e_0 = pybamm.Parameter("Dimensionless initial electrolyte concentration")
c_0 = pybamm.Parameter("Dimensionless initial particle concentration")

# Times have been scaled with the electrolyte diffusion time, macroscopic lengths
# have been scaled with the electrode thickness, and microscopic lengths have been
# scaled with the particle radius
model.timescale = tau_e
model.length_scales = {"separator": L_n, "electrode": L_n, "particle": R}

# Define variables
x_s = pybamm.SpatialVariable("x_s", domain="separator", coord_sys="cartesian")
x_n = pybamm.SpatialVariable("x_n", domain="electrode", coord_sys="cartesian")
r = pybamm.SpatialVariable(
    "r",
    domain="particle",
    auxiliary_domains={"secondary": "electrode"},
    coord_sys="spherical polar",
)

c = pybamm.Variable(
    "Electrode particle concentration",
    domain="particle",
    auxiliary_domains={"secondary": "electrode"},
)
c_e_s = pybamm.Variable("Separator electrolyte concentration", domain="separator")
c_e_n = pybamm.Variable("Electrode electrolyte concentration", domain="electrode")
c_e = pybamm.Concatenation(c_e_s, c_e_n)

# Write governing equations
i_e_s = pybamm.PrimaryBroadcast(1, "separator")
i_e_n = 1 - x_n
i_e = pybamm.Concatenation(i_e_s, i_e_n)
N_e = -pybamm.grad(c_e) - i_e
dc_e_dt = -pybamm.div(N_e) / eps

N = -D * pybamm.grad(c)
dc_dt = -pybamm.div(N)

model.rhs = {c_e: dc_e_dt, c: dc_dt}

# Add the boundary and initial conditions
model.boundary_conditions = {
    c_e: {"left": (-1, "Neumann"), "right": (0, "Neumann")},
    c: {"left": (0, "Neumann"), "right": (-1, "Neumann")},
}
model.initial_conditions = {c_e: c_e_0, c: c_0}

# Add desired output variables
model.variables = {
    "Electrolyte concentration": c_e,
    "Electrode particle concentration": c,
    "Surface concentration": pybamm.surf(c),
    "Electrolyte flux": N_e,
    "Electrode particle flux": N,
}

# Define geometry
geometry = {
    "separator": {x_s: {"min": -L, "max": 0}},
    "electrode": {x_n: {"min": 0, "max": 1}},
    "particle": {r: {"min": 0, "max": 1}},
}

# Set parameter values
param = pybamm.ParameterValues(
    {
        "Separator thickness [m]": 25 * 1e-6,
        "Electrode thickness [m]": 100 * 1e-6,
        "Particle radius [m]": 1e-5,
        "Electrolyte diffusion timescale [s]": 100,
        "Electrode diffusion timescale [s]": 1000,
        "Electrolyte volume fraction": 0.4,
        "Dimensionless initial electrolyte concentration": 2,
        "Dimensionless initial particle concentration": 1,
    }
)

# Process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# Pick mesh, spatial method, and discretise
submesh_types = {
    "separator": pybamm.Uniform1DSubMesh,
    "electrode": pybamm.Uniform1DSubMesh,
    "particle": pybamm.Uniform1DSubMesh,
}
var_pts = {x_s: 10, x_n: 30, r: 30}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {
    "separator": pybamm.FiniteVolume(),
    "electrode": pybamm.FiniteVolume(),
    "particle": pybamm.FiniteVolume(),
}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 10, 100)
solution = solver.solve(model, t)

# Extract output variables
c_e_out = solution["Electrolyte concentration"]
c_surf_out = solution["Surface concentration"]

# Plot
pybamm.dynamic_plot(
    solution,
    [
        "Electrolyte concentration",
        # "Electrode particle concentration",
        "Surface concentration",
    ],
)
