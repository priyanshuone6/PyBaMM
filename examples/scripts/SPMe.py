import pybamm
import numpy as np
import matplotlib.pyplot as plt


def main(ics=None, timestep=1.0):
    # load model
    options = {"thermal": "lumped"}
    model = pybamm.lithium_ion.SPMe(options)
    # create geometry
    geometry = model.default_geometry
    # load parameter values and process model and geometry
    param = model.default_parameter_values
    # Change some parameters
    param['Typical current [A]'] = 1.0
    param.update(ics)
    
    param.process_model(model)
    param.process_geometry(geometry)
    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    # solve model
    # Hours
    f = 1/3600
    t_eval = np.linspace(0, timestep*f, 2)
    
    solution = model.default_solver.solve(model, t_eval)
    state_variables = ['Electrolyte concentration [mol.m-3]',
                       'Negative particle concentration [mol.m-3]',
                       'Positive particle concentration [mol.m-3]',
                       'Total heating [A.V.m-3]']
    output = {}
    x = model.variables['x'].evaluate().flatten()
    x_p = model.variables['x_p'].evaluate().flatten()
    x_n = model.variables['x_n'].evaluate().flatten()
    r_p = model.variables['r_p'].evaluate().flatten()
    r_n = model.variables['r_p'].evaluate().flatten()
    
    for i, iv in enumerate(state_variables):
    
        f = pybamm.ProcessedVariable(model.variables[iv],
                                     solution.t, solution.y, mesh=mesh)
        try:
            output[iv] = np.mean(f(solution.t, x=x))
        except:
            try:
                output[iv] = np.mean(f(solution.t, x=x_n, r=r_n))
            except:
                output[iv] = np.mean(f(solution.t, x=x_p, r=r_p))

    return output

if __name__ == '__main__':
    ics = {'Initial concentration in electrolyte [mol.m-3]':1000.0,
            'Initial concentration in negative electrode [mol.m-3]':19986.609595075,
            'Initial concentration in positive electrode [mol.m-3]':30730.755438556498,
            'Initial temperature [K]':298.15}
    concs = []
    for t in range(3):
        print(t)
        out = main(ics=ics)
        ics['Initial concentration in electrolyte [mol.m-3]'] = out['Electrolyte concentration [mol.m-3]']
        ics['Initial concentration in negative electrode [mol.m-3]'] = out['Negative particle concentration [mol.m-3]']
        ics['Initial concentration in positive electrode [mol.m-3]'] = out['Positive particle concentration [mol.m-3]']
        concs.append(out['Negative particle concentration [mol.m-3]'])
        print(concs[-1])
    plt.figure()
    plt.plot(concs)