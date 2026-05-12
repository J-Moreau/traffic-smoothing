import numpy as np
import torch

import traffic_models.godunov as gdnv_numpy
import traffic_models.nn.flows_torch
import traffic_models.nn.godunov_torch as gdnv_torch
from traffic_models.flows import GreenshieldsFlow
from traffic_models.godunov import godunov_jacobian


def test_torch_godunov_is_the_same_as_the_numpy_one():
    dt = 0.1
    dx = 1.0
    v_max = 1.0
    rho_max = 1.0

    flow_nn = traffic_models.nn.flows_torch.GreenshieldsFlow(v_max, rho_max)
    flow = GreenshieldsFlow(v_max, rho_max)

    # Compute godunov step output
    rho = torch.tensor([0, 0.5], dtype=torch.float64, requires_grad=True)
    torch_output = gdnv_torch.godunov_step(
        rho, flow_nn, torch.exp(flow_nn.get_parameter("log_rho_max")) / 2, dt, dx
    )
    rho = np.array([0, 0.5])
    np_output = gdnv_numpy.godunov_step(rho, flow, dt, dx)

    # Assert that the outputs are close
    assert np.allclose(torch_output.detach().numpy(), np_output, atol=1e-6)


def test_autograd_jacobian_is_the_same_as_the_closed_form_equation():
    dt = 0.1
    dx = 0.1
    v_max = 1.0
    rho_max = 1.0

    flow_nn = traffic_models.nn.flows_torch.GreenshieldsFlow(v_max, rho_max)
    flow = GreenshieldsFlow(v_max, rho_max)

    rho_input = [0.1, 0.2, 0.8, 0.9]

    # example below doesn't work
    # rho_input = [0.1, 0.1, 0.9, 0.9]
    # because the minimum(d,s) is non differentiable when d=s
    # hence the gradient "chooses" one side

    # Compute autograd jacobian
    rho = torch.tensor(rho_input, dtype=torch.float32, requires_grad=True)
    autograd_jac = (
        torch.autograd.functional.jacobian(
            lambda rho: gdnv_torch.godunov_step(
                rho,
                flow_nn,
                torch.exp(flow_nn.get_parameter("log_rho_max")) / 2,
                dt,
                dx,
                detach_boundaries=True,
            ),
            rho,
        )
        .squeeze()
        .detach()
        .numpy()
    )

    # Compute closed-form jacobian
    rho = np.array(rho_input)
    closed_form_jac = godunov_jacobian(rho, flow, dt, dx)

    # Assert they are close
    assert autograd_jac.shape == closed_form_jac.shape
    assert (abs(autograd_jac - closed_form_jac) < 1e-6).all()
