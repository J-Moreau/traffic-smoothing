"""
Test script for flows_torch.py implementation.
"""

import torch

import traffic_models.flows as numpy_flows
from traffic_models.nn.flows_torch import torch_implem_from_numpy


class TestTorchImplFromNumpy:
    def test_greenshields_conversion(self) -> None:
        numpy_flow = numpy_flows.GreenshieldsFlow(v_max=30.0, rho_max=10.0)
        torch_flow = torch_implem_from_numpy(numpy_flow)
        assert torch.isclose(torch_flow.v_max, torch.tensor(30.0))
        assert torch.isclose(torch_flow.rho_max, torch.tensor(10.0))

    def test_triangular_conversion(self) -> None:
        numpy_flow = numpy_flows.TriangularFlow(rho_c=5.0, Q_c=50.0, rho_max=10.0, rho_free_flow=4.0)
        torch_flow = torch_implem_from_numpy(numpy_flow)
        assert torch.isclose(torch_flow.rho_c, torch.tensor(5.0))
        assert torch.isclose(torch_flow.rho_max, torch.tensor(10.0))

    def test_arz_conversion(self) -> None:
        numpy_flow = numpy_flows.ARZFlow(
            flux_function=numpy_flows.PowerLawFlux(v_max=30.0, rho_max=0.2, gamma=1.5),
            tau=3.0,
        )
        torch_flow = torch_implem_from_numpy(numpy_flow)
        assert torch.isclose(torch_flow.v_max, torch.tensor(30.0))
        assert torch.isclose(torch_flow.rho_max, torch.tensor(0.2))
        assert torch.isclose(torch_flow.gamma, torch.tensor(1.5))
        assert torch.isclose(torch_flow.tau, torch.tensor(3.0))

