import numpy as np
import pytest
import torch

from traffic_models.flows import ARZFlow as ARZFlowNumpy
from traffic_models.flows import DelCastilloBenitezFlux
from traffic_models.nn.arz_torch import ARZFlow_exponential


class TestARZFlowExponentialVsNumpy:
    def test_v_eq_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15, 0.18])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        
        v_torch = torch_flow.v_eq(rho_torch).detach().numpy()
        v_numpy = numpy_flow.v_eq(rho_values)
        
        assert np.allclose(v_torch, v_numpy, rtol=1e-5)

    def test_h_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15, 0.18])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        
        h_torch = torch_flow.h(rho_torch).detach().numpy()
        h_numpy = numpy_flow.h(rho_values)
        
        assert np.allclose(h_torch, h_numpy, rtol=1e-5)

    def test_q_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15])
        v_values = np.array([25.0, 20.0, 15.0])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        v_torch = torch.tensor(v_values, dtype=torch.float32)
        
        q_torch = torch_flow.q(rho_torch, v_torch).detach().numpy()
        q_numpy = numpy_flow.q(rho_values, v_values)
        
        assert np.allclose(q_torch, q_numpy, rtol=1e-5)

    def test_v_from_q_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        # Use density values that yield positive velocity: v = q/rho - h(rho) > 0
        rho_values = np.array([0.05, 0.1, 0.12])
        q_values = np.array([2.0, 3.0, 3.5])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        q_torch = torch.tensor(q_values, dtype=torch.float32)
        
        v_torch = torch_flow.v(rho_torch, q_torch).detach().numpy()
        v_numpy = numpy_flow.v(rho_values, q_values)
        
        assert np.allclose(v_torch, v_numpy, rtol=1e-5)

    def test_flux_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15])
        v_values = np.array([25.0, 20.0, 15.0])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        v_torch = torch.tensor(v_values, dtype=torch.float32)
        q_torch = torch_flow.q(rho_torch, v_torch)
        
        U_torch = torch.stack([rho_torch, q_torch], dim=0)
        U_numpy = np.stack([rho_values, numpy_flow.q(rho_values, v_values)], axis=0)
        
        flux_torch = torch_flow(U_torch).detach().numpy()
        flux_numpy = numpy_flow(U_numpy)
        
        assert np.allclose(flux_torch, flux_numpy, rtol=1e-5)

    def test_source_term_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15])
        v_values = np.array([25.0, 20.0, 15.0])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        v_torch = torch.tensor(v_values, dtype=torch.float32)
        q_torch = torch_flow.q(rho_torch, v_torch)
        
        U_torch = torch.stack([rho_torch, q_torch], dim=0)
        U_numpy = np.stack([rho_values, numpy_flow.q(rho_values, v_values)], axis=0)
        
        source_torch = torch_flow.source_term(U_torch).detach().numpy()
        source_numpy = numpy_flow.source_term(U_numpy)
        
        assert np.allclose(source_torch, source_numpy, rtol=1e-5)

    def test_lambdas_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15])
        v_values = np.array([25.0, 20.0, 15.0])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        v_torch = torch.tensor(v_values, dtype=torch.float32)
        q_torch = torch_flow.q(rho_torch, v_torch)
        
        U_torch = torch.stack([rho_torch, q_torch], dim=0)
        U_numpy = np.stack([rho_values, numpy_flow.q(rho_values, v_values)], axis=0)
        
        lambdas_torch = torch_flow.lambdas(U_torch).detach().numpy()
        lambdas_numpy = numpy_flow.lambdas(U_numpy)
        
        assert np.allclose(lambdas_torch, lambdas_numpy, rtol=1e-5)

    def test_rho_dh_matches_numpy(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        numpy_flux = DelCastilloBenitezFlux(v_max=v_max, rho_max=rho_max, c_jam=c_jam)
        numpy_flow = ARZFlowNumpy(flux_function=numpy_flux, tau=tau)
        
        rho_values = np.array([0.05, 0.1, 0.15])
        rho_torch = torch.tensor(rho_values, dtype=torch.float32)
        
        rho_dh_torch = torch_flow.rho_dh(rho_torch).detach().numpy()
        rho_dh_numpy = numpy_flow.rho_dh(rho_values)
        
        assert np.allclose(rho_dh_torch, rho_dh_numpy, rtol=1e-5)

    def test_extreme_values(self):
        v_max, rho_max, c_jam, tau = 30.0, 0.2, -5.0, 10.0
        torch_flow = ARZFlow_exponential(v_max_init=v_max, rho_max_init=rho_max, c_jam=c_jam, tau=tau)
        # At v=0 (jam), rho_eq should return rho_max exactly
        zero_v = torch.tensor(0.0)
        rho_torch = torch_flow.rho_eq(zero_v).detach().numpy()
        assert np.isclose(rho_torch, rho_max, atol=1e-5)
        
