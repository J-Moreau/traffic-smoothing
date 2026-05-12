"""Vehicle Flow functions Q = f(rho)"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class GreenshieldsFlow:
    v_max: float
    rho_max: float
    rho_c: float = field(init=False)  # critical density
    n_params: int = 2

    def __post_init__(self) -> None:
        self.rho_c = self.rho_max / 2

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        return rho * self.v_max * (1 - rho / self.rho_max)

    def derivative[T: (NDArray, float)](self, rho: T) -> T:
        """dQ_drho"""
        return self.v_max * (1 - 2 * rho / self.rho_max)

    def dflow_dparam(self, rho: NDArray| float) -> NDArray:
        """
        derivative of Flow over the parameters
        Input: rho, shape n
        Output: d Flow / d [V_max rho_max], shape (n, 2)
        """
        return np.stack([
            self.v_max * rho**2 / self.rho_max**2,
            rho * ( 1 - rho/self.rho_max)
        ], axis=1)
    
    def update_params(self, v_max: float, rho_max: float) -> None:
        """
        Update the parameters of the flow function
        """
        self.v_max = v_max
        self.rho_max = rho_max
        self.rho_c = rho_max / 2
        
    def density_from_velocity[T:(NDArray,float)](self, v: T) -> T:
        """
        Inverse function: density from velocity
        """
        return self.rho_max * (1 - v / self.v_max)
    
    def velocity_from_density[T:(NDArray,float)](self, rho: T) -> T:
        """
        Inverse function: velocity from density
        """
        return self.v_max * (1 - rho / self.rho_max)
    
    def dv_dparam(self, rho):
        """
        derivative of V(rho) over parameters [v_max rho_max]
        """
        return np.stack([
            1 - rho/self.rho_max,
            self.v_max * rho / self.rho_max**2,
        ], axis=1)

    def drho_dparam(self, v):
        """
        derivative of Rho(v) over parameters [v_max rho_max]
        """
        return np.stack([
            self.rho_max * v / self.v_max**2,
            1 - v/self.v_max,
        ], axis=1)
    
    def drho_dv(self, v: NDArray|float):
        """
        derivative of Rho(v) over v
        """
        return -self.rho_max / self.v_max * np.ones_like(v)

    def dv_drho(self, rho: NDArray|float) -> NDArray:
        """
        derivative of V(rho) over rho
        """
        return -self.v_max / self.rho_max * np.ones_like(rho)



@dataclass
class TriangularFlow:
    rho_c: float  # critical density
    Q_c: float
    rho_max: float
    n_params: int = 3
    rho_free_flow: float = -1.0  # placeholder
    v_max: float = field(init=False)

    def __post_init__(self) -> None:
        # assert self.rho_c <= self.rho_max/2
        # This is observed in density diagrams
        # vmax should be the free flow speed
        self.v_max = self.Q_c / self.rho_c
        if self.rho_free_flow == -1.0:
            self.rho_free_flow = self.rho_c/2

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        rho_c = self.rho_c
        Q_c = self.Q_c
        rho_max = self.rho_max

        under_critical = (Q_c * rho / rho_c) * (rho < rho_c)
        over_critical = (Q_c * (rho_max - rho) / (rho_max - rho_c)) * (rho >= rho_c)
        Q = under_critical + over_critical
        return Q

    def velocity_from_density[T: (NDArray, float)](self, rho: T) -> T:
        """
        Inverse function: velocity from density
        """
        rho_c = self.rho_c
        Q_c = self.Q_c
        rho_max = self.rho_max

        under_critical = self.v_max * (rho < rho_c)
        over_critical = Q_c * (rho_max/rho - 1) / (rho_max - rho_c) * (rho >= rho_c)
        return under_critical + over_critical
    
    def derivative[T: (NDArray, float)](self, rho: T) -> T:
        """
        dQ_drho
        """
        rho_c = self.rho_c
        rho_max = self.rho_max
        Q_c = self.Q_c

        under_critical = Q_c / rho_c * (rho < rho_c)
        over_critical = -Q_c / (rho_max - rho_c) * (rho >= rho_c)
        return under_critical + over_critical
    
    def density_from_velocity[T: (NDArray, float)](self, v: T) -> T:
        """
        Inverse function: density from velocity
        This relation is not bijective for free flow conditions v >= v_max
        so we return an arbitrary constant free-flow density rho_free_flow
        In "Incorporation of Lagrangian Measurements in freeway TSE" (2010)
        Herrera et Bayen use:   rho_free_flow = 5/6 * rho_c
        """
        rho_c = self.rho_c
        Q_c = self.Q_c
        rho_max = self.rho_max

        # congestion flow slope
        w = Q_c  / (rho_max - rho_c)

        congestion =  rho_max / (1 + v/w)* (v <= self.v_max)
        free_flow = self.rho_free_flow * (v > self.v_max)
        return free_flow + congestion
    
    def drho_dv[T: (NDArray, float)](self, v: T) -> T:
        """
        derivative of Rho(v) over v
        """
        rho_c = self.rho_c
        Q_c = self.Q_c
        rho_max = self.rho_max
        w = Q_c / (rho_max - rho_c)

        congestion = -rho_max /w / (1 + v/w)**2 * (v <= self.v_max)
        free_flow = np.zeros_like(v) * (v > self.v_max)
        # return np.ones_like(v)
        return free_flow + congestion
    
    def dv_drho[T: (NDArray, float)](self, rho: T) -> T:
        """
        derivative of V(rho) over rho
        """
        rho_c = self.rho_c
        Q_c = self.Q_c
        rho_max = self.rho_max
        w = Q_c / (rho_max - rho_c)

        under_critical = np.zeros_like(rho) * (rho < rho_c)
        over_critical = - w * rho_max/rho**2 * (rho >= rho_c)
        # return np.ones_like(rho)
        return under_critical + over_critical

@dataclass
class ThreeParameterFlow:
    """From Fan & Seibold 2013 eq. 3"""
    lambda_: float
    p: float
    rho_max: float
    alpha: float
    rho_c: float = field(init=False)
    n_params: int = 4

    def __post_init__(self) -> None:
        """sets the critical density when Q is maximal"""
        lambda_ = self.lambda_
        p = self.p
        rho_max = self.rho_max
        a = np.sqrt(1 + (lambda_ * p) ** 2)
        b = np.sqrt(1 + (lambda_ * (1 - p)) ** 2)
        self.rho_c = rho_max * (
            p + (b - a) / (lambda_ * np.sqrt(np.abs(lambda_**2 - (b - a) ** 2)))
        )

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        return self.base_function(rho, self.rho_max, self.lambda_, self.p, self.alpha)

    @staticmethod
    def base_function(rho, rho_max, lambda_, p, alpha):
        a = np.sqrt(1 + (lambda_ * p) ** 2)
        b = np.sqrt(1 + (lambda_ * (1 - p)) ** 2)
        y = lambda_ * (rho / rho_max - p)
        return alpha * (a + (b - a) * rho / rho_max - np.sqrt(1 + y**2))

    def derivative[T: (NDArray, float)](self, rho: T) -> T:
        lambda_ = self.lambda_
        p = self.p
        rho_max = self.rho_max
        alpha = self.alpha

        a = np.sqrt(1 + (lambda_ * p) ** 2)
        b = np.sqrt(1 + (lambda_ * (1 - p)) ** 2)
        y = lambda_ * (rho / rho_max - p)
        return alpha * ((b - a) - lambda_ * y / np.sqrt(1 + y**2)) / rho_max


@dataclass
class QuadraticLinearFlow:
    """
    From "A traffic model for velocity assimilation", D B Work et al 2010. Eq 10,11
    
    A flow function that is quadratic for low densities and linear for high densities.
    This allows the inversion of the velocity - density relation
    """
    v_max: float
    rho_max: float
    rho_c: float
    v_c: float = field(init=False)
    n_params: int = 3

    def __post_init__(self) -> None:
        self.v_c = self.v_max * (1 - self.rho_c / self.rho_max)

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        # Quadratic for rho < rho_c, linear for rho >= rho_c
        quad = self.v_max * rho * (1 - rho / self.rho_max) * (rho < self.rho_c)
        lin = self.rho_c * self.v_max * (1 - rho/self.rho_max) * (rho >= self.rho_c)
        return quad + lin

    def derivative[T: (NDArray, float)](self, rho: T) -> T:
        # dQ/drho
        quad_deriv = self.v_max * (1 - 2 * rho / self.rho_max) * (rho < self.rho_c)
        lin_deriv = -self.v_max * self.rho_c / self.rho_max * (rho >= self.rho_c)
        return quad_deriv + lin_deriv
    
    def density_from_velocity[T:(NDArray,float)](self, v: T) -> T:
        free_flow = self.rho_max * (1 - v / self.v_max) * (v > self.v_c)
        congestion = self.rho_max / (1 + v / self.v_max * self.rho_max/self.rho_c) * (v <= self.v_c)
        return free_flow + congestion

    def velocity_from_density[T:(NDArray,float)](self, rho: T) -> T:
        free_flow = self.v_max * (1 - rho / self.rho_max) * (rho < self.rho_c)
        # congestion = self.rho_c * self.v_max * (1/rho - 1/self.rho_max) * (rho >= self.rho_c)
        congestion = self.rho_c * self.v_max * (
            np.divide(1, rho, where=rho>=self.rho_c) - 1/self.rho_max
            ) * (rho >= self.rho_c)
        return free_flow + congestion

    def drho_dv(self, v: NDArray|float) -> NDArray:
        free_flow =  (- self.rho_max  / self.v_max) * (v > self.v_c)
        congestion = self.rho_max**2 / (self.v_max * self.rho_c) / (
            1 + v / self.v_max * self.rho_max/self.rho_c
            )**2 * (v <= self.v_c)
        return free_flow + congestion
    
    def dv_drho(self, rho: NDArray|float) -> NDArray:
        free_flow = -self.v_max / self.rho_max * (rho < self.rho_c)
        # congestion = - self.rho_c * self.v_max / rho**2 * (rho >= self.rho_c)
        congestion = - self.rho_c * self.v_max * np.divide(1, rho**2, where=rho >= self.rho_c) * (rho >= self.rho_c)
        return free_flow + congestion

    def dflow_dparam(self, rho: NDArray) -> NDArray:
        """Placeholder for the derivative of the flow function over the parameters"""
        return np.ones((rho.shape[0], self.n_params))
    
    def update_params(self, v_max: float, rho_max: float, rho_c: float) -> None:
        """
        Update the parameters of the flow function
        """
        self.v_max = v_max
        self.rho_max = rho_max
        self.rho_c = rho_c
        self.v_c = v_max * (1 - rho_c / rho_max)


@dataclass
class OtherQuadraticLinearFlow:
    """
    A flow function that is quadratic for low densities and linear for high densities.
    This is not invertible, not be used in LWR
    """
    v_max: float
    rho_max: float
    rho_c: float
    w: float = 4.0
    v_c: float = field(init=False)
    n_params: int = 3

    def __post_init__(self) -> None:
        self.rho_1 = self.rho_c*2
        self.v_c = self.v_max * (1 - self.rho_c / self.rho_1)

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        # Quadratic for rho < rho_c, linear for rho >= rho_c
        quad = self.v_max * rho * (1 - rho / self.rho_1) * (rho < self.rho_c)
        lin = self.w * (self.rho_max - rho) * (rho >= self.rho_c)
        return quad + lin

    # def derivative[T: (NDArray, float)](self, rho: T) -> T:
    #     # dQ/drho
    #     quad_deriv = self.v_max * (1 - 2 * rho / self.rho_1) * (rho < self.rho_c)
    #     lin_deriv = -self.w * (rho >= self.rho_c)
    #     return quad_deriv + lin_deriv
    
    def density_from_velocity[T:(NDArray,float)](self, v: T) -> T:
        free_flow = self.rho_1 * (1 - v / self.v_max) * (v > self.v_c)
        congestion = self.rho_max / (1 + v / self.w) * (v <= self.v_c)
        return free_flow + congestion

    def velocity_from_density[T:(NDArray,float)](self, rho: T) -> T:
        free_flow = self.v_max * (1 - rho / self.rho_1) * (rho < self.rho_c)
        congestion = self.w * (
            np.divide(self.rho_max, rho, where=rho>=self.rho_c) - 1
            ) * (rho >= self.rho_c)
        return free_flow + congestion

    def drho_dv(self, v: NDArray|float) -> NDArray:
        free_flow =  (- self.rho_max  / self.v_max) * (v > self.v_c)
        congestion = self.rho_max**2 / (self.v_max * self.rho_c) / (
            1 + v / self.v_max * self.rho_max/self.rho_c
            )**2 * (v <= self.v_c)
        return free_flow + congestion
    
    def dv_drho(self, rho: NDArray|float) -> NDArray:
        free_flow = -self.v_max / self.rho_max * (rho < self.rho_c)
        # congestion = - self.rho_c * self.v_max / rho**2 * (rho >= self.rho_c)
        congestion = - self.rho_c * self.v_max * np.divide(1, rho**2, where=rho >= self.rho_c) * (rho >= self.rho_c)
        return free_flow + congestion


@dataclass
class ZeroFlow:
    """A dummy flow function that is always zero, the resulting physics is the identity"""
    rho_c = 0. # dummy value
    v_max = 30. # dummy value
    rho_max = 1.0 # dummy value

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        return np.zeros_like(rho)

    def density_from_velocity[T:(NDArray,float)](self, v: T) -> T:
        # dummy conversion
        return v

    def velocity_from_density[T:(NDArray,float)](self, rho: T) -> T:
        # dummy conversion
        return rho

    def derivative[T: (NDArray, float)](self, rho: T) -> T:
        return np.zeros_like(rho)

    def drho_dv(self, v: NDArray|float):
        return np.ones_like(v)

    def dv_drho(self, rho: NDArray|float) -> NDArray:
        return np.ones_like(rho)


@dataclass
class PowerLawFlux:
    """
    A simple generalization of the Greenshields flow function, with an additional exponent parameter gamma.
    """
    v_max: float
    rho_max: float
    gamma: float
    
    def density_from_velocity(self, v):
        return self.rho_max * (np.maximum(self.v_max - v, 0) / self.v_max) ** (1 / self.gamma)

    def velocity_from_density(self, rho):
        return self.v_max * (1 - (rho / self.rho_max) ** self.gamma)

    def __call__(self, rho):
        return self.v_max * rho * (1 - (rho / self.rho_max) ** self.gamma)
    
    def drho_dv(self, v):
        return self.rho_max * (np.maximum(self.v_max - v, 0) / self.v_max) ** (1 / self.gamma - 1) * (1 / self.gamma) * (1 / self.v_max)

    def dv_drho(self, rho):
        return - self.v_max * self.gamma * (rho / self.rho_max) ** (self.gamma - 1) / self.rho_max

@dataclass
class DelCastilloBenitezFlux():
    """
    Del Castillo, J.M., Benítez, F.G., 1995.
    On the functional form of the speed-density relationship — I: general theory.
    Transp. Res. Part B 29, 373-389.
    """
    v_max: float
    rho_max: float
    c_jam: float
    
    def velocity_from_density(self, rho):
        return self.v_max * ( 1 - np.exp( 1 - np.exp( self.c_jam / self.v_max * (
            1 - self.rho_max / np.maximum(1e-6, rho)
            ) ) ) )
    
    def density_from_velocity(self, v):
        return self.rho_max / (1 - self.v_max / self.c_jam * np.log(
            np.maximum(1e-6, 1 - np.log(np.maximum(1e-6, 1 - v / self.v_max)) ) 
            ))

    def __call__(self, rho):
        return rho * self.velocity_from_density(rho)
    
    def dv_drho(self, rho):
        v_max = self.v_max
        rho_max = self.rho_max
        c_jam = self.c_jam

        dV_drho = (c_jam * rho_max / np.maximum(1e-6, rho**2)) * np.exp(1 - np.exp(c_jam / v_max * (1 - rho_max / rho)) + c_jam / v_max * (1 - rho_max / rho))
        return dV_drho

@dataclass
class CongestedFlow():
    rho_max: float
    v_max: float
    rho_c: float = field(init=False)

    def __post_init__(self) -> None:
        self.rho_c = self.rho_max / 2

    def density_from_velocity(self, v: np.ndarray) -> np.ndarray:
        """Compute density from velocity."""
        v_c = self.v_max/2
        free_flow = 1/4 * self.rho_max * self.v_max / np.maximum(v,v_c) * (v > v_c)
        congestion = self.rho_max * (1 - v / self.v_max) * (v <= v_c)
        return free_flow + congestion

    def velocity_from_density(self, rho: np.ndarray) -> np.ndarray:
        """Compute velocity from density."""
        free_flow = 1/4 * self.v_max * self.rho_max / np.maximum(rho, self.rho_max/4) * (rho < self.rho_max/2)
        congestion = self.v_max * (1 - rho / self.rho_max) * (rho >= self.rho_max/2)
        return free_flow + congestion
    
    def __call__(self, rho: np.ndarray) -> np.ndarray:
        free_flow = self.v_max * self.rho_max / 4 * (rho < self.rho_max/2)
        congestion = self.v_max * rho * (1 - rho / self.rho_max) * (rho >= self.rho_max/2)
        return free_flow + congestion


@dataclass
class PiecewiseQuadraticFlow:
    """
    A flow function that is quadratic for low densities and quadratic for high densities.
    """
    v_max: float
    rho_max: float
    rho_c: float
    Q_max: float
    v_c: float = field(init=False)
    n_params: int = 3

    def __post_init__(self) -> None:
        self.v_c = self.Q_max / self.rho_c
        # assert self.v_c < self.v_max

    def __call__[T: (NDArray, float)](self, rho: T) -> T:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        quad_1 = self.v_max * rho * (1 - rho / rho_1) 
        quad_2 = (self.Q_max * (rho - self.rho_max)/(self.rho_c - self.rho_max)
            * (2*self.rho_c - self.rho_max - rho) / (self.rho_c - self.rho_max))
        return quad_1 * (rho < self.rho_c) + quad_2 * (rho >= self.rho_c)

    def density_from_velocity[T:(NDArray,float)](self, v: T) -> T:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = rho_1 * (1 - v / self.v_max) * (v > self.v_c)
        # not the true inverse but still fine for our purposes, and easier to compute
        congestion = self.rho_max / (1 + v / self.v_max * self.rho_max/self.rho_c) * (v <= self.v_c)
        return free_flow + congestion

    def velocity_from_density[T:(NDArray,float)](self, rho: T) -> T:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = self.v_max * (1 - rho / rho_1) * (rho < self.rho_c)
        # congestion = self.rho_c * self.v_max * (1/rho - 1/self.rho_max) * (rho >= self.rho_c)
        congestion = self(rho) * np.divide(1, rho, where=rho>=self.rho_c) * (rho >= self.rho_c)
        return free_flow + congestion

    def dv_drho(self, rho: np.ndarray) -> np.ndarray:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = -self.v_max / rho_1 
        congestion = self.Q_max * (
            2 * self.rho_max * self.rho_c - self.rho_max**2 - rho**2
            ) / (self.rho_c - self.rho_max)**2 / np.where(rho >= self.rho_c, rho**2, 1.0)  # avoid division by zero
        
        return free_flow * (rho < self.rho_c) + congestion * (rho >= self.rho_c)


FlowFunction = GreenshieldsFlow | TriangularFlow | ThreeParameterFlow | QuadraticLinearFlow | ZeroFlow | PowerLawFlux | DelCastilloBenitezFlux

@dataclass
class ARZFlow():
    """
    ARZ model with equilibrium-based hesitation, i.e. h = v_eq(0) - v_eq(rho)
    """
    flux_function: GreenshieldsFlow | TriangularFlow | DelCastilloBenitezFlux | PowerLawFlux | CongestedFlow | PiecewiseQuadraticFlow | QuadraticLinearFlow
    tau: float = 5.0

    @property
    def rho_max(self):
        return self.flux_function.rho_max

    @property
    def v_max(self):
        return self.flux_function.v_max

    def v(self, rho, q):
        return q / np.maximum(rho, 1e-6) - self.h(rho)

    def h(self, rho):
        return self.v_eq(np.zeros_like(rho)) - self.v_eq(rho)
    
    def v_eq(self, rho):
        return self.flux_function.velocity_from_density(rho)

    def rho_eq(self, v):
        return self.flux_function.density_from_velocity(v)
    
    def q(self, rho, v):
        return rho * (v + self.h(rho))
    
    def rho_dh(self, rho):
        return - rho * self.flux_function.dv_drho(rho)
    
    def __call__(self, U: NDArray) -> NDArray:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        f_rho = q - rho * self.h(rho)
        f_q = q ** 2 / np.maximum(rho, 1e-6) - q * self.h(rho)
        return np.stack([f_rho, f_q], axis=-2)
    
    def source_term(self, U):
        rho = U[..., 0, :]
        q = U[..., 1, :]
        return np.stack([
            np.zeros_like(rho),
            (rho * self.v_eq(rho) + rho * self.h(rho) - q) / self.tau,
        ], axis=-2)
    
    def lambdas(self, U):
        rho = U[..., 0, :]
        q = U[..., 1, :]
        v = self.v(rho, q)
        rho_dh = self.rho_dh(rho)
        lambda1 = v - rho_dh
        lambda2 = v
        return np.stack([lambda1, lambda2], axis=-2)


@dataclass
class ARZFlow_modified:
    """Modified ARZ model from David Ketcheson"""
    v_max: float = 20.0
    rho_max: float = 1.0
    tau: float = 5.0
    gamma: float = 0.1

    def h(self, rho):
        rho = np.clip(rho/self.rho_max, 1e-6, 1-1e-6)
        gamma = self.gamma
        return 25. * rho**(gamma*2)/np.maximum(1-rho,1e-6)**gamma
        # return v_max - v_eq(rho)
    
    def rho_dh(self, rho):
        rho = np.clip(rho/self.rho_max, 1e-6, 1-1e-6)
        gamma = self.gamma
        return 25*gamma*(2-rho)*rho**(2*gamma)/np.maximum((1-rho),1e-6)**(1+gamma)
        # return v_max / rho_max

    def v_eq(self, rho):
        rho = np.clip(rho/self.rho_max, 1e-6, 1-1e-6)
        g = lambda y: np.sqrt(1. + (10.*(y-1./3))**2)
        v = 1.4976*(g(0.) + (g(1.)-g(0.))*rho - g(rho))/rho
        return self.v_max*v/20
        # return v_max * (1 - rho / rho_max)
