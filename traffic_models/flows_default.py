"""Fundamental diagram models with default parameters"""
from traffic_models.flows import GreenshieldsFlow, QuadraticLinearFlow, TriangularFlow


def greenshields_ngsim_trajectory_regression():
    # parameters obtained from Least Squares regression
    # on ngsim us101 trajectory data (not detector data)
    return GreenshieldsFlow(v_max=32.9, rho_max=0.0611)

def triangular_ngsim_trajectory_regression():
    # parameters obtained from Least Squares regression
    # on ngsim us101 trajectory data (not detector data)
    return TriangularFlow(rho_c=0.038, Q_c=0.493, rho_max=0.096)

def quadraticlinear_ngsim_trajectory_regression():
    # parameters obtained from Least Squares regression
    # on ngsim us101 trajectory data (not detector data)
    return QuadraticLinearFlow(rho_c=0.038, v_max=32.9, rho_max=0.096)

def triangular_ngsim_herrera_bayen_2010():
    # From Herrera and Bayen (2010)
    # Incorporation of Lagrangian measurements in freeway traffic state estimation
    # vmax = 68 mph = 30.4 mps = Q_c / rho_c
    # Q_c = 2040 vphpl = 0.56 vpspl
    # rho_c = 30vpmilepl = 0.0186 vpmpl
    # rho_max = 205 vpmilepl = 0.127 vpmpl
    # w = -11.7 mph = -Q_c/ (rho_max - rho_c) => rho_max = rho_c - Q_c/w = 0.048
    return TriangularFlow(rho_c=0.0186, Q_c=0.56, rho_max=0.127)

def triangular_mobile_century_herrera_bayen_2010():
    # Parameters for Mobile Century, (Herrera & Bayen, 2010)
    # qmax = 2275 vphpl, kj = 152 vpmpl, kc = 35 vpmpl, vf = 65 mph, and w = –19.4 mph.

    # Q_c = 2275 / 3600 = 0.63 vpspl, rho_c = 35 / 1609 = 0.022 vpmpl, rho_max = 152 / 1609 = 0.094 vpmpl
    return TriangularFlow(rho_c=0.022, Q_c=0.63, rho_max=0.094, rho_free_flow=0.018)


def quadraticlinear_mobile_century_herrera_bayen_2010():
    # Adapted from Triangular to Quadratic-Linear Flow
    # v_max = 65 miles/hour (* 1609/3600) => 29.05 meter/s
    return QuadraticLinearFlow(rho_c=0.022, v_max=29.05, rho_max=0.094)


def greenshields_mobile_century_herrera_bayen_2010():
    # Adapted from Triangular to Greenshields Flow
    return GreenshieldsFlow(v_max=29.05, rho_max=0.094)

# Fan and Seibold (2013) choose a quadratic flow with a fixed rho_max 
# and matching maximum speed

# flow = GreenshieldsFlow(v_max=30.1, rho_max=1/7.5)

# flow = GreenshieldsFlow(v_max=25, rho_max=0.0433) # better performance in practice
