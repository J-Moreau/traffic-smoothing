from dataclasses import asdict
from itertools import product

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from traffic_models.data.mobile_century_mapped import mobile_century_ramp_indexes
from traffic_models.experiment.probe_data import prepare_probe_experiment_data
from traffic_models.experiment.probe_experiment import (
    DatasetConfig,
    FourDVarQuickConfig,
    KalmanConfig,
    ProbeExperimentConfig,
)
from traffic_models.experiment.utils import log_run
from traffic_models.extended_kalman import (
    KalmanResult,
    run_ekf_on_trajectories,
    run_rts_smoother,
)
from traffic_models.flows import (
    ARZFlow,
    GreenshieldsFlow,
    PiecewiseQuadraticFlow,
    PowerLawFlux,
    QuadraticLinearFlow,
    TriangularFlow,
    ZeroFlow,
)
from traffic_models.flows_default import (
    greenshields_mobile_century_herrera_bayen_2010,
    quadraticlinear_mobile_century_herrera_bayen_2010,
    triangular_mobile_century_herrera_bayen_2010,
)
from traffic_models.metrics import trajectory_mse
from traffic_models.nn.asm import AdaptiveSmoothing
from traffic_models.pinn import PINNConfig, train_pinn
from traffic_models.sim import RampConfig
from traffic_models.var_optim import FourDVarConfig, windowed_w4DVAR

# Use a separate project from PINN experiments
WANDB_PROJECT_NAME = "itsc2026"

def run_experiment(conf: ProbeExperimentConfig, run_baselines: bool = False, run_pinn: bool = False) -> None:
    grid, discretized_trajectories, full_trajectories, fields = prepare_probe_experiment_data(conf)
    measured_v = np.zeros((grid.n_timesteps, grid.n_cells)) * np.nan
    measured_v[
        discretized_trajectories["t_index"], discretized_trajectories["x_index"]
    ] = discretized_trajectories["velocity"]

    INIT_MODEL_VARIANCE = np.nanvar(fields.speed)
    VELOCITY_MEASURE_VARIANCE = np.nanvar(measured_v - fields.speed).item()

    grid, discretized_trajectories, full_trajectories, fields = prepare_probe_experiment_data(conf)
    measured_v = np.zeros((grid.n_timesteps, grid.n_cells)) * np.nan
    measured_v[discretized_trajectories["t_index"], discretized_trajectories["x_index"]] = (
        discretized_trajectories["velocity"]
    )

    if conf.data.name=="mobile-century" and conf.data.ramps:
        on_ramps_index, off_ramps_index = mobile_century_ramp_indexes(grid, conf.data.xmin_meters)
    else:
        on_ramps_index, off_ramps_index = np.empty(0,dtype=int), np.empty(0,dtype=int)
    ramp_config = RampConfig(
        on_ramps_index=on_ramps_index,
        off_ramps_index=off_ramps_index,
    )

    def compute_metrics(matrix: np.ndarray) -> dict[str, float]:
        hidden_trajectories = full_trajectories.filter(
            ~pl.col("vehicle_id").is_in(discretized_trajectories["vehicle_id"])
        )
        hidden_mse = trajectory_mse(hidden_trajectories, matrix)
        mse = trajectory_mse(full_trajectories, matrix)
        grid_mse = ((matrix - fields.speed) ** 2).mean()
        return {"rmse": float(mse**0.5 * 3.6), "grid_rmse": float(grid_mse**0.5 * 3.6), "hidden_rmse": float(hidden_mse**0.5 * 3.6)}

    def asm_experiment(isotropic=False):
        sigma_x_meters = 200.0 if conf.data.name=="mobile-century" else 100.0
        sigma_t_seconds = 60.0 if conf.data.name=="mobile-century" else 5.0

        kernel_space_window = 100*sigma_x_meters
        kernel_time_window = 100*sigma_t_seconds
        asm_model = AdaptiveSmoothing(
            kernel_time_window=kernel_time_window,
            kernel_space_window=kernel_space_window,
            dx=grid.dx_meters,
            dt=grid.dt_seconds,
            init_delta=sigma_x_meters,
            init_tau=sigma_t_seconds,
            init_c_cong=-15.0/3.6 if not isotropic else 1.0e6, 
            init_c_free=70.0/3.6 if not isotropic else 1.0e6, 
        )
        asm_model.eval()
        measured_v_torch = torch.from_numpy(measured_v).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        asm_pred_torch = asm_model(measured_v_torch)
        v_asm = asm_pred_torch.detach().numpy().squeeze(0)
        metrics = compute_metrics(v_asm)
        log_conf = dict(
            model="ASM" if not isotropic else "isotropic-smoothing", **OmegaConf.to_container(conf, resolve=True)
        )  # type: ignore
        log_run(log_conf, metrics=metrics, name=WANDB_PROJECT_NAME)
        return v_asm
    
    def w4dvar_experiment():
        flow = dict(
            LWRGreenshields = greenshields_mobile_century_herrera_bayen_2010(),
            LWRTriangular = triangular_mobile_century_herrera_bayen_2010(),
            # LWRPiecewiseQuadratic = PiecewiseQuadraticFlow(v_max=40, rho_max=0.22, rho_c=0.05, Q_max=1.3), # not invertible
            ARZGreenshields = ARZFlow(PowerLawFlux(v_max=30, rho_max=0.15, gamma=1.0)), # gamma is fixed to 1.0 so this is Greenshields 
            ARZPiecewiseQuadratic= ARZFlow(PiecewiseQuadraticFlow(v_max=40, rho_max=0.22, rho_c=0.05, Q_max=1.3)),
            ARZQuadraticLinear = ARZFlow(QuadraticLinearFlow(v_max=40, rho_max=0.15, rho_c=0.05)),
        )[conf.fourdvar.fundamental_diagram]

        result = windowed_w4DVAR(
            trajectories=discretized_trajectories,
            v_0 = np.ones(grid.n_cells)*np.nanmean(fields.speed),
            conf=FourDVarConfig(
                background_variance=INIT_MODEL_VARIANCE, # type: ignore
                model_variance=conf.fourdvar.velocity_model_variance,
                measurement_variance=VELOCITY_MEASURE_VARIANCE,
                n_iters=conf.fourdvar.n_iters,
                solver=conf.fourdvar.solver, # type: ignore
            ),
            grid=grid,
            flow=flow,
            window_seconds=grid.n_timesteps*grid.dt_seconds/conf.fourdvar.n_windows,
            # window_seconds=t_pred*grid.dt_seconds,
            ramp_config=ramp_config,
            learn_flow=conf.fourdvar.learn_flow,
            device="cuda" if torch.cuda.is_available() else "cpu",
            init=conf.fourdvar.init, # type: ignore
            forecast=False,
        )
        v_smooth_4dvar, v_pred_4dvar, history, torch_flow, rho_hat = result.velocity_hat, result.velocity_pred, result.history, result.flow, result.rho_hat
        metrics = compute_metrics(v_smooth_4dvar)
        log_conf = dict(
            model="4DVar", **OmegaConf.to_container(conf, resolve=True), flow=asdict(flow)
        )  # type: ignore
        log_run(log_conf, metrics=metrics, name=WANDB_PROJECT_NAME)
        return
    
    def ekf_experiment(flow, model) -> KalmanResult:
        pred = run_ekf_on_trajectories(
            discretized_trajectories=discretized_trajectories,
            P_0=INIT_MODEL_VARIANCE * np.eye(grid.n_cells),
            v_0=fields.speed.mean(),
            flow=flow,
            Q=conf.kalman.velocity_model_variance * np.eye(grid.n_cells),
            VELOCITY_MEASURE_VARIANCE=VELOCITY_MEASURE_VARIANCE,
            grid=grid,
        ) 
        metrics = compute_metrics(pred.v_hat)
        log_conf = dict(
            model=model, **OmegaConf.to_container(conf, resolve=True), flow=asdict(flow)
        )  # type: ignore
        log_run(log_conf, metrics=metrics, name=WANDB_PROJECT_NAME)
        return pred
    
    def rts_experiment(ekf_pred, flow) -> None:
        v_smooth, P_smooth = run_rts_smoother(
            ekf_pred,
            Q=conf.kalman.velocity_model_variance * np.eye(grid.n_cells),
            flow=flow,
        )
        metrics = compute_metrics(v_smooth)
        log_conf = dict(
            model="LWR-RTS", **OmegaConf.to_container(conf, resolve=True), flow=asdict(flow)
        )  # type: ignore
        log_run(log_conf, metrics=metrics, name=WANDB_PROJECT_NAME)
    
    def pinn_experiment(physics_weight=100.0):
        pinn_flow = dict(
            LWRGreenshields = greenshields_mobile_century_herrera_bayen_2010(),
            LWRTriangular = triangular_mobile_century_herrera_bayen_2010(),
            # LWRPiecewiseQuadratic = PiecewiseQuadraticFlow(v_max=40, rho_max=0.22, rho_c=0.05, Q_max=1.3), # not invertible
            ARZGreenshields = ARZFlow(PowerLawFlux(v_max=30, rho_max=0.15, gamma=1.0)), # gamma is fixed to 1.0 so this is Greenshields 
            ARZPiecewiseQuadratic= ARZFlow(PiecewiseQuadraticFlow(v_max=40, rho_max=0.22, rho_c=0.05, Q_max=1.3)),
            ARZQuadraticLinear = ARZFlow(QuadraticLinearFlow(v_max=40, rho_max=0.15, rho_c=0.05)),
        )[conf.fourdvar.fundamental_diagram]

        pinn_result = train_pinn(
            trajectories=discretized_trajectories,
            grid=grid,
            flow=pinn_flow,
            conf=PINNConfig(
                n_collocation=150_000,
                epochs=50_500,
                n_epochs_adam=50_000,
                lr=1e-3,
                observation_weight=1.0,
                physics_weight=physics_weight, # type: ignore
                log_every=500,
                learn_flow=False,
            ),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        metrics = compute_metrics(pinn_result.velocity_hat)
        print(f"PINN RMSE: {metrics['rmse']:.2f} kmph")
        log_conf = dict(
            model="PINN" if physics_weight > 0 else "PUNN", **OmegaConf.to_container(conf, resolve=True), flow=asdict(pinn_flow)
        )  # type: ignore
        log_run(log_conf, metrics=metrics, name=WANDB_PROJECT_NAME)
    
    if run_pinn:
        pinn_experiment()
        # pinn_experiment(0.0) # ablation without physics loss
        return
    # main experiment
    w4dvar_experiment()

    # pred_trajectories = full_trajectories.with_columns(pl.col("t_index") - t_past).filter(pl.col("t_index") >= 0)
    # trajectory_mse(pred_trajectories, v_pred_4dvar[t_past:]), trajectory_mse(pred_trajectories, asm_pred)
    # assert asm_pred.shape == v_pred_4dvar[t_past:].shape
    if not run_baselines:
        return

    asm_experiment()
    asm_experiment(isotropic=True)
    
    
    # Skip the baselines
    # # prediction on the next x seconds given the past y seconds
    # t_past = 900//grid.dt_seconds
    # t_pred = 900//grid.dt_seconds
    # T = t_past + t_pred
    # N = measured_v.shape[0]
    # n_windows = int(np.ceil((N-t_past) / t_pred))

    # asm_preds = []
    # for i in range(n_windows):
    #     v_window = measured_v[i*t_pred:((i+1)*t_pred+t_past), :].copy()
    #     v_window[t_past:, :] = np.nan  # Mask the future measurements
    #     windowed_measured_masked = torch.tensor(v_window)
    #     asm_preds.append(asm_model(windowed_measured_masked.unsqueeze(0).unsqueeze(0)).detach().numpy()[0, t_past:, :])
    # asm_pred = np.concatenate(asm_preds, axis=0)

    flow = ZeroFlow() # this naive baseline works better for mobile century
    ekf_experiment(flow, "Identity-EKF")
    
    flow = triangular_mobile_century_herrera_bayen_2010() # this one is better when using RTS
    ekf_pred = ekf_experiment(flow, "LWR-EKF")

    # re-use ekf predictions for RTS
    rts_experiment(ekf_pred, flow)


if __name__ == "__main__":
    
    # conf = ProbeExperimentConfig(
    #     dt_seconds=1,
    #     dx_meters=30,
    #     data = DatasetConfig(
    #         name="us-101",
    #         start_seconds=180,  # start 3 minutes after the beginning
    #         end_seconds=2700,
    #         probe_fraction=0.02,
    #         boundary_fraction=0.0,
    #         smoothing=False
    #     )
    # # I80 is cut into three 15-minute periods: 4:00 p.m. to 4:15 p.m.; 5:00 p.m. to 5:15 p.m.; and 5:15 p.m. to 5:30 p.m
    # #     data=DatasetConfig(
    # #         name = "i-80",
    # #         start_seconds = 3780, #5:03 PM
    # #         end_seconds = 5400, #5:30 PM
    # #         boundary_fraction=0.0,
    # #         probe_fraction=0.1,
    # #         smoothing=False
    # # )
    # )

    mobile_century_conf = ProbeExperimentConfig(
        dt_seconds=5,
        dx_meters=200,
        seed=0,
        data=DatasetConfig(
            name="mobile-century",
            path="data/mobilecentury/NB_veh_files",
            probe_fraction=0.1,
            boundary_fraction=1.0,
            start_seconds=1_600, end_seconds=32_000, xmin_meters=34_000, xmax_meters=43_000,
            # start_seconds=18_000, end_seconds=24_000, xmin_meters=34_000, xmax_meters=43_000, # congestion only period
            smoothing=True,
        ),
        kalman=KalmanConfig(velocity_model_variance=0.1, desroziers=False),
        fourdvar=FourDVarQuickConfig(velocity_model_variance=0.01, n_iters=500, solver="rusanov", n_windows=1, init="naive_smoothing", learn_flow=True),
    )


    def run_sweep(sweep_name: str, basic_conf: ProbeExperimentConfig, sweep: dict, run_baselines: bool = True, run_pinn: bool = False) -> None:
        # Generate all combinations (grid search)
        keys, values = zip(*sweep.items())
        for combo in tqdm(product(*values)):
            conf = OmegaConf.structured(basic_conf)
            conf["name"] = sweep_name
            for k, v in zip(keys, combo):
                OmegaConf.update(conf, k, v)
            print(OmegaConf.to_container(conf, resolve=True))
            try:
                # use omegaconf instead of dataclass to allow nested updates
                run_experiment(conf, run_baselines=run_baselines, run_pinn=run_pinn)
            except Exception as e:
                print(f"Error with config {conf}: {e}")
    
    # sweep_name = "itsc_v4"
    # sweep = {
    #     "fourdvar.fundamental_diagram": ["ARZPiecewiseQuadratic"],
    #     "fourdvar.solver": ["rusanov"],
    #     "data.probe_fraction": [0.05, 0.1, 0.2, 0.3], 
    #     "data.boundary_fraction": [1.0],
    #     "fourdvar.init": ["naive_smoothing"],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep, run_baselines=True) 

    # # ABLATION SWEEP
    sweep_name = "itsc_ablation_v3"
    sweep_init = {
        "fourdvar.fundamental_diagram": ["ARZPiecewiseQuadratic"],
        "fourdvar.solver": ["rusanov"],
        "data.probe_fraction": [0.1], 
        "data.boundary_fraction": [1.0],
        "fourdvar.init": ["rts", "forward_fill"],
        "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    run_sweep(sweep_name, mobile_century_conf, sweep_init, run_baselines=False) 

    sweep_arz_fd = {
        "fourdvar.fundamental_diagram": ["ARZQuadraticLinear"],
        "fourdvar.solver": ["rusanov"],
        "data.probe_fraction": [0.1],
        "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    run_sweep(sweep_name, mobile_century_conf, sweep_arz_fd, run_baselines=False) 
    # sweep_solver = {
    #     "fourdvar.fundamental_diagram": ["LWRPiecewiseQuadratic"],
    #     "fourdvar.solver": ["rusanov"],
    #     "data.probe_fraction": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_solver, run_baselines=False) 
    # sweep_solver = {
    #     "fourdvar.fundamental_diagram": ["LWRTriangular"],
    #     "fourdvar.solver": ["godunov"],
    #     "fourdvar.learn_flow": [False], # otherwise it fails
    #     "fourdvar.velocity_model_variance": [0.01],
    #     "data.probe_fraction": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_solver, run_baselines=False) 

    # sweep_arz_fd = {
    #     "fourdvar.fundamental_diagram": ["ARZGreenshields"],
    #     "fourdvar.solver": ["rusanov"],
    #     "fourdvar.velocity_model_variance": [0.1],
    #     "fourdvar.learn_flow": [True],
    #     "fourdvar.n_windows": [3],
    #     "data.probe_fraction": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_arz_fd, run_baselines=False)

    # sweep_ramp = {
    #     "fourdvar.fundamental_diagram": ["ARZPiecewiseQuadratic"],
    #     "fourdvar.solver": ["rusanov"],
    #     "data.ramps": [False],
    #     "data.probe_fraction": [0.1],
    #     "fourdvar.velocity_model_variance": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_ramp, run_baselines=False) 

    # sweep_solver = {
    #     "fourdvar.fundamental_diagram": ["LWRGreenshields"],
    #     "fourdvar.solver": ["godunov"],
    #     "data.probe_fraction": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_solver, run_baselines=False) 

    # # PINN sweep
    # sweep_name = "itsc_ablation_v3"
    # sweep_pinn = {
    #     "fourdvar.fundamental_diagram": ["ARZGreenshields","ARZPiecewiseQuadratic"],
    #     "data.probe_fraction": [0.1],
    #     "seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # }
    # run_sweep(sweep_name, mobile_century_conf, sweep_pinn, run_baselines=False, run_pinn=True)