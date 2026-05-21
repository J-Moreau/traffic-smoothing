import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

import wandb
from traffic_models.experiment.probe_data import prepare_probe_experiment_data
from traffic_models.experiment.probe_experiment import (
    DatasetConfig,
    FourDVarQuickConfig,
    KalmanConfig,
    ProbeExperimentConfig,
)
from traffic_models.flows import ARZFlow, PiecewiseQuadraticFlow, PowerLawFlux
from traffic_models.metrics import trajectory_mse
from traffic_models.pinn import PINNConfig, PINNResult, train_pinn

WANDB_PROJECT_NAME = "itsc2026"


def build_mobile_century_conf(seed: int, fundamental_diagram: str) -> ProbeExperimentConfig:
    return ProbeExperimentConfig(
        name="itsc_ablation_v3",
        dt_seconds=5,
        dx_meters=200,
        seed=seed,
        data=DatasetConfig(
            name="mobile-century",
            path="data/mobilecentury/NB_veh_files",
            probe_fraction=0.1,
            boundary_fraction=1.0,
            start_seconds=1_600,
            end_seconds=32_000,
            xmin_meters=34_000,
            xmax_meters=43_000,
            smoothing=True,
        ),
        kalman=KalmanConfig(velocity_model_variance=0.1, desroziers=False),
        fourdvar=FourDVarQuickConfig(
            velocity_model_variance=0.01,
            n_iters=500,
            solver="rusanov",
            n_windows=1,
            init="naive_smoothing",
            learn_flow=True,
            fundamental_diagram=fundamental_diagram,
        ),
    )


def make_pinn_flow(flow_function: str):
    flow_name = flow_function.lower().replace("_", "").replace("-", "")
    if flow_name == "greenshields":
        return ARZFlow(PowerLawFlux(v_max=30, rho_max=0.15, gamma=1.0))
    if flow_name == "piecewisequadratic":
        return ARZFlow(PiecewiseQuadraticFlow(v_max=40, rho_max=0.22, rho_c=0.05, Q_max=1.3))
    raise ValueError("flow_function must be one of: greenshields, piecewisequadratic")


def compute_metrics(
    full_trajectories: pl.DataFrame,
    discretized_trajectories: pl.DataFrame,
    fields_speed: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, float]:
    hidden_trajectories = full_trajectories.filter(
        ~pl.col("vehicle_id").is_in(discretized_trajectories["vehicle_id"])
    )
    hidden_mse = trajectory_mse(hidden_trajectories, matrix)
    mse = trajectory_mse(full_trajectories, matrix)
    grid_mse = ((matrix - fields_speed) ** 2).mean()
    return {
        "rmse": float(mse ** 0.5 * 3.6),
        "grid_rmse": float(grid_mse ** 0.5 * 3.6),
        "hidden_rmse": float(hidden_mse ** 0.5 * 3.6),
    }


def log_final_artifacts(
    run: wandb.sdk.wandb_run.Run,
    result: PINNResult,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outputs_path = tmp_path / "final_outputs.npz"
        weights_path = tmp_path / "model_weights.pt"
        metadata_path = tmp_path / "metadata.json"

        np.savez_compressed(
            outputs_path,
            velocity_hat=result.velocity_hat,
            rho_hat=result.rho_hat,
        )
        torch.save(
            {
                "network_state_dict": result.network.state_dict(),
                "flow_state_dict": result.flow.state_dict(),
                "flow_class": result.flow.__class__.__name__,
            },
            weights_path,
        )

        artifact = wandb.Artifact(
            name="pinn_final_outputs",
            type="model-output",
        )
        artifact.add_file(str(outputs_path), name="final_outputs.npz")
        artifact.add_file(str(weights_path), name="model_weights.pt")
        artifact.add_file(str(metadata_path), name="metadata.json")
        run.log_artifact(artifact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the PINN experiment from the sweep pipeline.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data sampling and training.")
    parser.add_argument(
        "--flow-function",
        type=str,
        default="greenshields",
        choices=["greenshields", "piecewisequadratic"],
        help="Flow function used by the ARZ model.",
    )
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_NAME, help="W&B project name.")
    parser.add_argument("--epochs", type=int, default=50_500, help="Total PINN epochs.")
    parser.add_argument("--n-collocation", type=int, default=150_000, help="Number of collocation points.")
    parser.add_argument("--physics-weight", type=float, default=100.0, help="Physics loss weight.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fundamental_diagram = (
        "ARZGreenshields" if args.flow_function == "greenshields" else "ARZPiecewiseQuadratic"
    )
    conf = build_mobile_century_conf(seed=args.seed, fundamental_diagram=fundamental_diagram)
    flow = make_pinn_flow(args.flow_function)
    grid, discretized_trajectories, full_trajectories, fields = prepare_probe_experiment_data(conf)

    pinn_result = train_pinn(
        trajectories=discretized_trajectories,
        grid=grid,
        flow=flow,
        conf=PINNConfig(
            n_collocation=args.n_collocation,
            epochs=args.epochs,
            n_epochs_adam=min(50_000, args.epochs),
            lr=1e-3,
            observation_weight=1.0,
            physics_weight=args.physics_weight,
            log_every=500,
            learn_flow=False,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    def compute_metrics(matrix: np.ndarray) -> dict[str, float]:
        hidden_trajectories = full_trajectories.filter(
            ~pl.col("vehicle_id").is_in(discretized_trajectories["vehicle_id"])
        )
        hidden_mse = trajectory_mse(hidden_trajectories, matrix)
        mse = trajectory_mse(full_trajectories, matrix)
        grid_mse = ((matrix - fields.speed) ** 2).mean()
        return {"rmse": float(mse**0.5 * 3.6), "grid_rmse": float(grid_mse**0.5 * 3.6), "hidden_rmse": float(hidden_mse**0.5 * 3.6)}


    metrics = compute_metrics(
        matrix=pinn_result.velocity_hat,
    )

    run_name = f"pinn-{args.flow_function}-seed-{args.seed}"
    conf = OmegaConf.to_container(OmegaConf.structured(conf), resolve=True)
    conf = dict(
            model="PINN" if args.physics_weight > 0 else "PUNN", **conf, flow=asdict(flow)
        )
    with wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=conf,
    ) as run:
        run.log(metrics)
        run.summary.update(metrics)
        log_final_artifacts(run, pinn_result, metrics, conf, args.flow_function)

    print(f"PINN RMSE: {metrics['rmse']:.2f} km/h")


if __name__ == "__main__":
    main()