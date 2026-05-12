import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from traffic_models.dense_fields import FlowData
from traffic_models.extended_kalman import KalmanResult
from traffic_models.sim import DiscretizationGrid

def trajectory_mse(trajectories: pl.DataFrame, v_pred: NDArray):
    t_indexes = trajectories["t_index"]
    x_indexes = trajectories["x_index"]
    velocities = trajectories["velocity"].to_numpy()
    return np.mean((velocities - v_pred[t_indexes,x_indexes])**2) 

def compute_error_metrics(v_hat: NDArray, P_hat: NDArray|None, fields: FlowData, full_trajectories: pl.DataFrame) -> dict:
    if fields.sparse:
        mse = trajectory_mse(
            full_trajectories,
            v_hat,
        )
    else:
        mse = ((v_hat - fields.speed) ** 2).mean()
    metrics = dict(mse=mse)
    if P_hat is not None:
        log_lik = [
                multivariate_normal.logpdf(
                    x=fields.speed[i],
                    mean=v_hat[i],
                    cov=P_hat[i],
                )  # full likelihood
                for i in range(v_hat.shape[0])
            ]
        neg_log_lik = -np.mean(log_lik) 
        metrics["neg_log_lik"] = neg_log_lik
    return metrics