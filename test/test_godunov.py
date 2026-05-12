import numpy as np

from traffic_models.godunov import godunov_step
from traffic_models.flows import GreenshieldsFlow


def test_godunov_works_on_batches():

    Q = GreenshieldsFlow(rho_max=0.2, v_max=33.33)

    # single example
    rho = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    rho_next = godunov_step(rho, Q, dt=1.0, dx=10.0)

    # batched examples
    rho_batch = np.array(
        [
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.15, 0.1, 0.05, 0.0],
        ]
    )
    rho_next_batch = godunov_step(rho_batch, Q, dt=1.0, dx=10.0)

    assert np.allclose(rho_next, rho_next_batch[0])
