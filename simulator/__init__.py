import numpy as np
import math
import csv
from simulator.fake_simulator import Fake_Simulator
import logging

logger = logging.getLogger(__name__)
WORLD_LIST = ["gaussian_random_field", "temperature_field"]


def get_simulator(simulator_cfg):
    if isinstance(simulator_cfg, dict):
        world_name = simulator_cfg["world_name"]

        if world_name in WORLD_LIST:
            logger.info(f"initialize {world_name} simulation world")
            if world_name == "gaussian_random_field":
                sim_world = generate_ground_truth(simulator_cfg[world_name])
            elif world_name == "temperature_field":
                sim_world = load_ground_truth(simulator_cfg[world_name])
            return Fake_Simulator(simulator_cfg[world_name], sim_world)
        else:
            raise RuntimeError(f"{world_name} is not implemented")

    else:
        raise RuntimeError(f"{type(simulator_cfg)} not a valid config file")


def load_ground_truth(cfg):
    file_path = cfg["path"]
    with open(file_path) as file:
        gt_matrix = csv.reader(file)

    # TODO: preprocess data set

    return (gt_matrix - np.min(gt_matrix)) / (np.max(gt_matrix) - np.min(gt_matrix))


def generate_ground_truth(cfg):
    x_range = cfg["x_range"]
    y_range = cfg["y_range"]
    alpha = cfg["alpha"]

    Pk = lambda k: k**alpha

    def fftIndgen(n):
        a = list(range(0, math.floor(n / 2) + 1))
        b = reversed(range(1, math.floor(n / 2)))
        b = [-i for i in b]
        return a + b

    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))

    noise = np.fft.fft2(np.random.normal(size=(y_range, x_range)))
    amplitude = np.zeros((y_range, x_range))

    for i, kx in enumerate(fftIndgen(x_range)):
        for j, ky in enumerate(fftIndgen(y_range)):
            amplitude[i, j] = Pk2(kx, ky)

    random_field = np.fft.ifft2(noise * amplitude).real
    return (random_field - np.min(random_field)) / (
        np.max(random_field) - np.min(random_field)
    )
