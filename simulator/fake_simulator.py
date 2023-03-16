import numpy as np
import matplotlib.cm as cm
import seaborn as sb
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Fake_Simulator:
    def __init__(self, cfg, sim_world):
        self.sim_world = sim_world
        self.gsd_x = cfg["gsd_x"]
        self.gsd_y = cfg["gsd_y"]
        self.x_range = cfg["x_range"]
        self.y_range = cfg["y_range"]
        self.hs_threshold = cfg["hs_threshold"]
        self.cmap = cfg["color_map"]
        self.agent = None
        self.x_boundary = self.gsd_x * self.x_range
        self.y_boundary = self.gsd_y * self.y_range

    def run(self, agent):
        self.agent = agent
        self.agent.mapper.load_gt_info(self.ground_truth_info)
        # self.agent.record_metrics_history()
        logger.info("start mission")

        while self.agent.budget > 0:
            self.agent.move()
            self.agent.update_map(self.take_measurements())

        logger.info("mission completed")
        self.agent = None

    def take_measurements(self):
        pixel_x, pixel_y, gsd_x, gsd_y, fov = self.agent.sensor.FoV
        altitude_noise = self.agent.sensor.altitude_noise
        x_num = self.agent.sensor.x_resolution
        y_num = self.agent.sensor.y_resolution

        measurement_data = {
            "data_matrix": np.zeros((y_num, x_num)),
            "center_matrix": np.zeros((y_num, x_num, 2)),
            "sensor_info": {
                "fov": fov,
                "gsd_x": gsd_x,
                "gsd_y": gsd_y,
                "altitude_noise": altitude_noise,
            },
        }

        # simulate measurment process
        for j in range(y_num):
            for i in range(x_num):
                point_center = [
                    0.5 * (pixel_x[i] + pixel_x[i + 1]),
                    0.5 * (pixel_y[j] + pixel_y[j + 1]),
                ]
                index_xmin = np.floor(pixel_x[i] / self.gsd_x).astype(int)
                index_xmax = np.ceil(pixel_x[i + 1] / self.gsd_x).astype(int)
                index_ymin = np.floor(pixel_y[j] / self.gsd_y).astype(int)
                index_ymax = np.ceil(pixel_y[j + 1] / self.gsd_y).astype(int)
                block = self.sim_world[index_ymin:index_ymax, index_xmin:index_xmax]
                v = np.mean(block) + np.random.normal(
                    0, altitude_noise
                )  # add Gaussian noise
                measurement_data["data_matrix"][j, i] = v
                measurement_data["center_matrix"][j, i] = point_center

        return measurement_data

    @property
    def ground_truth_info(self):
        x = np.arange(0, self.x_boundary, self.gsd_x) + 0.5 * self.gsd_x
        y = np.arange(0, self.y_boundary, self.gsd_y) + 0.5 * self.gsd_y
        xx, yy = np.meshgrid(x, y)
        gt_X = np.c_[xx.ravel(), yy.ravel()]
        gt_Y = self.sim_world.flatten()
        hotspot_mask = gt_Y > self.hs_threshold

        return gt_X, gt_Y, hotspot_mask, self.hs_threshold, [self.gsd_x, self.gsd_y]

    def visualizer(self, ax):
        ax.clear()
        color_map = cm.get_cmap(self.cmap)

        num_ticks = 5
        yticks = np.linspace(0, len(self.sim_world) - 1, num_ticks, dtype=int)
        yticklabels = np.linspace(0, self.y_boundary, num_ticks, dtype=int)
        xticks = np.linspace(0, len(self.sim_world[0]) - 1, num_ticks, dtype=int)
        xticklabels = np.linspace(0, self.x_boundary, num_ticks, dtype=int)

        sb.heatmap(
            self.sim_world,
            cmap=color_map,
            ax=ax,
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            cbar_kws=dict(use_gridspec=False, location="left"),
        )

        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Y [m]", fontsize=14)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.tick_params(
            labelsize=14,
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        ax.margins(x=0, y=0)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)
        ax.axis("scaled")

        return ax
