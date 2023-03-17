import numpy as np
import time
from utils.utils import kernel_fcn
from mapping_module.fr_mapper import FR_Mapper


class FR_GPF(FR_Mapper):
    def __init__(self, cfg):
        super().__init__()
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.x_resolution = cfg["x_resolution"]
        self.y_resolution = cfg["y_resolution"]
        self.f_prior = cfg["f_prior"]
        self.v_prior = cfg["v_prior"]
        self.l = cfg["l"]

        # ground truth information
        self.gt_X = None
        self.gt_Y = None
        self.gt_mask = None
        self.hs_threshold = None
        self.hs_mask = None
        self.hs_trace = None
        self.gt_gsd = None

        # map setting
        self.x_gsd = self.x_boundary / self.x_resolution
        self.y_gsd = self.y_boundary / self.y_resolution
        self.state_vector = None  # cell center list
        self.state_value = None  # cell value list
        self.covariance = None
        self.exploration_mask = None
        self.mapping_time = 0

        self.init_f_map()
        self.init_v_map()

    def init_f_map(self):
        x = np.arange(0, self.x_boundary, self.x_gsd) + 0.5 * self.x_gsd
        y = np.arange(0, self.y_boundary, self.y_gsd) + 0.5 * self.y_gsd
        xx, yy = np.meshgrid(x, y)
        self.state_vector = np.c_[xx.ravel(), yy.ravel()]
        self.state_value = self.f_prior * np.ones(len(self.state_vector))
        self.exploration_mask = np.zeros(len(self.state_vector), dtype=bool)

    def init_v_map(self):
        self.covariance = kernel_fcn(
            self.state_vector, self.state_vector, self.v_prior, self.l
        )

    def update_map(self, measurement_dict):
        time_s = time.time()
        f_post, cov_post, Hm = self.posterior_cal(measurement_dict)
        time_e = time.time()
        self.mapping_time += time_e - time_s

        self.state_value = f_post
        self.covariance = cov_post

        self.exploration_mask[Hm] = True
        self.update_hs_mask()

    def posterior_cal(self, measurement_dict):
        data_matrix = measurement_dict["data_matrix"]
        sensor_info = measurement_dict["sensor_info"]
        fov = sensor_info["fov"]
        altitude_noise = sensor_info["altitude_noise"]

        query_list = self.state_query(fov)
        m, Hm = self.process_measurement(query_list, data_matrix, sensor_info)
        f_post, cov_post = self.kalman_filter(m, Hm, altitude_noise)
        return f_post, cov_post, Hm

    def process_measurement(self, query_list, data_matrix, sensor_info):
        Hm = []
        m = []
        gsd_x = sensor_info["gsd_x"]
        gsd_y = sensor_info["gsd_y"]
        fov = sensor_info["fov"]
        y_max, x_max = data_matrix.shape

        for n in query_list:
            state_index = n[0]
            grid_fp = n[1]
            x_index_min = max(
                0, np.floor((grid_fp[0] - fov[0] + 0.5 * gsd_x) / gsd_x).astype(int)
            )
            x_index_max = min(
                x_max, np.ceil((grid_fp[1] - fov[0] - 0.5 * gsd_x) / gsd_x).astype(int)
            )
            y_index_min = max(
                0, np.floor((grid_fp[2] - fov[2] + 0.5 * gsd_y) / gsd_y).astype(int)
            )
            y_index_max = min(
                y_max, np.ceil((grid_fp[3] - fov[2] - 0.5 * gsd_y) / gsd_y).astype(int)
            )

            m_grid = np.mean(
                data_matrix[y_index_min:y_index_max, x_index_min:x_index_max]
            )
            m.append(m_grid)
            Hm.append(state_index)

        return m, Hm

    def kalman_filter(self, m, Hm, sigma):
        P = self.covariance[np.ix_(Hm, Hm)]
        S = P + sigma**2 * np.eye(len(m))
        S = 0.5 * np.add(S, np.transpose(S))
        L = np.transpose(np.linalg.cholesky(S))
        L_inv = np.linalg.inv(L)
        Wc = np.dot(self.covariance[:, Hm], L_inv)
        kalman_gain = np.dot(Wc, np.transpose(L_inv))
        f_post = self.state_value + np.dot(
            kalman_gain, (np.array(m) - self.state_value[Hm])
        )
        P_post = self.covariance - np.dot(Wc, np.transpose(Wc))

        return f_post, P_post
