import numpy as np
import time
from mapping_module.fr_mapper import FR_Mapper


class FR_IDP(FR_Mapper):
    def __init__(self, cfg):
        super().__init__()
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.x_resolution = cfg["x_resolution"]
        self.y_resolution = cfg["y_resolution"]
        self.f_prior = cfg["f_prior"]
        self.v_prior = cfg["v_prior"]
        self.record_metrics = cfg["record_metrics"]

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

    # covariance matrix with all off-diagonal elements 0: independence cells assumption
    def init_v_map(self):
        self.covariance = self.v_prior**2 * np.eye(len(self.state_vector))

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
        f_post, cov_post = self.idp_update(m, Hm, altitude_noise)
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

            m_grid = np.reshape(
                data_matrix[y_index_min:y_index_max, x_index_min:x_index_max], (-1)
            )
            m.append(m_grid)
            Hm.append(state_index)

        return m, Hm

    def idp_update(self, m, Hm, sigma):
        f = self.state_value.copy()
        cov = self.covariance.copy()

        # for i, index in enumerate(Hm):
        #     for m_single in m[i]:
        #         k = cov[index, index] / (cov[index, index] + sigma**2)
        #         f[index] += k * (m_single - f[index])
        #         cov[index, index] = (1 - k) * cov[index, index]
        for i, index in enumerate(Hm):
            m_single = np.mean(m[i])
            k = cov[index, index] / (cov[index, index] + sigma**2)
            f[index] += k * (m_single - f[index])
            cov[index, index] = (1 - k) * cov[index, index]
        return f, cov

    def get_effective_memory(self):
        return [
            np.diagonal(self.covariance),
            self.state_vector,
            self.state_value,
            self.x_gsd,
            self.y_gsd,
        ]
