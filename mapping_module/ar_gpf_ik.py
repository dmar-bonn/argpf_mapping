import numpy as np
import time
from utils.utils import integral_kernel_fcn, matrix_inverse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
from constants import EXP_DIR
from mapping_module.ar_mapper import AR_Mapper, Node


class AR_GPF_IK(AR_Mapper):
    def __init__(self, cfg):
        self.max_depth = cfg["max_depth"]
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.f_prior = cfg["f_prior"]
        self.v_prior = cfg["v_prior"]
        self.l = cfg["l"]
        self.split_degree = cfg["split_degree"]
        self.merging_th = cfg["merging_threshold"]

        # ground truth information
        self.gt_X = None
        self.gt_Y = None
        self.gt_mask = None
        self.hs_threshold = None
        self.hs_mask = None
        self.hs_trace = None
        self.gt_gsd = None

        # map setting
        self.childern_num = self.split_degree**2
        self.x_resolution = self.split_degree**self.max_depth
        self.y_resolution = self.split_degree**self.max_depth
        self.depth = 0
        self.root = None
        self.tree = None
        self.node_vector = []  # cell instance list
        self.node_value = []  # cell value list
        self.node_center_list = None
        self.state_mask = None  # mask for active cell, pruned cells are no more active
        self.covariance = None
        self.exploration_mask = None
        self.mapping_time = 0

        self.init_tree()

    def init_tree(self):
        self.root = Node(
            parent=None,
            children=[],
            center=0.5 * np.array([self.x_boundary, self.y_boundary]),
            x_len=self.x_boundary,
            y_len=self.y_boundary,
            state_value=self.f_prior,
            variance=self.v_prior,
            depth=0,
            merging_th=self.merging_th,
        )
        self.tree = [[self.root]]

        for i in range(self.max_depth):
            self.tree.append([])  # new layer in the tree
            for n in self.tree[i]:
                children_list = self.split_node(n)
                n.children = children_list
                self.tree[i + 1].extend(children_list)

        self.init_f_map()
        self.init_v_map()

    def init_f_map(self):
        idx = 0
        for row in self.tree:
            for node in row:
                node.index = idx
                self.node_vector.append(node)
                self.node_value.append(node.f)
                idx += 1

        self.node_vector = np.array(self.node_vector)
        self.node_center_list = np.array([node.center for node in self.node_vector])
        self.node_value = np.array(self.node_value)
        self.exploration_mask = np.zeros(len(self.node_vector), dtype=bool)

        # create state mask for indicating active cells
        self.state_mask = np.ones(len(self.node_vector), dtype=bool)
        leaf_num = len(self.tree[-1])
        self.state_mask[: idx - leaf_num] = False
        self.leaf_vector = self.node_vector[self.state_mask]

    def init_v_map(self):
        corner_list = np.array([node.get_grid_range for node in self.node_vector])
        self.covariance = integral_kernel_fcn(
            corner_list, corner_list, self.v_prior, self.l
        )

    def update_map(self, measurement_dict):
        time_s = time.time()
        f_post, cov_post, Hm = self.posterior_cal(measurement_dict)
        time_e = time.time()
        self.mapping_time += time_e - time_s

        self.node_value[self.state_mask] = f_post
        self.covariance[self.state_mask_2d] = cov_post

        for node in self.state_vector:
            node.f = self.node_value[node.index]
            node.v = self.covariance[node.index, node.index]

        self.exploration_mask[Hm] = True

        self.merge_node()
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

    def state_query(self, fov):
        def tree_search(parent_node):
            for node in parent_node.children:
                check_result = self.within_FoV(node, fov)
                if check_result == 0:
                    continue
                elif check_result == 1:
                    query_list.extend(node.get_leaf_nodes)
                    continue
                elif check_result == 2:
                    tree_search(node)

        query_list = []
        tree_search(self.root)
        return query_list

    def process_measurement(self, query_list, data_matrix, sensor_info):
        Hm = []
        m = []
        gsd_x = sensor_info["gsd_x"]
        gsd_y = sensor_info["gsd_y"]
        fov = sensor_info["fov"]
        y_max, x_max = data_matrix.shape

        for n in query_list:
            grid_fp = self.node_vector[n].get_grid_range
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
            Hm.append(n)

        return np.array(m), Hm

    def kalman_filter(self, m, Hm, sigma):
        P = self.covariance[np.ix_(Hm, Hm)]
        S = P + sigma**2 * np.eye(len(m))
        S = 0.5 * np.add(S, np.transpose(S))
        L = np.transpose(np.linalg.cholesky(S))
        L_inv = np.linalg.inv(L)
        Wc = np.dot(self.covariance[:, Hm][self.state_mask], L_inv)
        kalman_gain = np.dot(Wc, np.transpose(L_inv))
        f_pos = self.node_value[self.state_mask] + np.dot(
            kalman_gain, (m - self.node_value[Hm])
        )
        P_pos = self.covariance[self.state_mask_2d] - np.dot(Wc, np.transpose(Wc))

        return f_pos, P_pos

    def reconstruction(self, ax):
        state_vector = self.state_vector
        f_x = np.array([node.f for node in state_vector])
        x = np.array([node.get_grid_range for node in state_vector])

        # set reconstruction resolution
        xa1 = []
        xmin = 0
        ymin = -0.4
        for i in range(2500):
            if i % 50 == 0:
                ymin += 0.4
                xmin = 0
            xa1.append([xmin, xmin + 0.4, ymin, ymin + 0.4])
            xmin += 0.4

        k_x_x = integral_kernel_fcn(x, x, 5, 2.73)
        k_xa1_x = integral_kernel_fcn(xa1, x, 5, 2.73)
        k_x_x_inv = matrix_inverse(k_x_x)  # np.linalg.inv(k_x_x)

        f1 = 0.5 + np.dot(k_xa1_x, np.dot(k_x_x_inv, f_x - 0.5))
        f1 = np.reshape(f1, (50, 50))

        num_ticks = 5
        yticks = np.linspace(0, 50, num_ticks, dtype=int)
        yticklabels = np.linspace(0, 20, num_ticks, dtype=int)
        xticks = np.linspace(0, 50, num_ticks, dtype=int)
        xticklabels = np.linspace(0, 20, num_ticks, dtype=int)

        sb.heatmap(
            f1,
            ax=ax,
            cmap=cm.get_cmap("viridis"),
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            cbar=False,
        )

        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Y [m]", fontsize=14)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.yaxis.set_tick_params(rotation=0)
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
