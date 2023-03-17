import numpy as np
import time
import copy
from utils.utils import integral_kernel_fcn, matrix_inverse
from mapping_module.ar_mapper import AR_Mapper, Node


class AR_GPR_IK(AR_Mapper):
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
        self.m_history = []
        self.Hm_history = []
        self.exploraton_mask = None
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

    def init_v_map(self):
        corner_list = np.array([node.get_grid_range for node in self.node_vector])
        self.covariance = integral_kernel_fcn(
            corner_list, corner_list, self.v_prior, self.l
        )
        self.covariance_prior = copy.deepcopy(self.covariance)

    def update_map(self, measurement_dict):
        time_s = time.time()
        f_post, cov_post, Hm = self.posterior_cal(measurement_dict)
        time_e = time.time()
        self.mapping_time += time_e - time_s

        self.node_value[self.state_mask] = f_post
        self.covariance[self.state_mask_2d] = cov_post
        for node in self.node_vector[Hm]:
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
        f_post, cov_post = self.gpr(m, Hm, altitude_noise)
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
            self.m_history.append(m_grid)
            self.Hm_history.append(n)

        return self.m_history, self.Hm_history

    def gpr(self, m, Hm, sigma):
        K_Xs_X = self.covariance_prior[:, Hm][self.state_mask]
        K_Xs_Xs = self.covariance_prior[self.state_mask_2d]
        K_X_X = self.covariance_prior[np.ix_(Hm, Hm)]
        S = K_X_X + sigma**2 * np.eye(len(m))
        S_inv = matrix_inverse(S)
        f_post = self.f_prior + np.dot(
            K_Xs_X, np.dot(S_inv, np.array(m) - self.f_prior)
        )
        P_post = K_Xs_Xs - np.dot(K_Xs_X, np.dot(S_inv, np.transpose(K_Xs_X)))

        return f_post, P_post

    def merge_node(self):
        for row in reversed(self.tree[:-1]):
            for node in row:
                if node.ready_to_merge:
                    children_idx = [child.index for child in node.children]
                    if np.all(self.exploration_mask[children_idx]):
                        self.state_mask[children_idx] = False
                        self.exploration_mask[children_idx] = False
                        self.state_mask[node.index] = True
                        self.exploration_mask[node.index] = True

                        self.node_value_and_covariance_update(children_idx, node.index)
                        node.children = []
                        node.f = self.node_value[node.index]
                        node.v = self.covariance[node.index, node.index]
                        m_idx = np.where(np.in1d(self.Hm_history, children_idx))[0]
                        if len(m_idx) != 0:
                            parent_m = np.mean(np.array(self.m_history)[m_idx])
                            for idx in sorted(m_idx, reverse=True):
                                del self.m_history[idx]
                                del self.Hm_history[idx]

                            self.m_history.append(parent_m)
                            self.Hm_history.append(node.index)
