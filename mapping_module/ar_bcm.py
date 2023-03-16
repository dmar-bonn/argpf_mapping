import numpy as np
import time
import copy
from utils.utils import kernel_fcn, matrix_inverse
from mapping_module.ar_mapper import AR_Mapper, Node

import matplotlib.pyplot as plt


class AR_BCM(AR_Mapper):
    def __init__(self, cfg):
        super().__init__()
        self.max_depth = cfg["max_depth"]
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.f_prior = cfg["f_prior"]
        self.v_prior = cfg["v_prior"]
        self.l = cfg["l"]
        self.split_degree = cfg["split_degree"]
        self.record_metrics = cfg["record_metrics"]
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
        self.count = 1
        self.depth = 0
        self.root = None
        self.tree = None
        self.node_vector = []
        self.node_value = []
        self.node_center_list = None
        self.state_mask = None
        self.covariance = None
        self.covariance_prior = None
        self.covariance_inv_prior = None
        self.info_vec = 0
        self.info_mat = 0
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

        self.state_mask = np.ones(len(self.node_vector), dtype=bool)
        self.info_vec = np.zeros(len(self.node_vector))
        self.info_mat = np.zeros((len(self.node_vector), len(self.node_vector)))

    def init_v_map(self):
        self.covariance = kernel_fcn(
            self.node_center_list, self.node_center_list, self.v_prior, self.l
        )
        self.covariance_prior = copy.deepcopy(self.covariance)
        self.covariance_prior_inv = matrix_inverse(self.covariance_prior)

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

        # hard-coded for testing
        if self.count == 16:
            self.merge_node()
            self.update_hs_mask()

        self.count += 1

    def posterior_cal(self, measurement_dict):
        data_matrix = measurement_dict["data_matrix"]
        sensor_info = measurement_dict["sensor_info"]
        fov = sensor_info["fov"]
        altitude_noise = sensor_info["altitude_noise"]

        query_list = self.state_query(fov)
        m, Hm = self.process_measurement(query_list, data_matrix, sensor_info)
        f_post, cov_post = self.bcm(m, Hm, altitude_noise)
        return f_post, cov_post, Hm

    def state_query(self, fov):
        def tree_search(parent_node):
            for node in parent_node.children:
                check_result = self.within_FoV(node, fov)
                if check_result == 0:
                    continue
                elif check_result == 1:
                    query_list.extend(node.get_offspring)
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
        return np.array(m), np.array(Hm)

    # bayesian committee machine
    def bcm(self, m, Hm, sigma):
        K_Xs_X = self.covariance_prior[self.state_mask][:, Hm]
        K_Xs_Xs = self.covariance_prior[self.state_mask_2d]
        K_X_X = self.covariance_prior[np.ix_(Hm, Hm)]
        S = K_X_X + sigma**2 * np.eye(len(m))
        S_inv = matrix_inverse(S)  # matrix_inverse(S)

        f_post_i = np.dot(K_Xs_X, np.dot(S_inv, m))
        P_post_i = K_Xs_Xs - np.dot(K_Xs_X, np.dot(S_inv, np.transpose(K_Xs_X)))
        P_post_inv_i = matrix_inverse(P_post_i)  # matrix_inverse(P_post_i)

        self.info_vec[self.state_mask] += np.dot(P_post_inv_i, f_post_i)
        self.info_mat[self.state_mask_2d] += P_post_inv_i

        if self.count == 16:
            C = -(self.count - 1) * self.covariance_prior_inv + self.info_mat
            P_post = np.linalg.inv(C[self.state_mask_2d])  # matrix_inverse(C)
            f_post = np.dot(P_post, self.info_vec[self.state_mask])
            return f_post, P_post

        return f_post_i, P_post_i

    def merge_node(self):
        for row in reversed(self.tree[:-1]):
            for node in row:
                if node.ready_to_merge:
                    children_idx = [child.index for child in node.children]
                    self.state_mask[children_idx] = False
                    self.state_mask[node.index] = True
                    node.children = []

    @property
    def final_map(self):
        center_list = []
        size_list = []
        value_list = []
        for node in self.state_vector:
            if node.is_leaf:
                center_list.append(node.center)
                size_list.append([node.x_len, node.y_len])
                value_list.append(node.f)
        return [[center_list, size_list, value_list], self.get_effective_memory()]

    def get_effective_memory(self):
        node_idx = [node.index for node in self.state_vector if node.is_leaf]
        return [self.node_vector[node_idx], self.covariance[np.ix_(node_idx, node_idx)]]
