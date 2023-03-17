import numpy as np
import time
from mapping_module.ar_mapper import AR_Mapper, Node


class AR_IDP(AR_Mapper):
    def __init__(self, cfg):
        self.max_depth = cfg["max_depth"]
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.f_prior = cfg["f_prior"]
        self.v_prior = cfg["v_prior"]
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

    # covariance matrix with all off-diagonal elements 0: independence cells assumption
    def init_v_map(self):
        self.covariance = self.v_prior**2 * np.eye(len(self.node_vector))

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
        f_post, cov_post = self.idp_update(m, Hm, altitude_noise)
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

            m_grid = np.reshape(
                data_matrix[y_index_min:y_index_max, x_index_min:x_index_max], (-1)
            )

            m.append(m_grid)
            Hm.append(n)

        return m, Hm

    def idp_update(self, m, Hm, sigma):
        f = self.node_value.copy()
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

        return f[self.state_mask], cov[self.state_mask_2d]

    def merge_node(self):
        for row in reversed(self.tree[:-1]):
            for node in row:
                if node.ready_to_merge:
                    children_idx = [child.index for child in node.children]
                    self.state_mask[children_idx] = False
                    self.state_mask[node.index] = True
                    node.children = []
                    self.node_value_and_covariance_update(children_idx, node.index)
                    node.f = self.node_value[node.index]
                    node.v = self.covariance[node.index, node.index]

    def node_value_and_covariance_update(self, children_idx, parent_idx):
        self.node_value[parent_idx] = np.mean(self.node_value[children_idx])
        self.covariance[parent_idx] = np.mean(
            self.covariance[children_idx, children_idx]
        )

    def get_effective_memory(self):
        return [
            self.state_vector,
            np.diagonal(self.covariance[self.state_mask_2d]),
        ]

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
