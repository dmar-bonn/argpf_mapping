import numpy as np
import matplotlib
import matplotlib.cm as cm


class AR_Mapper:
    def init_f_map(self):
        raise NotImplementedError("'init_f_map' method is not defined!")

    def init_v_map(self):
        raise NotImplementedError("'init_v_map' method is not defined!")

    def state_query(self):
        raise NotImplementedError("'state_query' method is not defined!")

    def update(self):
        raise NotImplementedError("'update' method is not defined!")

    def load_gt_info(self, gt_info):
        self.gt_X, self.gt_Y, self.gt_mask, self.hs_threshold, self.gt_gsd = gt_info
        self.update_hs_mask()

    def update_hs_mask(self):
        # self.hs_mask = np.array(self.node_value[self.state_mask] >= self.hs_threshold)
        self.hs_mask = (
            np.array([node.f for node in self.state_vector]) >= self.hs_threshold
        )

        self.hs_trace = np.sum(
            np.diagonal(self.covariance)[self.state_mask][self.hs_mask]
        )

    def forward_simulation(self, fake_measurement_dict):
        _, cov_post, _ = self.posterior_cal(fake_measurement_dict)
        hs_trace = np.sum(np.diagonal(cov_post)[self.hs_mask])
        return hs_trace

    def within_FoV(self, node, fov):
        node_range = node.get_grid_range
        xr = min(node_range[1], fov[1])
        xl = max(node_range[0], fov[0])
        yd = min(node_range[3], fov[3])
        yu = max(node_range[2], fov[2])
        dx = xr - xl
        dy = yd - yu

        # 1: fully covered; 2:partially covered; 0: no overlap
        if (dx > 0) and (dy > 0):
            if abs(dx - node.x_len) < 1e-4 and abs(dy - node.y_len) < 1e-4:
                return 1
            else:
                return 2
        else:
            return 0

    # for quadtree only TODO: generalize for ND tree
    def split_node(self, parent_node):
        center = parent_node.center
        x_len = parent_node.x_len
        y_len = parent_node.y_len

        child0 = Node(
            parent=parent_node,
            children=[],
            center=center + np.array([-x_len / 4, -y_len / 4]),
            x_len=x_len / 2,
            y_len=y_len / 2,
            state_value=self.f_prior,
            variance=self.v_prior,
            depth=parent_node.depth + 1,
            merging_th=self.merging_th,
        )
        child1 = Node(
            parent=parent_node,
            children=[],
            center=center + np.array([x_len / 4, -y_len / 4]),
            x_len=x_len / 2,
            y_len=y_len / 2,
            state_value=self.f_prior,
            variance=self.v_prior,
            depth=parent_node.depth + 1,
            merging_th=self.merging_th,
        )
        child2 = Node(
            parent=parent_node,
            children=[],
            center=center + np.array([x_len / 4, y_len / 4]),
            x_len=x_len / 2,
            y_len=y_len / 2,
            state_value=self.f_prior,
            variance=self.v_prior,
            depth=parent_node.depth + 1,
            merging_th=self.merging_th,
        )
        child3 = Node(
            parent=parent_node,
            children=[],
            center=center + np.array([-x_len / 4, y_len / 4]),
            x_len=x_len / 2,
            y_len=y_len / 2,
            state_value=self.f_prior,
            variance=self.v_prior,
            depth=parent_node.depth + 1,
            merging_th=self.merging_th,
        )

        return [child0, child1, child2, child3]

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

    def node_value_and_covariance_update(self, children_idx, parent_idx):
        self.node_value[parent_idx] = np.mean(self.node_value[children_idx])
        children_cov = self.covariance[children_idx]
        parent_cov = np.mean(children_cov, axis=0)
        self.covariance[parent_idx] = parent_cov
        self.covariance[:, parent_idx] = parent_cov
        self.covariance[parent_idx, parent_idx] = np.mean(children_cov[:, children_idx])

    # calculate true RMSE and recall compared with hotspot in ground truth
    def evaluate_metrics(self):
        gt_X_hs = self.gt_X[self.gt_mask]
        gt_Y_hs = self.gt_Y[self.gt_mask]
        area_gt_pixel = self.gt_gsd[0] * self.gt_gsd[1]

        err_square = []
        hs_point_num = len(gt_X_hs)
        union_map = 0
        recall_point_num = 0

        # for node in self.state_vector:
        #     f = node.f
        #     if f > self.hs_threshold and self.exploration_mask[node.index]:
        #         union_map += node.x_len * node.y_len

        #     for i, x in enumerate(gt_X_hs):
        #         if (
        #             abs(x[0] - node.center[0]) < 0.5 * node.x_len
        #             and abs(x[1] - node.center[1]) < 0.5 * node.y_len
        #         ):
        #             err_square.append((f - gt_Y_hs[i]) ** 2)
        #             if f > self.hs_threshold and self.exploration_mask[node.index]:
        #                 recall_point_num += 1
        for node in self.state_vector:
            if self.exploration_mask[node.index]:
                f = node.f
                if f > self.hs_threshold:
                    union_map += node.x_len * node.y_len

                for i, x in enumerate(gt_X_hs):
                    if (
                        abs(x[0] - node.center[0]) < 0.5 * node.x_len
                        and abs(x[1] - node.center[1]) < 0.5 * node.y_len
                    ):
                        err_square.append((f - gt_Y_hs[i]) ** 2)
                        if f > self.hs_threshold and self.exploration_mask[node.index]:
                            recall_point_num += 1

        hs_recall = recall_point_num / hs_point_num
        iou = (recall_point_num * area_gt_pixel) / (
            (hs_point_num - recall_point_num) * area_gt_pixel + union_map
        )

        return np.sqrt(np.mean(err_square)), hs_recall, iou

    # record all the data needed for reconstructing the map
    def get_effective_memory(self):
        return [
            self.state_vector,
            self.covariance[self.state_mask_2d],
        ]

    def get_cell_number(self):
        return len(self.state_vector)

    @property
    def state_vector(self):
        return self.node_vector[self.state_mask]

    @property
    def state_mask_2d(self):
        return np.ix_(self.state_mask, self.state_mask)

    @property
    def final_map(self):
        center_list = self.node_center_list[self.state_mask]
        size_list = [[node.x_len, node.y_len] for node in self.state_vector]
        value_list = self.node_value[self.state_mask]
        return [[center_list, size_list, value_list], self.get_effective_memory()]

    def visualizer(self, ax):
        ax.clear()
        color_map = cm.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(0, 1)

        for node in self.state_vector:
            if node.is_leaf:
                x1, x2, y1, y2 = node.get_grid_range
                color_f = color_map(norm(node.f))
                p = matplotlib.patches.Rectangle(
                    (x1, y2),
                    x2 - x1,
                    y1 - y2,
                    facecolor=color_f,
                    edgecolor="black",
                    lw=0.001,
                    alpha=1,
                )
                ax.add_patch(p)

        ax.set_xlabel("X [m]", fontsize=14)
        ax.set_ylabel("Y [m]", fontsize=14)
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
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)
        ax.axis("scaled")

        return ax


class Node:
    def __init__(
        self,
        parent,
        children,
        center,
        x_len,
        y_len,
        state_value,
        variance,
        depth,
        merging_th,
    ):
        self.parent = parent
        self.children = children
        self.f = state_value
        self.v = variance
        self.center = center
        self.x_len = x_len
        self.y_len = y_len
        self.depth = depth
        self.merging_th = merging_th
        self.is_explored = False
        self.index = None

    @property
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    @property
    def ready_to_merge(self):
        if not self.is_leaf:
            if np.all([child.is_leaf for child in self.children]):
                children_f_ub = np.array(
                    [child.f + np.sqrt(child.v) for child in self.children]
                )
                if np.all(children_f_ub < self.merging_th):
                    return True

    @property
    def get_grid_range(self):
        xmin = self.center[0] - 0.5 * self.x_len
        xmax = xmin + self.x_len
        ymin = self.center[1] - 0.5 * self.y_len
        ymax = ymin + self.y_len
        return [xmin, xmax, ymin, ymax]

    @property
    def get_leaf_nodes(self):
        leaf_list = []

        def _get_leaf_nodes(node):
            if node is not None:
                if len(node.children) == 0:
                    leaf_list.append(node.index)
                for n in node.children:
                    _get_leaf_nodes(n)

        _get_leaf_nodes(self)
        return leaf_list

    @property
    def get_offspring(self):
        offspring_list = [self.index]

        def _get_offspring(node):
            if node is not None:
                for n in node.children:
                    offspring_list.append(n.index)
                    _get_offspring(n)

        _get_offspring(self)
        return offspring_list
