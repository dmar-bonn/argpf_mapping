import numpy as np
import matplotlib
import matplotlib.cm as cm


class FR_Mapper:
    def init_f_map(self):
        raise NotImplementedError("'init_f_map' method is not defined!")

    def init_v_map(self):
        raise NotImplementedError("'init_v_map' method is not defined!")

    def update(self):
        raise NotImplementedError("'update' method is not defined!")

    def load_gt_info(self, gt_info):
        self.gt_X, self.gt_Y, self.gt_mask, self.hs_threshold, self.gt_gsd = gt_info
        self.update_hs_mask()

    # hotspot definded by the posterior mean of the map
    def update_hs_mask(self):
        self.hs_mask = np.array(self.state_value >= self.hs_threshold)
        self.hs_trace = np.sum(self.covariance[self.hs_mask, self.hs_mask])

    def forward_simulation(self, fake_measurement_dict):
        _, cov_post, _ = self.posterior_cal(fake_measurement_dict)
        hs_trace = np.sum(np.diagonal(cov_post)[self.hs_mask])
        return hs_trace

    # get the grid cells observed by the current measurment
    def state_query(self, fov):
        query_list = []
        if fov[0] > 0:
            col_s = np.floor(fov[0] / self.x_gsd).astype(int)
        else:
            col_s = 0
        if fov[2] > 0:
            row_s = np.floor(fov[2] / self.y_gsd).astype(int)
        else:
            row_s = 0
        if fov[1] < self.x_boundary:
            col_e = np.ceil(fov[1] / self.x_gsd).astype(int)
        else:
            col_e = self.x_resolution
        if fov[3] < self.y_boundary:
            row_e = np.ceil(fov[3] / self.y_gsd).astype(int)
        else:
            row_e = self.y_resolution

        for r in range(row_s, row_e):
            for c in range(col_s, col_e):
                index = r * self.x_resolution + c  # index of grid cell in the cell list
                x_min = self.state_vector[index][0] - self.x_gsd / 2
                x_max = x_min + self.x_gsd
                y_min = self.state_vector[index][1] - self.y_gsd / 2
                y_max = y_min + self.y_gsd
                query_list.append((index, np.array([x_min, x_max, y_min, y_max])))
        return query_list

    # calculate true RMSE and recall compared with hotspot in ground truth
    def evaluate_metrics(self):
        gt_X_hs = self.gt_X[self.gt_mask]
        gt_Y_hs = self.gt_Y[self.gt_mask]
        area_gt_pixel = self.gt_gsd[0] * self.gt_gsd[1]

        err_square = []
        hs_point_num = len(gt_X_hs)
        union_map = 0
        recall_point_num = 0

        # for j, n in enumerate(self.state_vector):
        #     f = self.state_value[j]
        #     if f > self.hs_threshold and self.exploration_mask[j]:
        #         union_map += self.x_gsd * self.y_gsd

        #     for i, x in enumerate(gt_X_hs):
        #         if (
        #             abs(x[0] - n[0]) < 0.5 * self.x_gsd
        #             and abs(x[1] - n[1]) < 0.5 * self.y_gsd
        #         ):
        #             err_square.append((f - gt_Y_hs[i]) ** 2)
        #             if f > self.hs_threshold and self.exploration_mask[j]:
        #                 recall_point_num += 1
        for j, n in enumerate(self.state_vector):
            if self.exploration_mask[j]:
                f = self.state_value[j]
                if f > self.hs_threshold:
                    union_map += self.x_gsd * self.y_gsd

                for i, x in enumerate(gt_X_hs):
                    if (
                        abs(x[0] - n[0]) < 0.5 * self.x_gsd
                        and abs(x[1] - n[1]) < 0.5 * self.y_gsd
                    ):
                        err_square.append((f - gt_Y_hs[i]) ** 2)
                        if f > self.hs_threshold and self.exploration_mask[j]:
                            recall_point_num += 1
        hs_recall = recall_point_num / hs_point_num
        iou = (recall_point_num * area_gt_pixel) / (
            (hs_point_num - recall_point_num) * area_gt_pixel + union_map
        )

        return np.sqrt(np.mean(err_square)), hs_recall, iou

    # record all the data needed for reconstructing the map
    def get_effective_memory(self):
        return [
            self.covariance,
            self.state_vector,
            self.state_value,
            self.x_gsd,
            self.y_gsd,
        ]

    def get_cell_number(self):
        return len(self.state_vector)

    @property
    def final_map(self):
        center_list = self.state_vector
        size_list = [[self.x_gsd, self.y_gsd] for _ in range(len(center_list))]
        value_list = self.state_value

        return [[center_list, size_list, value_list], self.get_effective_memory()]

    def visualizer(self, ax):
        ax.clear()
        color_map = cm.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(0, 1)

        for i, n in enumerate(self.state_vector):
            x1 = n[0] - self.x_gsd / 2
            y1 = n[1] + self.y_gsd / 2
            x2 = n[0] + self.x_gsd / 2
            y2 = n[1] - self.y_gsd / 2
            color_f = color_map(norm(self.state_value[i]))
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
