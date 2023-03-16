import logging
import os
import time
import sys
from matplotlib import axis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb
from constants import PLOT_FONT_SIZE, PLOT_LINE_WIDTH, PLOT_TICKS_SIZE, EXP_DIR
import logging
import matplotlib


logger = logging.getLogger(__name__)


class Analysis_Tool:
    def __init__(self, cfg):
        self.task_type = cfg["analysis_tool"]["task_type"]

        self.experiment_data = {}
        self.analysis_data = {}
        self.ground_truth_data = {}

        self.max_time = cfg["planner"]["greedy"]["budget"]

        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.save_folder = os.path.join(EXP_DIR, f"experiment_{timestamp}")
        os.makedirs(self.save_folder, exist_ok=True)
        self.plot_folder = os.path.join(self.save_folder, "plots")
        os.makedirs(self.plot_folder, exist_ok=True)

    def record_simulator_data(self, simulator, num):
        self.ground_truth_data[num] = {}

        self.ground_truth_data[num]["sim_world"] = simulator.sim_world
        self.ground_truth_data[num]["ground_truth_info"] = simulator.ground_truth_info

        self.ground_truth_plot(simulator, num)

    def record_experiment_data(self, agent, agent_name, num):
        self.experiment_data.setdefault(num, {})
        self.experiment_data[num][agent_name] = {}

        # record data for mapping analysis
        self.experiment_data[num][agent_name][
            "mapping_time"
        ] = agent.mapper.mapping_time
        self.experiment_data[num][agent_name]["final_map"] = agent.mapper.final_map

        # record data for planning analysis
        self.experiment_data[num][agent_name]["rmse_hs"] = agent.rmse_history
        self.experiment_data[num][agent_name]["iou"] = agent.iou_history
        self.experiment_data[num][agent_name]["timeline"] = agent.mission_timeline
        self.experiment_data[num][agent_name]["trajectory"] = agent.trajectory

        self.map_plot(agent.mapper, agent_name, num)

    def ground_truth_plot(self, simulator, num):
        map_folder = os.path.join(self.plot_folder, f"experiment{num}")
        os.makedirs(map_folder, exist_ok=True)

        plt.clf()
        _, ax = plt.subplots()
        ax = simulator.visualizer(ax)
        plt.savefig(
            map_folder + f"/ground_truth.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def map_plot(self, mapper, agent_name, num):
        map_folder = os.path.join(self.plot_folder, f"experiment{num}")
        os.makedirs(map_folder, exist_ok=True)

        plt.clf()
        if self.task_type == "planning":
            fig, ax = plt.subplots(
                2, 1, figsize=(9, 16), gridspec_kw={"height_ratios": [2, 1]}
            )
            ax[0] = mapper.visualizer(ax[0])
            self.flying_path_plot(ax, num, agent_name)

        else:
            if hasattr(mapper, "reconstruction"):
                fig_rec, ax_rec = plt.subplots()
                ax_rec = mapper.reconstruction(ax_rec)
                plt.savefig(
                    map_folder + f"/reconstruction.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.clf()

            fig, ax = plt.subplots()
            ax = mapper.visualizer(ax)

        plt.savefig(
            map_folder + f"/{agent_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def offline_analysis(self, path):
        raw_data = pd.read_pickle(path)
        self.ground_truth_data = raw_data["ground_truth"]
        self.experiment_data = raw_data["experiment"]
        self.metrics_plot()

    def analyse(self):
        data_bag = {}
        data_bag["ground_truth"] = self.ground_truth_data
        data_bag["experiment"] = self.experiment_data
        raw_dataframe = pd.DataFrame(data_bag)
        raw_dataframe.to_pickle(self.save_folder + "/raw_experiment_data_bag.pkl")
        logger.info("start analysing experiments data")

        if self.task_type == "mapping":
            self.analyse_mapping()
        elif self.task_type == "planning":
            self.analyse_planning()

        logger.info("analysis done!")

    def analyse_mapping(self):
        for round_num, round_dict in self.experiment_data.items():
            for exp_name, exp_dict in round_dict.items():
                self.analysis_data.setdefault(exp_name, [])
                exp_results_list = []
                exp_results_list.append(exp_dict["mapping_time"])
                map_data, effective_memory = exp_dict["final_map"]
                gt_data = self.ground_truth_data[round_num]["ground_truth_info"]
                rmse, rmse_hs, iou = self.cal_metrics(gt_data, map_data)
                exp_results_list.append(rmse)
                exp_results_list.append(rmse_hs)
                exp_results_list.append(iou)
                exp_results_list.append(self.get_obj_size(effective_memory))
                exp_results_list.append(len(map_data[0]))
                self.analysis_data[exp_name].append(exp_results_list)

        final_resutls = {}
        for exp_instance, exp_data in self.analysis_data.items():
            final_resutls[exp_instance] = {}
            final_resutls[exp_instance]["mean"] = np.mean(exp_data, axis=0)
            final_resutls[exp_instance]["std"] = np.std(exp_data, axis=0)
        dataframe = pd.DataFrame(final_resutls)
        print(dataframe)
        dataframe.to_pickle(self.save_folder + "/final_results.pkl")

    def analyse_planning(self):
        self.metrics_plot()

    # calculate RMSE, RMSE_hs, IoU
    def cal_metrics(self, gt_data, map_data):
        gt_X, gt_Y, hs_mask, hs_threshold, gt_gsd = gt_data
        area_gt_pixl = gt_gsd[0] * gt_gsd[1]
        hs_point_num = len(gt_X[hs_mask])
        hs_point_intersec = 0
        union_map = 0

        center_list, size_list, value_list = map_data
        grid_num = len(center_list)

        err_square = []
        err_square_hs = []

        for i in range(grid_num):
            f = value_list[i]
            if f > hs_threshold:
                union_map += size_list[i][0] * size_list[i][1]

            for j, p in enumerate(gt_X):
                if (
                    abs(p[0] - center_list[i][0]) < 0.5 * size_list[i][0]
                    and abs(p[1] - center_list[i][1]) < 0.5 * size_list[i][1]
                ):
                    err_square_term = (f - gt_Y[j]) ** 2
                    err_square.append(err_square_term)
                    if hs_mask[j]:
                        err_square_hs.append(err_square_term)
                        if f > hs_threshold:
                            hs_point_intersec += 1

        iou = (hs_point_intersec * area_gt_pixl) / (
            (hs_point_num - hs_point_intersec) * area_gt_pixl + union_map
        )

        return np.sqrt(np.mean(err_square)), np.sqrt(np.mean(err_square_hs)), iou

    def metrics_plot(self):
        rmse_hs_total_df = pd.DataFrame({"hue": [], "x": [], "y": []})
        iou_total_df = pd.DataFrame({"hue": [], "x": [], "y": []})

        for round_num, exp_instance in self.experiment_data.items():
            for exp_title, exp_data in exp_instance.items():

                timeline = exp_data["timeline"]
                rmse_hs = exp_data["rmse_hs"]
                iou = exp_data["iou"]

                mission_time = np.linspace(
                    0, self.max_time, int(np.ceil(self.max_time))
                )
                rmse_hs_interpolated = np.interp(mission_time, timeline, rmse_hs)
                iou_interpolated = np.interp(mission_time, timeline, iou)

                rmse_hs_df = pd.DataFrame(
                    {"x": mission_time, "y": rmse_hs_interpolated, "hue": exp_title}
                )
                iou_df = pd.DataFrame(
                    {"x": mission_time, "y": iou_interpolated, "hue": exp_title}
                )

                rmse_hs_total_df = rmse_hs_total_df.append(
                    rmse_hs_df, ignore_index=True
                )
                iou_total_df = iou_total_df.append(iou_df, ignore_index=True)

        plt.clf()
        ax = sb.lineplot(
            x="x",
            y="y",
            hue="hue",
            err_style="band",
            ci="sd",
            data=rmse_hs_total_df,
            linewidth=2.5,
        )
        ax.set_ylabel("RMSE (hotspots)", fontsize=PLOT_FONT_SIZE)
        ax.set_xlabel("Mission Time [s]", fontsize=PLOT_FONT_SIZE)
        plt.setp(ax.get_legend().get_texts(), fontsize=PLOT_FONT_SIZE)
        ax.get_legend().set_title(None)
        ax.tick_params(labelsize=PLOT_TICKS_SIZE)
        plt.savefig(
            self.plot_folder + "/rmse_hs.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.clf()
        ax = sb.lineplot(
            x="x",
            y="y",
            hue="hue",
            err_style="band",
            ci="sd",
            data=iou_total_df,
            linewidth=2.5,
        )
        ax.set_ylabel("IoU", fontsize=PLOT_FONT_SIZE)
        ax.set_xlabel("Mission Time [s]", fontsize=PLOT_FONT_SIZE)
        plt.setp(ax.get_legend().get_texts(), fontsize=PLOT_FONT_SIZE)
        ax.get_legend().set_title(None)
        ax.tick_params(labelsize=PLOT_TICKS_SIZE)
        plt.savefig(self.plot_folder + "/iou.png", dpi=300, bbox_inches="tight")
        plt.close()

    def flying_path_plot(self, ax, round_num, exp_name):
        waypoints = np.array(self.experiment_data[round_num][exp_name]["trajectory"])
        timeline = np.array(self.experiment_data[round_num][exp_name]["timeline"])
        wp_interp = self.interpolate_wp_2d(waypoints[:, :2])
        num_wp = len(wp_interp)
        color_map = cm.cool(np.linspace(0, 1, num_wp))
        for i in range(num_wp - 1):
            ax[0].plot(
                wp_interp[i : i + 2, 0],
                wp_interp[i : i + 2, 1],
                color=color_map[i],
                linewidth=3,
            )
        ax[0].plot(waypoints[:, 0], waypoints[:, 1], "rx", markersize=13, mew=3)
        ax[1].plot(timeline, waypoints[:, 2], "x-", markersize=8, mec="r")

        return ax

    @staticmethod
    def interpolate_wp_2d(wp):
        new_wp = []
        wp_len = len(wp)
        for i in range(wp_len - 1):
            x0 = wp[i, 0]
            x1 = wp[i + 1, 0]
            x_dis = abs(x1 - x0)
            y0 = wp[i, 1]
            y1 = wp[i + 1, 1]
            y_dis = abs(y1 - y0)
            dis = np.sqrt(x_dis**2 + y_dis**2)
            num = int(np.floor(dis / 0.5))

            x_ls = np.linspace(x0, x1, num=num, endpoint=True)
            y_ls = np.linspace(y0, y1, num=num, endpoint=True)
            wp_interp = np.stack((x_ls, y_ls), axis=-1)
            new_wp.extend(wp_interp)
        return np.array(new_wp)

    def get_obj_size(self, obj, seen=None):
        # From https://goshippo.com/blog/measure-real-size-any-python-object/
        # Recursively finds size of objects
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self.get_obj_size(v, seen) for v in obj.values()])
            size += sum([self.get_obj_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, "__dict__"):
            size += self.get_obj_size(obj.__dict__, seen)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self.get_obj_size(i, seen) for i in obj])
        return size
