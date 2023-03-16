from planning_module.planner import Planner
import numpy as np
import time


class Greedy(Planner):
    def __init__(self, cfg):
        super().__init__()
        self.budget_type = cfg["budget_type"]
        self.budget = cfg["budget"]
        self.flight_speed = cfg["flight_speed"]
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.current_pose = cfg["start_pose"]
        self.prediction_horizon = cfg["prediction_horizon"]
        self.altitude_level = cfg["altitude_level"]
        self.waypoint_distance = cfg["waypoint_distance"]

        self.sensor = None
        self.mapper = None
        self.planning_time = 0
        self.action_space = self.create_action_space()

    def create_action_space(self):
        """
        Hard-coded action space (grid)
        """

        action_space = []
        # l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18]
        l1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        l2 = [5, 7.5, 10, 12.5, 15]
        for i in l1:
            for j in l1:
                action_space.append([i, j, 2])

        for i in l2:
            for j in l2:
                action_space.append([i, j, 5])
        return action_space

    # generate fake measurments for forward simulation
    def generate_fake_measurement(self):
        pixel_x, pixel_y, gsd_x, gsd_y, fov = self.sensor.FoV
        altitude_noise = self.sensor.altitude_noise
        x_num = self.sensor.x_resolution
        y_num = self.sensor.y_resolution

        x_center = 0.5 * (pixel_x[1:] + pixel_x[:-1])
        y_center = 0.5 * (pixel_y[1:] + pixel_y[:-1])
        center_matrix = np.c_[x_center.ravel(), y_center.ravel()]
        data_matrix = np.random.randn(y_num, x_num)

        fake_measurement_data = {
            "data_matrix": data_matrix,
            "center_matrix": center_matrix,
            "sensor_info": {
                "fov": fov,
                "gsd_x": gsd_x,
                "gsd_y": gsd_y,
                "altitude_noise": altitude_noise,
            },
        }

        return fake_measurement_data

    def plan_waypoint(self, mapper, sensor):
        self.mapper = mapper
        self.sensor = sensor
        self.planning_time = 0
        next_pose = self.greedy_search()[0]

        return next_pose, self.planning_time

    def simulate_prediction(self, current_trace_hs, current_position, next_position):
        self.sensor.move_sensor(next_position)
        fake_measurement = self.generate_fake_measurement()

        time_s = time.time()
        next_hs_trace = self.mapper.forward_simulation(fake_measurement)
        time_e = time.time()
        self.planning_time += time_e - time_s

        reward = self.reward(
            current_trace_hs, next_hs_trace, current_position, next_position
        )
        return reward, next_position, next_hs_trace

    def greedy_search(self):
        argmax_action = None
        argmax_trace_hs = 0
        max_reward = -np.inf
        waypoints = []
        current_trace_hs = self.mapper.hs_trace
        current_position = self.current_pose

        for _ in range(self.prediction_horizon):
            simulation_args = [
                (current_trace_hs, current_position, action)
                for action in self.action_space
                if np.any(action != current_position)
            ]
            simulation_values = []
            for arg in simulation_args:
                simulation_values.append(self.simulate_prediction(*arg))

            for simulation_value in simulation_values:
                reward, action, next_trace_hs = simulation_value
                if reward > max_reward:
                    max_reward = reward
                    argmax_action = action
                    argmax_trace_hs = next_trace_hs

            current_trace_hs = argmax_trace_hs
            current_position = argmax_action
            waypoints.append(argmax_action)

        return waypoints

    def reward(self, current_trace_hs, next_trace_hs, current_position, next_position):
        flight_time = self.cal_flight_time(current_position, next_position)
        gain = current_trace_hs - next_trace_hs
        return gain / (flight_time + 1e-5)
