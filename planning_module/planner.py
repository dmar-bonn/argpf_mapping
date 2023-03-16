import numpy as np


class Planner:
    def plan_waypoint(self):
        raise NotImplementedError("'plan_waypoint' method is not defined!")

    def budget_update(self, execution_time):

        if self.budget_type == "step":
            self.budget -= 1
        elif (
            self.budget_type == "execution_time"
        ):  # constant velocity model for simplicity
            self.budget -= execution_time

    # constant velocity model
    def cal_flight_time(self, current_pose, next_pose):
        flight_dist = self.cal_flight_dist(current_pose, next_pose)
        return flight_dist / self.flight_speed

    @staticmethod
    def cal_flight_dist(current_position, next_position):
        return np.sqrt(
            np.sum((np.array(current_position) - np.array(next_position)) ** 2)
        )
