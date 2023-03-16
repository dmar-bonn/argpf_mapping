import numpy as np
from planning_module.planner import Planner


class Lawn_Mower(Planner):
    def __init__(self, cfg):
        super().__init__()
        self.budget_type = cfg["budget_type"]
        self.budget = cfg["budget"]
        self.flight_speed = cfg["flight_speed"]
        self.x_boundary = cfg["x_boundary"]
        self.y_boundary = cfg["y_boundary"]
        self.current_pose = cfg["start_pose"]
        self.step_length = cfg["step_length"]
        self.direction_flag = 1

    # dummie lawn mower starting from upper left
    def plan_waypoint(self, _mapper, _sensor):
        if np.all(np.array(self.current_pose) == 0):  # skip first planning
            next_pose = [2.5, 2.5, 2.5]
            return next_pose, 0

        next_pose = self.current_pose[:]
        if (
            self.current_pose[0] + self.direction_flag * 0.5 * self.step_length
            >= self.x_boundary
        ):
            next_pose[1] += self.step_length
            self.direction_flag *= -1
        elif (
            self.current_pose[0] + self.direction_flag * 0.5 * self.step_length <= 0.01
        ):
            next_pose[1] += self.step_length
            self.direction_flag *= -1
        else:
            next_pose[0] += self.direction_flag * self.step_length

        return next_pose, 0
