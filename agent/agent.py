class Agent:
    def __init__(self, mapper, sensor, planner):
        self.mapper = mapper
        self.sensor = sensor
        self.planner = planner
        self.trajectory = []
        self.mission_timeline = []
        self.rmse_history = [0.3]  # for hotspot defined as value > 0.7
        self.recall_history = [0]
        self.iou_history = [0]
        self.move_to_starting_pose()

    # initialize planner and sensor pose
    def move_to_starting_pose(self):
        self.sensor.move_sensor(self.planner.current_pose)
        self.trajectory.append(self.planner.current_pose)
        self.mission_timeline.append(0)

    # plan the next best way point
    def move(self):
        current_pose = self.planner.current_pose
        next_pose, planning_time = self.planner.plan_waypoint(self.mapper, self.sensor)
        self.planner.current_pose = next_pose
        self.sensor.move_sensor(next_pose)
        self.trajectory.append(next_pose)

        flight_time = self.planner.cal_flight_time(current_pose, next_pose)
        time_consumption = planning_time + flight_time
        self.mission_timeline.append(self.mission_timeline[-1] + time_consumption)
        self.planner.budget_update(time_consumption)
        print(next_pose, self.budget)

    # update map with ture measurments
    def update_map(self, measurment_data):
        self.mapper.update_map(measurment_data)
        self.record_metrics_history()

    # record rmse and recall during planning task
    def record_metrics_history(self):
        if self.mapper.record_metrics:
            rmse, recall, iou = self.mapper.evaluate_metrics()
            self.rmse_history.append(rmse)
            self.recall_history.append(recall)
            self.iou_history.append(iou)

    @property
    def budget(self):
        return self.planner.budget
