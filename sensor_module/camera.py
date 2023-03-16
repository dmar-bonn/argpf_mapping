import numpy as np


class Camera:
    def __init__(self, cfg):
        self.x_resolution = cfg["x_resolution"]
        self.y_resolution = cfg["y_resolution"]
        self.x_angle = cfg["x_angle"]
        self.y_angle = cfg["y_angle"]
        self.noise_coeff = cfg["noise_coeff"]

        self.sensor_x = 0
        self.sensor_y = 0
        self.sensor_al = 0

    def move_sensor(self, pose):
        self.sensor_x = pose[0]
        self.sensor_y = pose[1]
        self.sensor_al = pose[2]

    @property
    def altitude_noise(self):
        return self.noise_coeff * self.sensor_al

    @property
    def FoV(self):
        x_min = self.sensor_x - self.sensor_al * np.tan(np.radians(self.x_angle))
        x_max = self.sensor_x + self.sensor_al * np.tan(np.radians(self.x_angle))
        y_min = self.sensor_y - self.sensor_al * np.tan(np.radians(self.y_angle))
        y_max = self.sensor_y + self.sensor_al * np.tan(np.radians(self.y_angle))
        fov = np.array([x_min, x_max, y_min, y_max])

        x_pixel_int = np.linspace(x_min, x_max, self.x_resolution + 1, endpoint=True)
        y_pixel_int = np.linspace(y_min, y_max, self.y_resolution + 1, endpoint=True)

        gsd_x = x_pixel_int[1] - x_pixel_int[0]
        gsd_y = y_pixel_int[1] - y_pixel_int[0]

        return x_pixel_int, y_pixel_int, gsd_x, gsd_y, fov
