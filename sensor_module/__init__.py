from sensor_module.camera import Camera
import logging

logger = logging.getLogger(__name__)
SENSOR_LIST = ["camera"]


def get_sensor(cfg, sensor_name):
    sensor_cfg = cfg["sensor"]

    if isinstance(sensor_cfg, dict):

        if sensor_name in SENSOR_LIST:
            logger.info(f"------instantiate {sensor_name}\n")
            if sensor_name == "camera":
                return Camera(sensor_cfg[sensor_name])
        else:
            raise RuntimeError(f"{sensor_name} is not implemented")

    else:
        raise RuntimeError(f"{type(sensor_cfg)} not a valid config file")
