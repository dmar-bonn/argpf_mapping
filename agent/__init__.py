from mapping_module import get_mapper
from planning_module import get_planner
from sensor_module import get_sensor
from agent.agent import Agent
import logging

logger = logging.getLogger(__name__)


def get_agent(cfg, agent_cfg):
    if isinstance(cfg, dict):
        logger.info("create agent")
        mapper_name = agent_cfg["mapper_name"]
        mapper = get_mapper(cfg, mapper_name)
        sensor_name = agent_cfg["sensor_name"]
        sensor = get_sensor(cfg, sensor_name)
        planner_name = agent_cfg["planner_name"]
        planner = get_planner(cfg, planner_name)

        return Agent(mapper, sensor, planner)

    else:
        raise RuntimeError(f"{type(cfg)} not a valid config file")
