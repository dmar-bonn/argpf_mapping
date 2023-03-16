import numpy as np
from planning_module.lawn_mower import Lawn_Mower
from planning_module.greedy import Greedy
import logging

logger = logging.getLogger(__name__)
PLANNER_LIST = ["lawn_mower", "greedy"]


def get_planner(cfg, planner_name):
    planner_cfg = cfg["planner"]

    if isinstance(planner_cfg, dict):

        if planner_name in PLANNER_LIST:
            logger.info(f"------instantiate {planner_name} planner")
            if planner_name == "lawn_mower":
                return Lawn_Mower(planner_cfg[planner_name])
            elif planner_name == "greedy":
                return Greedy(planner_cfg[planner_name])
        else:
            raise RuntimeError(f"{planner_name} is not implemented")

    else:
        raise RuntimeError(f"{type(planner_cfg)} not a valid config file")
