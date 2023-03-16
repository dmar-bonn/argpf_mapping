import yaml
import copy
from simulator import get_simulator
from agent import get_agent
from logger import setup_logger
from utils.analysis_tool import Analysis_Tool
import argparse


def main():
    logger = setup_logger()
    args = parse_args()

    round_num = args.repeat
    experiment_type = args.experiment_type
    with open(f"config/{experiment_type}_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    analysis_tool = Analysis_Tool(cfg)

    for i in range(round_num):
        experiment_num = str(i + 1)
        simulator = get_simulator(cfg["simulator"])
        analysis_tool.record_simulator_data(simulator, experiment_num)
        agent_list = cfg["agent_list"]

        for agent_name, agent_cfg in agent_list.items():
            logger.info(
                f"---------- start experiment: round{experiment_num}/{agent_name} ----------\n"
            )
            agent = get_agent(copy.deepcopy(cfg), agent_cfg)
            simulator.run(agent)
            analysis_tool.record_experiment_data(agent, agent_name, experiment_num)

            logger.info(
                f"---------- finish experiment: round{experiment_num}/{agent_name} ----------\n"
            )

    if not args.test_run:
        analysis_tool.analyse()


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_type",
        "-E",
        required=True,
        type=str,
        help="experiment type, either planning or mapping",
    )

    parser.add_argument(
        "--repeat",
        "-R",
        type=int,
        default=5,
        help="experiment repeat times",
    )

    parser.add_argument(
        "--test_run", action="store_true", help="no analysis if in test_run mode"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
