from mapping_module.fr_idp import FR_IDP
from mapping_module.ar_idp import AR_IDP
from mapping_module.ar_bcm import AR_BCM
from mapping_module.ar_gpr_ik import AR_GPR_IK
from mapping_module.fr_gpf import FR_GPF
from mapping_module.ar_gpf_ik import AR_GPF_IK
import logging

logger = logging.getLogger(__name__)
MAPPER_LIST = [
    "fr-idp",
    "ar-idp",
    "ar-bcm",
    "ar-gpr-ik",
    "fr-gpf",
    "ar-gpf-ik",
]


def get_mapper(cfg, mapper_name):
    mapper_cfg = cfg["mapper"]

    if isinstance(mapper_cfg, dict):

        if mapper_name in MAPPER_LIST:
            logger.info(f"------instantiate {mapper_name} mapper")
            if mapper_name == "fr-idp":
                return FR_IDP(mapper_cfg[mapper_name])
            elif mapper_name == "ar-idp":
                return AR_IDP(mapper_cfg[mapper_name])
            elif mapper_name == "ar-bcm":
                return AR_BCM(mapper_cfg[mapper_name])
            elif mapper_name == "ar-gpr-ik":
                return AR_GPR_IK(mapper_cfg[mapper_name])
            elif mapper_name == "fr-gpf":
                return FR_GPF(mapper_cfg[mapper_name])
            elif mapper_name == "ar-gpf-ik":
                return AR_GPF_IK(mapper_cfg[mapper_name])

        else:
            raise RuntimeError(f"{mapper_name} is not implemented")

    else:
        raise RuntimeError(f"{type(mapper_cfg)} not a valid config file")
