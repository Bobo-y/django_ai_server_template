# coding:utf-8
"""
Parameters tools
"""
import copy


def merge_dict(input_configs: dict,
               default_configs: dict,
               merged_configs: dict = None) -> dict:
    """ load default configs to input configs

    Args:
        input_configs: a dict including some params;
        default_configs: a default dict like input configs;
        merged_configs: output dict

    Returns:
        a merged dict
    """
    if not merged_configs:
        merged_configs = copy.deepcopy(default_configs)

    for key, val in input_configs.items():
        if key not in default_configs:
            merged_configs[key] = val
        elif isinstance(val, dict):
            merged_configs[key] = merge_dict(input_configs[key],
                                             default_configs[key],
                                             merged_configs[key])
        else:
            merged_configs[key] = val

    return merged_configs