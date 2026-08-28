from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession


def get_imputed_HLA_indicator_vector(
    class_i_list: List[str], class_ii_list: List[str], HLA_list: Union[None, List[str]]
) -> Tuple[np.ndarray, Dict[str, int]]:  # type:ignore[type-arg]
    """
    For a given list of imputed HLAs, generate an indicator vector for each sample

    :param class_i_list: A list of where each element is expected to be a comma separated string
            of all class I HLAs
    :param class_ii_list: A list of where each element is expected to be a comma separated string
            of all class I HLAs
    :param HLA_list: if you have a list of HLAs for which you want the indicator vector
    :return: X_hla is the indicator vector, hla_dict is the dictionary where each HLA is the key
            and the index in X_hla is the value
    """
    hla_dict = {hla_i: i for i, hla_i in enumerate(HLA_list)}
    HLA_indicator_out = []
    for ci, cii in zip(class_i_list, class_ii_list):
        hla_i = np.zeros(len(hla_dict), dtype=int)
        if (type(ci) == str) and (type(cii) == str):
            if len(ci) > 0:
                for hi in ci.split(","):
                    if hi in hla_dict:
                        hla_i[hla_dict[hi]] = 1
            if len(cii) > 0:
                for hii in cii.split(","):
                    if hii in hla_dict:
                        hla_i[hla_dict[hii]] = 1
        HLA_indicator_out.append(list(hla_i))

    return np.asarray(HLA_indicator_out), hla_dict