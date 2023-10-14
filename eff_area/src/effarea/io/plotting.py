#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import yaml


class Plots:
    def __init__(self):
        return None

    def from_result_yaml(self, yaml_path):
        """ """
        self._yaml_path = yaml_path
        if not os.path.exists(self._yaml_path):
            raise FileNotFoundError(
                "you have to supply the path to the correct yaml file"
            )
