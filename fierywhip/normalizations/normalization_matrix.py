#!/usr/bin/python

import yaml
import numpy as np
from fierywhip.io.export import matrix_from_yaml


class NormalizationMatrix:
    """
    Create Matrix which stores the relative normalizations in
    dependence of the normalizing det
    """

    def __init__(self, result_yml=None):
        if result_yml is not None:
            self._result_matrix = matrix_from_yaml(result_yml)
        self.create_norm_matrix()
        print(self._matrix)

    def create_norm_matrix(self, lims=(0.5, 1.5)):
        vals = self._result_matrix[:, :, 0]
        error_pos = np.array(self._result_matrix[:, :, 1])
        error_neg = np.array(self._result_matrix[:, :, 2])
        matrix = np.empty((12, 12), dtype=np.float64)
        for i in range(12):
            for j in range(12):
                try:
                    if i != j:
                        ep = error_pos[i, j]
                        en = error_neg[i, j]
                        if lims is not None:
                            pop_indices = []
                            for ind in range(len(vals[i, j])):
                                if (
                                    np.abs(vals[i, j][ind] / lims[0] - 1) < 0.1
                                    or np.abs(vals[i, j][ind] / lims[1] + 1) < 0.1
                                ):
                                    if (np.abs(en[ind]) + np.abs(en[ind])) < 0.1:
                                        pop_indices.append(ind)
                            try:
                                for p in pop_indices.reverse():
                                    vals[i, j].pop(p)
                                    en.pop(p)
                                    ep.pop(p)
                            except TypeError:
                                pass
                        ep = np.abs(np.array(ep))
                        en = np.abs(np.array(en))
                        try:
                            matrix[i, j] = np.float64(
                                np.average(vals[i, j], weights=1 / (ep + en))
                            )
                        except ZeroDivisionError:
                            matrix[i, j] = np.float64(np.mean(vals[i, j]))
                    else:
                        matrix[i, j] = np.float64(0)
                except ValueError:
                    matrix[i, j] = np.nan
        print(matrix)
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix
