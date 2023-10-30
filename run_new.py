#!/usr/bin/python

from effarea.data.grbs import GRB, GRBList
from effarea.model.model import GRBModel
from effarea.io.export import Exporter
if __name__ == "__main__":
    grb_list = GRBList()
    for grb in grb_list.grbs:
        try:
            model = GRBModel(grb)
            exporter = Exporter(model)
            exporter.export_yaml()
            exporter.export_matrix()
        except Exception as e:
            print(e)

