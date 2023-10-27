#!/usr/bin/python

from effarea.data.grbs import GRB, GRBList
from effarea.model.model import GRBModel
if __name__ == "__main__":
    grb_list = GRBList()
    for grb in grb_list.grbs:
        try:
            model = GRBModel(grb)
            model.export_yaml()
            model.export_csv()
        except Exception as e:
            print(e)
            with open("log.txt", "a+") as f:
                f.write(str(e))

