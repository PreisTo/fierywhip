#!/usr/bin/python

import sys
from fierywhip.model.tte_individual_norm import GRBModelIndividualNorm
from fierywhip.frameworks.grbs import GRB

grb = GRB.grb_from_file(sys.argv[1])

model = GRBModelIndividualNorm(grb)
model.fit()
