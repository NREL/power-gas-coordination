# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:22:35 2019

@author: Brian Sergi (bsergi@nrel.gov)

Main script for running gas power network coordination using PLEXOS and SAInt.
Based on original code provided by Michael Craig.

Users should change the 'baseDir' and 'runFolder' variables
and also specify list of models white are to be run at (models generated in scenarios.py)
"""

## specify working directory and folder with SAInt / PLEXOS models here ##
baseDir = 'F:\\Users\\bsergi'
runFolder = 'Week1'

import os
os.chdir(baseDir)

## Packages and libraries ####

# load scenarios  (user-specified)
from scenarios import *

# other packages from this module
from classes import *
from market_function import *

# general python packages
import subprocess, pandas as pd, numpy as np, time, threading, copy
import traceback
from os.path import join
from shutil import copyfile, copytree, rmtree
from ctypes import *
from math import *
import pdb

# execute call to run desired models
if __name__=='__main__':
    modelsTime = print("Running models")
    models = [model1, model2, model3, model4]  # see scenarios.py

    results = list()
    # iterate over models
    for model in models:
        try:
            results.append(runScenario(model, baseDir, runFolder))
        except:
            print("Problem running scenario %d." % model.scenario)
            print(traceback.format_exc())

    keepTime("Running models", None, modelsTime)
