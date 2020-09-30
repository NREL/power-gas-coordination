# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:48:35 2019
@author: Brian Sergi (bsergi@nrel.gov)

specifies scenarios to be run in gas/power coordinated market
"""

from classes import *

# see NewModel class object in classes.py for details on what parameters are specified

# June week (Week 1) #######################################################################################################
model1 = newModel(scenario=1, startDate='6/02/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True)
model2 = newModel(scenario=2, startDate='6/02/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True)
model3 = newModel(scenario=3, startDate='6/02/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')
model4 = newModel(scenario=4, startDate='6/02/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')

## July week (Week 2) #######################################################################################################
model5 = newModel(scenario=5, startDate='7/14/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True)
model6 = newModel(scenario=6, startDate='7/14/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True)
model7 = newModel(scenario=7, startDate='7/14/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')
model8 = newModel(scenario=8, startDate='7/14/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')

# November week (Week 3) ##########################################################################################################
model9 = newModel(scenario=9, startDate='11/17/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True)
model10 = newModel(scenario=10, startDate='11/17/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True)
model11 = newModel(scenario=11, startDate='11/17/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')
model12 = newModel(scenario=12, startDate='11/17/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')

# December week (Week 4) ############################################################################################################
model13 = newModel(scenario=13, startDate='12/15/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True)
model14 = newModel(scenario=14, startDate='12/15/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True)
model15 = newModel(scenario=15, startDate='12/15/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')
model16 = newModel(scenario=16, startDate='12/15/2024', daRuns=7, idFrcstImprvmt=20, iters=[2,2,2], idCoord=True, ratable=True, xml='PowerModel_DA_ID_RT_EqualHorizons_2026.xml')
