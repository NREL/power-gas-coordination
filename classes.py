# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:23:52 2019
@author: Brian Sergi (bsergi@nrel.gov)

creates classes / data structures for gas-power coordination model
    - modelRun: information about a specific scenario
    - directory: path variables for inputs/outputs and modeling scripts
    - saintModel: settings and functions related to running the gas model
    - plexosModel: settings and functions related to running the power model
    - resultsDF: stores relevant results across runs
"""

import os
from os.path import join
from shutil import copyfile, copytree, rmtree
from ctypes import *
import pandas as pd
import numpy as np
import pdb
import subprocess
import time

from helper_functions import *

# structure that sets parameters for model runs (see calls in main file for descriptions)
class newModel:
    def __init__(self, scenario, startDate, daRuns, idFrcstImprvmt, iters, idCoord, ratable=False,
                 xml='PowerModel_DA_ID_RT_EqualHorizons.xml', allowInc=[True, True, True], reScalars=(1,1)):

        # main model settings
        self.names = ['DA','ID','RT']   # run day-ahead (DA), intra-day (ID), and real-time (RT) scenarios
        self.scenario = scenario        # scenario name
        self.startDate = startDate      # start date for the model being run
        self.idFrcstImprvmt = idFrcstImprvmt # forecast improvement for wind and solar from DA to ID
        self.iters = dict(zip(self.names, iters))   # number of iterations to run each model
        self.reScalars = dict({"wind": reScalars[0], "solar": reScalars[1]})    # scalar parameters for wind and solar generation
        self.plexosXML = xml    # xml for PLEXOS

        # establish other model parameters (either calculated or not typically changed)

        # DA, ID, and RT optimization hours (fixed at 24, 8, and 1 hours each)
        self.optHours = dict(zip(self.names, [24, 8, 1]))

        # DA, ID, and RT look ahead windows
        idHoursLA = 18 - self.optHours['ID'] # calculate for ID (fixed total horizon at 18)
        self.optLookAhead = dict(zip(self.names, [24, idHoursLA, 2]))

        # number of runs (subtract 1 since runs start at 0)
        self.daRuns = daRuns - 1
        self.idRuns = self.optHours['DA']//self.optHours['ID'] - 1
        self.rtRuns = self.optHours['DA']//self.optHours['RT'] - 1

        self.runs = dict(zip(self.names, [self.daRuns, self.idRuns, self.rtRuns]))
        self.mktForInitConds = 'RT' #DON'T CHANGE: which mkt to use for initial conditions in subsequent simulations

        # use ratable quanties (i.e. fixed over a day) in gas network for DA and ID
        if ratable:
            self.ratable = dict(zip(self.names, [True, True, False]))
        # otherwise allow non-ratable quantities (shaped flows) in all markets
        else:
            self.ratable = dict(zip(self.names, [False, False, False]))

        # Start dt for PLEXOS and SAINT runs and start dt for data
        # self.runStartDt, self.dataStartDt = pd.to_datetime('1/1/2012 0:00'), pd.to_datetime(startDate + ' 0:00') # test system
        self.runStartDt, self.dataStartDt = pd.to_datetime('1/1/2024 0:00'), pd.to_datetime(startDate + ' 0:00') # test system

        # DON'T CHANGE: whether to coord power-gas in DA and RT (defaults to True)
        self.coord = dict(zip(self.names, [True, idCoord, True]))

        # whether to allow unconstrained generators to increase offtakes in 2nd iteration (defaults to True for each market)
        self.allowInc = dict(zip(self.names, allowInc))


    # enables print method for modelRun objects
    def __str__(self):
        returnString = []
        returnString.append("Scenario " + str(self.scenario))
        returnString.append("Week of " + str(self.startDate))
        returnString.append("Allow gas offtake increase: " + str(self.allowInc))
        returnString.append("Renewables forecast improvement " + str(self.idFrcstImprvmt) + "%, scalar " + str(self.reScalars))
        returnString.append("Coordination: " + str(self.coord))
        returnString.append("Optimization hours: " + str(self.optHours))
        returnString.append("Look-ahead window: " + str(self.optLookAhead))
        returnString.append("Iterations: " + str(self.iters))
        returnString.append("Market for initial conditions: " + self.mktForInitConds)
        returnString.append("Days run: " + str(self.daRuns + 1)) # add one since DA runs adjusted to start at 0
        returnString.append("PLEXOS xml: " + str(self.plexosXML))
        returnString.append("Use ratable flows: " + str(self.ratable))

        return "\n".join(returnString)

    # enforce all iters >= 1 (otherwise, no market coordination)
    def checkIterations(self):
        for market in self.names:
            self.iters[market] = 1 if self.iters[market] < 1 else self.iters[market]
            # if not coordinating, force iteration to 1
            # if coordinating, make sure iter is at least 2
            if not self.coord[market]:
                self.iters[market] = 1
            elif self.coord[market] and self.iters[market] < 2:
                self.iters[market] = 2

    # outputs information on scenario to text file
    def save(self, directory):
       # write model settings as text file
       currentDir = os.getcwd()
       os.chdir(directory.resultsFolder)
       file = open("scenario.txt", "w")
       file.write(self.__str__())
       file.close()
       os.chdir(currentDir)

    # updates model progress
    def saveUpdateTime(self, directory, updateTime):
      # write model settings as text file
      currentDir = os.getcwd()
      os.chdir(directory.resultsFolder)
      file = open("scenario.txt", "a")
      file.write("\n" + updateTime)
      file.close()
      os.chdir(currentDir)

    # adds details on run periods based on PLEXOS market freqs
    def addRunPeriods(self, plexos):
        runPeriods = dict()
        for name in self.names:
            timeStep = pd.Timedelta(hours=self.optHours[name]+self.optLookAhead[name]-1)
            runPeriods.update({name: pd.date_range(self.runStartDt, self.runStartDt + timeStep, freq=plexos.freq[name]) })
        self.runPeriods = runPeriods

# structure to organize directories and folders
class newDirectory:
    def __init__(self, baseDir, weekFolder):
        # construct hierarchy of files
        self.baseDir = baseDir
        self.weekDir = os.path.join(baseDir, weekFolder)

        # grouping folder for all results and output files
        self.resultsName = "Results"

        # SAINT model files
        self.netFile = os.path.join(self.weekDir, "GasModelCOSystem_New")  # CO system
        self.dllFile = 'C:\\Program Files\\encoord\\SAInt Software 2.0'    # use for CO system

        # PLEXOS model files
        self.plexosFolder = join(self.weekDir,'PowerModelCOSystem')         # CO system
        self.plexosInputs = join(self.plexosFolder,'InputData')
        self.plexosDir = 'C:\\Program Files (x86)\\Energy Exemplar\\Plexos 7.4 R2' # CO system

        # log files
        self.outputText = os.path.join(baseDir, self.resultsName)

    # enables print for directory structure
    def __str__(self):
        return ' ,'.join([self.weekDir, self.netFile, self.plexosFolder])

    # currently set up so that each run has its own sub-folder
    def createRunResultsDir(self, model):
        # store name for log file
        self.name = model.scenario

        self.resultsFolder = join(self.baseDir, self.resultsName, "Scenario " + str(model.scenario))
        # create results folder if it doesn't already exist
        if os.path.isdir(self.resultsFolder):
            scenarioTime = keepTime("Deleting previous results folder...", self)
            rmtree(self.resultsFolder)
            keepTime("Deleting previous results folder...", self, scenarioTime)
        os.makedirs(self.resultsFolder)

    # establish folder structure for storing PLEXOS results
    def createPlexosRunFolders(self, model, plexos):

        # Folders for saving raw PLEXOS inputs and outputs
        subFolders = ['Inputs', 'Results', 'Solutions']
        for sub in subFolders:
            d = os.path.join(self.resultsFolder, sub + "Saved")
            folderName = "plexosSave" + sub + "Dir"
            setattr(self, folderName, d)
            # remove files and replace any existing files
            if os.path.isdir(d): rmtree(d)
            os.makedirs(d)

        # create empty pointer to interval folder (update when results are created)
        self.plexosIntervalFolder = None

    # creates folders to save SAInt outputs
    def createSaintRunFolders(self):
        saintFolders = ['SavedSaintSolns', 'SavedSaintInputs']
        for subFolder in saintFolders:
            d = os.path.join(self.resultsFolder, subFolder)
            setattr(self, subFolder, d)
            if os.path.isdir(d): rmtree(d)
            os.makedirs(d)

# structure for storing saint model information
class saintModel:
    def __init__(self, APIlink, modelDirectory):
        self.saintDll = cdll.LoadLibrary(os.path.join(modelDirectory.dllFile, APIlink))

        # formatting for Saint model return values (e.g character, int, boolean, float)
        self.saintDll.evalStr.restype=c_wchar_p
        self.saintDll.evalInt.restype=c_int
        self.saintDll.evalBool.restype=c_bool
        self.saintDll.evalFloat.restype=c_float

        # Set up initial SAINT network
        #self.saintDll.openGNET(os.path.join(modelDirectory.netFile, 'GNET90.net')) # test system
        self.saintDll.openGNET(os.path.join(modelDirectory.netFile, 'KM.net')) # CO system

        self.saintDll.loadMappingFile(os.path.join(modelDirectory.netFile, "Mapping.txt"))

        # names for different SAINT runs
        self.steady = "STECase"
        self.quasi = "QUACase"
        self.dynamic = "DYNCase"

        # load list of gas plants that are linked between PLEXOS and SAINT
        mapping = pd.read_csv(os.path.join(modelDirectory.netFile, "Mapping.txt"), sep="\t")
        mapping.columns = ["Fuel", "Name", "Generator", "Node"]
        self.mapping = mapping

    # define parameters
    def setSaintParameters(self, model, directory):
        self.saintDynDays = model.optHours['DA']//24
        directory.createSaintRunFolders()

    # Update max fuel offtakes based on SAINT output. Called after PLEXOS --> SAINT iteration
    def updateMaxFuelOfftake(self, runInfo, model, directory, offtakeData):

        # get run periods used by PLEXOS for this market level
        runPeriods = model.runPeriods[runInfo['market']]

        # read in offtake data from latest SAINT run
        os.chdir(directory.netFile)
        fileIn = ''.join(["FuelOfftakeSaint_ForPLEXOS_", runInfo["market"], ".csv"])

        try:
            saintOfftakes = pd.read_csv(fileIn, index_col=0, parse_dates=['Datetime'], date_parser=dateparseAgg)
        except:
            print("Error reading SAINT offtakes.")
            pdb.set_trace()

        # ensure that offtakes matches dates of PLEXOS run, otherwise flag as error
        try:
            assert(saintOfftakes.shape[0] == len(runPeriods))
        except:
            print("Error with SAINT offtakes length.")
            pdb.set_trace()
        # subject to appropriate timeframe

        saintOfftakes = saintOfftakes.loc[runPeriods]

        # check whether to increase max fuel offtakes in SAINT at generators with non-binding constraints
        # (specified by user for each of the three models: DA, ID, RT)
        if model.allowInc[runInfo["market"]]:
            saintOfftakes = incMaxOfftakesOfNonconstrainedGens(saintOfftakes, model, directory, offtakeData, runInfo)
        else:
            offtakeData['offtakeDiffs'].append([runInfo['runName'], runInfo['iteration'], None])
            offtakeData['cumulativeOfftakeLimits'] = None

        # save files for future PLEXOS runs (unless this is the last iteration)
        #if runInfo["iteration"] < model.iters[runInfo["market"]]:
        # re-index for PLEXOS run
        saintOfftakes.index = runPeriods
        saintOfftakes.index.name = 'Datetime'

        # save new offtake values (2 copies: 1 for PLEXOS and 1 to save)
        fileOut = ''.join(["FuelOfftake", "_", runInfo["market"], ".csv"])
        os.chdir(directory.plexosInputs)
        saintOfftakes.to_csv(fileOut)

        fileSave = ''.join(["FuelOfftakeCoord", "_", runInfo["runName"], " iter ", str(runInfo["iteration"]), ".csv"])
        os.chdir(directory.plexosSaveInputsDir)
        saintOfftakes.to_csv(fileSave)
        os.chdir(directory.baseDir)

    def runSaint(self, model, directory, runInfo, startDate, results, writeFuelOfftakes=True, useLookAhead=True,):

        # define optimization window (length differs by market)
        if useLookAhead:
            simLength = model.optHours[runInfo['market']] + model.optLookAhead[runInfo['market']]
        # in case of final RT runs, do not use look-ahead window
        else:
            simLength = model.optHours[runInfo['market']]

        # check if model is running on first market on first day
        firstRun = runInfo["day"] + runInfo["run"] == 0

        # update cumulative runs
        if runInfo['market'] == "DA":
            runInfo['cumulativeRun'] = runInfo["day"]
        else:
            runInfo['cumulativeRun'] = runInfo["day"] * model.optHours['DA'] // model.optHours[runInfo['market']] + runInfo['run']

        # if this is the first run, run steady-state and quasi-steady-state models
        if firstRun:
            self.runSaintSteady(model.runStartDt, directory)
            self.runSaintQuasi(model.runStartDt, directory, simLength)
        else:
            # Past 1st run, SAINT runs for extra hour relative to PLEXOS because .con file is
            # soln to first period, so need to go first period plus full simulation length.
            simLength += 1
            # Need to add offtakes for first period from RT solution (specified by .con soln) to PLEXOS files
            self.adjustPlexosOfftakes(model, directory, simLength, runInfo)
        #For dynamic, either use SS/QS .con (1st runs) or use prior RT dynamic .con
        conRunName = self.quasi if firstRun else 'RT' + self.dynamic
        self.runSaintDynamic(startDate, simLength, model, directory, firstRun, runInfo, conRunName, results, writeFuelOfftakes, useLookAhead)

    def runSaintSteady(self, runStartDate, directory):
        startTime = keepTime("SAINT: steady-state", directory, indent=2)
        # set timeframe for simulation (run for 1 hour to get steady-state)
        simEnd = runStartDate + pd.Timedelta(hours=1)
        simStart, simEnd = runStartDate.strftime('%d/%m/%Y %H:%M'), simEnd.strftime('%d/%m/%Y %H:%M')
        os.chdir(directory.netFile)

        # initialize simulation with time and PLEXOS data
        self.saintDll.writeGCON(True) # ensure the .con file is written for first SAINT run
        self.saintDll.newGSCE(self.steady, "SteadyGas", simStart, simEnd, 3600) #3600 = 1 hour time step
        self.saintDll.importGSCE("SceEventSteady.txt")

        self.saintDll.fetchPlexosData(directory.plexosIntervalFolder) # get gas offtake from PLEXOS
        self.saintDll.writePlexosDataToGSCE(directory.plexosIntervalFolder)

        # run simulation
        self.saintDll.runGSIM()

        # there's an issue with SAInt threading...testing a pause in the script
        # time.sleep(200)

        keepTime("SAINT: steady-state", directory, startTime, indent=2)
        os.chdir(directory.baseDir)

    def runSaintQuasi(self, runStartDate, directory, simLength):
        startTime = keepTime("SAINT: quasi-steady-state", directory, indent=2)

        # if simLength is 1, don't subtract (need a range for dynamic solution w/o LA)
        adjust = 0 if simLength == 1 else 1
        simEnd = runStartDate + pd.Timedelta(hours=simLength-adjust)
        simStart, simEnd = runStartDate.strftime('%d/%m/%Y %H:%M'), simEnd.strftime('%d/%m/%Y %H:%M')
        os.chdir(directory.netFile)

        # initialize simulation with time and PLEXOS data
        self.saintDll.newGSCE(self.quasi, "DynamicGas", simStart, simEnd, 1800) #1800 = 30 min time step
        self.saintDll.importGSCE("SceEventQuasiSteady.txt") # BS: no events to simulate, so this reads nothing at the moment
        # start with results from steady-state run
        self.saintDll.openGCON(''.join([self.steady, ".con"]))
        # run simulation
        self.saintDll.writeGCON(True) # ensure the .con file is written for first SAINT run

        self.saintDll.runGSIM()
        # copyfile(os.path.join(directory.netFile, "QUACase.con"), os.path.join(directory.netFile, "QUACase_saved.con"))

        keepTime("SAINT: quasi-steady-state", directory, startTime, indent=2)
        os.chdir(directory.baseDir)

    def runSaintDynamic(self, dataStartDate, simLength, model, directory, firstRun, runInfo, conRunName,
                        results, writeFuelOfftakes, useLookAhead):
        startTime = keepTime("SAINT: dynamic", directory, indent=2)

        # flag for final iteration ('iter' in runInfo starts at 1, not 0)
        finalIter = runInfo['iteration'] == model.iters[runInfo['market']]

        dynSceFile, dynSceFileForRun = 'SceEventDynamic', 'SceEventDynamicForRun'
        # prfFile, prfFileForRun = 'PRF_CGS_Import','PRF_CGS_ImportForRun'          # test system
        prfFile,prfFileForRun = 'PRF_Demand_Import','PRF_Demand_ImportForRun'       # CO system

        # If running RT w/ LA for coord, give SAINT files diff name and don't write .con file (uses prior RT .con)
        if runInfo['market'] == 'RT' and useLookAhead:
            mktNameForSaintFiles, writeConFile = ''.join([runInfo['market'], 'WithLA']), False
            nameForSolutionFile = '_WithLA_'
        else:
            mktNameForSaintFiles, writeConFile = runInfo['market'], finalIter
            nameForSolutionFile = '_'

        # Dyn .sce file has timed events. Next 2 lines insert timed events at right time.
        if firstRun: takeOutEventsFromDynSce(dynSceFile, directory.netFile)
        createDynSceFileWithEvents(dynSceFile, dynSceFileForRun, runInfo, model, directory, useLookAhead)

        # Match profile of CGS demand w/ current simulation time
        createProfileForRun(model, directory, runInfo, prfFile, prfFileForRun, useLookAhead)

        # set window for SAINT run
        # if simLength is 1, don't subtract (need a range for dynamic solution w/o look-ahead)
        adjust = 0 if simLength == 1 else 1
        simEnd = model.runStartDt + pd.Timedelta(hours=simLength-adjust)
        simStart, simEnd = model.runStartDt.strftime('%d/%m/%Y %H:%M'), simEnd.strftime('%d/%m/%Y %H:%M')

        #Run SAINT w/ updated profiles, dyn sce file, and PLEXOS fuel offtakes
        newConName = mktNameForSaintFiles + self.dynamic

        self.saintDll.newGSCE(newConName, "DynamicGas", simStart, simEnd, 1800) #1800 = 30 min time step
        self.saintDll.importGPRF(os.path.join(directory.netFile, ''.join([prfFileForRun, ".txt"]))) # profile
        self.saintDll.importGSCE(os.path.join(directory.netFile, ''.join([dynSceFileForRun, ".txt"]))) # gas scenario events
        self.saintDll.openGCON(os.path.join(directory.netFile, ''.join([conRunName, ".con"])))
        self.saintDll.writeGCON(writeConFile) #whether to write .con file or not (don't do it when iterating w/ PLEXOS)

        # retrieve PLEXOS offtake data from last interval folder
        if directory.plexosIntervalFolder is None:
            print("Error: PLEXOS offtakes not found.")
            pdb.set_trace()

        try:
            self.saintDll.fetchPlexosData(directory.plexosIntervalFolder)
            self.saintDll.writePlexosDataToGSCE(directory.plexosIntervalFolder) # outputs profiles in interval folders
        except:
            print("Error fetching PLEXOS offtakes.")
            pdb.set_trace()

        # run dynamic gas simulation in SAInt
        self.saintDll.runGSIM()

        # set to print out SAINT log (don't need if running through powershell)
        # also, these files are HUGE so only want to use when debugging
        # currently not working for some reason as well
        # self.saintDll.setSIMLOGFile(os.path.join(directory.resultsFolder, "log_saint.txt"))
        # self.saintDll.writeSIMLOG(True)

        # save copy of .con and .sce file in Saint Solutions folder
        if writeConFile:
            copyfile(os.path.join(directory.netFile, newConName + ".con"), os.path.join(directory.SavedSaintSolns, newConName + runInfo["runName"] + ".con"))
            copyfile(os.path.join(directory.netFile, newConName + ".sce"), os.path.join(directory.SavedSaintInputs, newConName + runInfo["runName"] + ".sce"))

        if writeFuelOfftakes:
            createFuelOfftakeCSVs(self, directory, runInfo)

        # Save inventories and consumptions by gas nodes as well as solution files (save on all iterations)
        saveGasConsumptions(results, self, model, directory, dataStartDate, runInfo)
        solFile = 'gsoloutDyn'
        resultsFile = ''.join(['gsoloutDyn', nameForSolutionFile, runInfo['runName'], runInfo['coordinationTag'], '.txt'])
        self.saintDll.writeGSOL(os.path.join(directory.netFile, "gsolin.txt"), os.path.join(directory.netFile, ''.join([solFile, ".txt"])))
        copyfile(os.path.join(directory.netFile, ''.join([solFile, '.txt'])), os.path.join(directory.SavedSaintSolns, resultsFile))
        keepTime("SAINT: dynamic", directory, startTime, indent=2)

    # function to ensure times line up between SAINT and PLEXOS runs
    # SAINT provides intial solution in first line, so need to add corresponding time
    # period to the PLEXOS offtakes
    def adjustPlexosOfftakes(self, model, directory, simLength, runInfo):
        # prior interval folder: saved solution from last RT run
        # interval folder: folder from current iteration

        # note from Michael: to avoid benefits of coordination in uncoordinated run,
        # use the uncoordinated prior solution for uncoordinated runs and the
        # coordinated prior solution for coordinated runs (see also updateDataFile and updatePlexosFixedVal)
        #priorCoordTag = ' Coord' if model.coord[model.mktForInitConds] else ''
        priorIntervalFolderRT = os.path.join(''.join(['Model Solution ', runInfo['priorRunName'], runInfo["coordinationTag"]]), 'interval')

        # go to interval results from previous PLEXOS run
        for f in os.listdir(directory.plexosIntervalFolder):

            if '.Offtake' in f:
                # read in prior and current offtakes ('plexosIntervalFolderRT' refers to most recent RT solution)
                try:
                    priorOfftakes = pd.read_csv(os.path.join(directory.plexosSaveSolutionsDir, priorIntervalFolderRT, f), index_col=0, parse_dates=['DATETIME'], date_parser=dateparse)
                    offtakes = pd.read_csv(os.path.join(directory.plexosIntervalFolder, f), index_col=0, parse_dates=['DATETIME'], date_parser=dateparse)
                except:
                    print("Error reading offtakes from PLEXOS.")
                    pdb.set_trace()

                firstPeriodSoln = priorOfftakes.values[model.optHours['RT']-1][0]
                # In RT coord, run SAINT twice, so have already done this when run SAINT a second time.
                # Therefore only add the initial row if not already done (if # offtakes < sim length).
                if offtakes.shape[0] < simLength:
                    firstIdx = offtakes.index.values[0]
                    offtakes.index += pd.Timedelta(hours=1)
                    offtakes.loc[firstIdx] = firstPeriodSoln
                    offtakes.sort_index(inplace=True)
                    offtakes.to_csv(os.path.join(directory.plexosIntervalFolder, f))

# structure to hold information on plexos model
class plexosModel:
    def __init__(self, model, directory):

        # Names and frequencies of markets
        self.daName, self.idName, self.rtName = 'DA','ID','RT'
        self.names = [self.daName, self.idName, self.rtName]
        freq = ['1H','1H','1H']
        self.freq = dict(zip(self.names, freq))

        # call function fo create folders for storing PLEXOS output
        directory.createPlexosRunFolders(model, self)
        # set executable file
        self.exe = os.path.join(directory.plexosDir, "PLEXOS64.exe")
        self.xml = os.path.join(directory.plexosFolder, model.plexosXML)

    # create and save intraday wind, solar, and load forecasts for PLEXOS run
    def createIDForecasts(self, model, directory):
        os.chdir(directory.plexosInputs)

        for param in ['Wind','Solar','Load']:
            try:
                daDf = pd.read_csv(''.join([param, 'All_', self.daName, ".csv"]) ,index_col=0, parse_dates=True)
                rtDf = pd.read_csv(''.join([param, 'All_', self.rtName, ".csv"]), index_col=0, parse_dates=True)
            except:
                print("Error reading %s all file" % param)
                pdb.set_trace()
            idDf = (daDf - rtDf)*(1-model.idFrcstImprvmt/100) + rtDf
            # drop any missing dates (occurs with Load for some reason)

            #idDf.dropna(inplace=True)
            idDf.to_csv(''.join([param, 'All_', self.idName, ".csv"]))
            # BS? : how are the RT and DA forecasts generated? see paper from Carlo
        os.chdir(directory.baseDir)

    def updateGasPrice(self, model, directory):
        # following price scheme fixed (EIA 923 data for 2018, $ per mmBtu)
        # june: 3, july: 2.93, nov: 4.5, dec: 4.2
        prices = dict({"6":3, "7":2.93, "11":4.5, "12": 4.2})
        # set based on month of start date
        priceNew = prices[model.startDate.split('/')[0]]

        # read in price files
        os.chdir(directory.plexosInputs)
        gasPriceDA = pd.read_csv("NaturalGasPrice_DA.csv", index_col=0, parse_dates=True)
        gasPriceID = pd.read_csv("NaturalGasPrice_ID.csv", index_col=0, parse_dates=True)
        gasPriceRT = pd.read_csv("NaturalGasPrice_RT.csv", index_col=0, parse_dates=True)

        # overvwrite value and replace
        gasPriceDA['VALUE'] = priceNew
        gasPriceDA.to_csv("NaturalGasPrice_DA.csv")

        gasPriceID['VALUE'] = priceNew
        gasPriceID.to_csv("NaturalGasPrice_ID.csv")

        gasPriceRT['VALUE'] = priceNew
        gasPriceRT.to_csv("NaturalGasPrice_RT.csv")
        os.chdir(directory.baseDir)


    # function to read in Plexos objects. Object lists copied from PLEXOS GUI -> Generators -> Objects
    def getPlexosObjs(self, directory):

        os.chdir(directory.plexosFolder)

        # list of all generators in the model
        self.gensAll = pd.read_csv('GeneratorList.csv', header=None, names=['class','name','category'], index_col=False)
        self.gens = self.gensAll['name'].values
        fuels = pd.read_csv('FuelList.csv', header=None,names=['class','name',''], index_col=False)
        self.fuels = fuels['name'].values

        fuelNames = pd.read_csv("FuelNames.csv", header=None,names=['class','name'], index_col=False)
        self.fuelNames = fuelNames['name'].values

        # list of generators that can't recommit (either ID or RT)
        gensNotRecommitID = pd.read_csv('GeneratorNotRecommitListID.csv', header=None, names=['class','name','category'], index_col=False)
        self.gensNotRecommitID = gensNotRecommitID['name'].values
        gensNotRecommitRT = pd.read_csv('GeneratorNotRecommitListRT.csv', header=None, names=['class','name','category'], index_col=False)
        self.gensNotRecommitRT = gensNotRecommitRT['name'].values

        # Hydro generators
        gensHydro = pd.read_csv('GeneratorHydroList.csv', header=None, names=['class','name','category'], index_col=False)
        self.gensHydro = gensHydro['name'].values

        # Regions, nodes, and transmissions
        regions = pd.read_csv('RegionList.csv', header=None, names=['class','name',''], index_col=False)
        self.regions = regions['name'].values
        nodes = pd.read_csv('NodeList.csv', header=None,names=['class','name',''], index_col=False)
        self.nodes = nodes['name'].values
        lines = pd.read_csv('LineList.csv', header=None,names=['class','name',''], index_col=False)
        self.lines = lines['name'].values

        # types of reserves model: regulation (up and down) and spinning
        reserves = pd.read_csv('ReserveList.csv', header=None, names=['class','name',''], index_col=False)
        self.reserves = reserves['name'].values

        # emissions types
        self.emissions = ['CA Emission_CO2', 'Non-CA Emission_CO2', 'System Emission_NOx', 'System Emission_SO2']

        os.chdir(directory.baseDir)

    # updates PLEXOS csvs before run. Also saves input files to results
    def update(self, runInfo, results, start, intervalsFull, model, directory, saint, offtakeData, useOffTakes):

        # coordinated if this upcoming PLEXOS run occurs after a round of SAINT
        # use SAINT offtake data if currently on > 1st iteration AND coordination is allowed in market
        if useOffTakes:
            runInfo["coordinationTag"] = ' Coord'
        else:
            runInfo["coordinationTag"] = ''

        # get run periods used by PLEXOS for this market level
        runPeriods = model.runPeriods[runInfo['market']]

        updateDataFile("Load", runInfo, directory, intervalsFull, runPeriods)

        # if in RT, hydro generation already committed so don't read file here
        if runInfo['market'] != "RT":
            updateDataFile("HydroGen", runInfo, directory, intervalsFull, runPeriods)

        # pass scalar multiplier for increasing wind/solar penetration
        updateDataFile("Wind", runInfo, directory, intervalsFull, runPeriods, model.reScalars["wind"])
        updateDataFile("Solar", runInfo, directory, intervalsFull, runPeriods, model.reScalars["solar"])

        # 1st PLEXOS run uses initial gas offtakes from file
        # 2nd PLEXOS run (coordinated) uses inputs from SAINT
        # 3rd PLEXOS run etc. builds on cumulative limits

        # w/ coordinated SAINT run, restrict offtakes to max feasible from gas run
        if useOffTakes:
            print(''.join(["  ", "Using SAINT offtakes"]))
            # note: updateMaxFuel called immediately after SAINT after 1st iteration, so don't need to call here

        # in absence of a prior SAINT run, take fuel offtakes from initial PLEXOS run
        else:
            print(''.join(["  " + "Not using SAINT offtakes"]))
            updateDataFile("FuelOfftake", runInfo, directory, intervalsFull, runPeriods)
            offtakeData['cumulativeOfftakeLimits'] = None

        # note from Michael: only use coordinated priors during coordinated run
        # however, Michael's code always uses coordinated RT priors (changed here to run correctly;
        # should also make change to updatePlexosFixedVal when adjusting)
        #priorCoordTag = ' Coord' if model.coord[model.mktForInitConds] else ''
        #priors = results[''.join([model.mktForInitConds, priorCoordTag])]
        priors = results[''.join([model.mktForInitConds, runInfo["coordinationTag"]])]

        # data on whether generators are on/off in previous period
        # use data from prior RT runs when applicable (stored in priors: rtGen ,rtOnOff, rtHUp, rtHDown)
        updateGenInitCSV('InitialGen', runInfo, self, model, directory, intervalsFull, priors.gen, True)
        updateGenInitCSV('InitialGenUnits', runInfo, self, model, directory, intervalsFull, priors.onOff)
        updateGenInitCSV('InitialHoursUp', runInfo, self, model, directory, intervalsFull, priors.HUp)
        updateGenInitCSV('InitialHoursDown', runInfo, self, model, directory, intervalsFull, priors.HDown)

    # run PLEXOS by calling executable
    # note: make sure related input/output files and folders are closed in order to run
    def runPlexos(self, runInfo, date, results, directory):
        os.chdir(directory.plexosFolder)

        # modelName --> DA, ID, RT
        # date indicates current run date for model
        modelName = runInfo['market']
        resultsName = runInfo['market'] + runInfo["coordinationTag"]

        # iteratively try calls to PLEXOS
        tryPlexos, attempt = True, 1
        while tryPlexos:
            modelSpecifier, endProgSpecifier = '\\m', '\\n'
            plexosInput = [self.exe, self.xml, modelSpecifier, modelName, endProgSpecifier]

            try:
                plexosResult = subprocess.call(plexosInput)
                results[resultsName].PlexosOutput.loc[date] = plexosResult
                error = False
            except:
                print("Error with PLEXOS call.")
                error = True

            # check if PLEXOS failed to run
            filename = ' '.join(["~Model (", modelName, ") Log.txt"])
            if filename in os.listdir() or error:
                attempt += 1
                if attempt > 3:
                    print("3 attempts at PLEXOS failed.")
                    raise ValueError
                else:
                    print("PLEXOS run failed...trying again in a minute (attempt %d)" % attempt)
                    time.sleep(60)
            else:
                tryPlexos = False

        os.chdir(directory.baseDir)

    # steps through objects of interest in PLEXOS model and extracts results
    # from PLEXOS' csv output; stores in results data structure for use across markets
    def saveResults(self, results, model, directory, runInfo, intervals):

        # get run periods used by PLEXOS for this market level
        runPeriods = model.runPeriods[runInfo['market']]

        # go to PLEXOS working directory (where PLEXOS automatically save results)
        solutionName = ''.join(['Model ', runInfo['market'], ' Solution'])
        solutionFolder = os.path.join(directory.plexosFolder, solutionName)
        os.chdir(solutionFolder)

        # read solution ID/name mapping file
        idNameMapping = pd.read_csv('id2name.csv', index_col=False)
        intervalFolder = os.path.join(solutionFolder, 'Interval')
        os.chdir(intervalFolder)

        # identify market where results should be stored (DA, ID, RT)
        currentResults = results[''.join([runInfo['market'], runInfo['coordinationTag']])]

        # 1. generator results
        generatorOutputs = ['Generation', 'Units Generating', 'Hours Up', 'Hours Down',
                            'Units Started', 'Units Shutdown', "Fuel Offtake",
                            'Fuel Cost', "VO&M Cost", "Start & Shutdown Cost", "FO&M Cost"]
        # access sections of the results data that correspond to the list above
        generatorResults = [currentResults.gen, currentResults.onOff, currentResults.HUp, currentResults.HDown,
                            currentResults.starts, currentResults.stops, currentResults.offtake,
                            currentResults.fuelCost, currentResults.vomCost, currentResults.startCost, currentResults.fomCost]
        # cycle through each variable and within each one, iterate over all objects and consolidate results into one df
        for genVar, genResult in zip(generatorOutputs, generatorResults):
            self.readPlexosResult(idNameMapping, "Generator", genVar, runInfo, genResult, intervals, runPeriods)

        # 2. fuel results (use these for updated fuel offtakes from SAINT generators)

        # use subset of id mapping to only save SAINT generator offtakes
        fuelMapping = idNameMapping.loc[idNameMapping['name'].isin(self.fuelNames)]
        self.readPlexosResult(fuelMapping, "Fuel", "Offtake", runInfo, currentResults.fuelOfftakes, intervals, runPeriods)

        # 3. region results
        regionOutputs = ['Price', 'SRMC', 'Unserved Energy', 'Dump Energy']
        # access sections of the results data that correspond to the list above
        regionResults = [currentResults.price, currentResults.RSRMC, currentResults.RUnserved, currentResults.RDump]
        # cycle through each variable and within each one, iterate over all objects and consolidate results into one df
        for regionVar, regionResult in zip(regionOutputs, regionResults):
            self.readPlexosResult(idNameMapping, "Region", regionVar, runInfo, regionResult, intervals, runPeriods)

        # 4. nodal results (currently not being used)
        #nodeOutputs = ['Price', 'Unserved Energy', 'Dump Energy']
        # access sections of the results data that correspond to the list above
        #nodeResults = [currentResults.NPrice, currentResults.NUnserved, currentResults.NDump]
        # cycle through each variable and within each one, iterate over all objects and consolidate results into one df
        #for nodeVar, nodeResult in zip(nodeOutputs, nodeResults):
        #    self.readPlexosResult(idNameMapping, "Node", nodeVar, runInfo, nodeResult, intervals, runPeriods)

        # 5. line results
        self.readPlexosResult(idNameMapping, "Line", "Flow", runInfo, currentResults.flows, intervals, runPeriods)

        # 6. reserves
        self.readPlexosResult(idNameMapping, "Reserve", "Shortage", runInfo,
                              currentResults.ResShort, intervals, runPeriods)

        # 7. reserve provision by generator (multi-index result)
        self.readMultiIndexPlexosResult(idNameMapping, "Reserve", "Generator", "Provision", runInfo,
                                        currentResults.Res, intervals, runPeriods)

        # 8. emissions by generator (multi-index result)
        self.readMultiIndexPlexosResult(idNameMapping, "Emission", "Generator", "Production", runInfo,
                                        currentResults.GenEmit, intervals, runPeriods)
        os.chdir(directory.baseDir)

    # Read PLEXOS flat file, then add to df w/ prior results. See plexos.saveResults().
    def readPlexosResult(self, idNameMapping, objectType, outputVar, runInfo, resultsSubDF, intervals, runPeriods):
        # select IDs for corresponding object type (e.g. Generator, Fuel, Region, Node, etc.)
        idObjects = idNameMapping.loc[idNameMapping['class'] == objectType]
        # list of IDs and names
        ids = idObjects['id'].tolist()
        names = idObjects['name'].tolist()

        # Iterate across files within each object class and save results
        for plexosId, objectName in zip(ids, names):
            # filename; based on PLEXOS output naming convention
            fileName = 'ST ' +  objectType + "(" + str(plexosId) + ")." + outputVar + '.csv'
            # read in data and store in appropriate results data frame
            try:
                resultsTemp = pd.read_csv(fileName, index_col=0, parse_dates=['DATETIME'], date_parser=dateparse)
                # select relevant run periods
                resultsSubDF[objectName].loc[intervals] = resultsTemp['VALUE'].loc[runPeriods].values
            except:
                print("Error in reading PLEXOS results.")
                pdb.set_trace()

    # Read PLEXOS flat file w/ multi-index, then add to df w/ prior results. See plexosMode.saveResults().
    def readMultiIndexPlexosResult(self, idNameMapping, objType1, objType2, outputVar, runInfo,
                                   resultsSubDF, intervals, runPeriods):
        # extract indices for both object types
        idObjects1 = idNameMapping.loc[idNameMapping['class']==objType1]
        idObjects2 = idNameMapping.loc[idNameMapping['class']==objType2]

        # convert IDs and names to lists for iteration
        ids1, names1 = idObjects1['id'].tolist(), idObjects1['name'].tolist()
        ids2, names2 = idObjects2['id'].tolist(), idObjects2['name'].tolist()

        # Save results use double-nested for loop across by object types
        for plexosId1, objName1 in zip(ids1, names1):
            for plexosId2, objName2 in zip(ids2, names2):
                # filename based on IDs for both object types (add "s" to second object type)
                fileName = 'ST ' +  objType1 + "(" + str(plexosId1) + ")." + \
                                    objType2 + "s(" + str(plexosId2) + ")." + outputVar + '.csv'
                # read data from PLEXOS output
                if os.path.isfile(fileName):
                    resultsTemp = pd.read_csv(fileName, index_col=0, parse_dates=['DATETIME'], date_parser=dateparse)
                    resultsTemp = resultsTemp['VALUE'].loc[runPeriods].values
                # if generator/reserve combination is missing, assume no reserves supplied
                else:
                    resultsTemp = 0.0
                # store in results object
                resultsSubDF[objName2].loc[pd.MultiIndex.from_product([[objName1], intervals])] = resultsTemp

    # copies entire PLEXOS run output into results directory for long-term storage
    def copyPlexosOutput(self, directory, runInfo):
        # folder where PLEXOS automatically save results
        solutionName = ''.join(['Model ', runInfo['market'], ' Solution'])
        solutionFolder = os.path.join(directory.plexosFolder, solutionName)
        # folder in results section for final save
        destinationName = ''.join(['Model Solution ', runInfo['runName'], runInfo["coordinationTag"]])
        destinationFolder = os.path.join(directory.plexosSaveSolutionsDir, destinationName)
        # copy entire folder into results
        if os.path.isdir(destinationFolder): rmtree(destinationFolder)
        try:
            copytree(solutionFolder, destinationFolder)
        except:
            print("Error copying results! Trying again.")
            try:
                if os.path.isdir(destinationFolder): rmtree(destinationFolder)
                copytree(solutionFolder, destinationFolder)
            except:
                print("Second error copying results.")
                pdb.set_trace()
        # folder where PLEXOS automatically save results (interval level)
        intervalFolder = os.path.join(directory.plexosFolder, solutionName, "interval")
        # update pointer to interval file (used by SAINT run)
        directory.plexosIntervalFolder = intervalFolder

# data structure for carrying intermediate results from PLEXOS and SAINT.
# see plexosModel.saveResults() for details on what data are stored here
# make sure to also update in saveResultCSVs
class resultsDF:
    def __init__(self, market, model, directory, plexos, saveSaintResults=False):

        # create a results data frame for market (DA, ID, or RT)
        self.market = market
        # build out appropriate time series for each market (add 1 since daRuns start at 0)
        timeStep = pd.Timedelta(hours = (model.daRuns+1)*model.optHours["DA"] + model.optLookAhead["DA"])
        self.fullIntervals = pd.date_range(model.dataStartDt, model.dataStartDt + timeStep, freq=plexos.freq[market])

        self.gen = pd.DataFrame(index=self.fullIntervals, columns=plexos.gens)
        # variables of interest with results structured like gens
        variables = ["onOff", "HUp", "HDown", "starts", "stops", "fuelCost", "vomCost", "startCost", "fomCost", "offtake"]
        for var in variables:
            setattr(self, var, self.gen.copy())

        # take offtakes from fuels (not from generators)
        self.fuelOfftakes = pd.DataFrame(index=self.fullIntervals, columns=plexos.fuelNames)

        self.price = pd.DataFrame(index=self.fullIntervals, columns=plexos.regions)
        # variables of interest with results structured like price
        variables = ["RSRMC", "RUnserved", "RDump"]
        for var in variables:
            setattr(self, var, self.price.copy())

        # currently not using nodal results (would need to activate in plexos xml)
        #self.NDump = pd.DataFrame(index=self.fullIntervals, columns=plexos.nodes)
        ## variables of interest with results structured like dump
        #variables = ["NUnserved", "NPrice"]
        #for var in variables:
        #    setattr(self, var, self.NDump.copy())

        self.flows = pd.DataFrame(index=self.fullIntervals, columns=plexos.lines)
        self.ResShort = pd.DataFrame(index=self.fullIntervals, columns=plexos.reserves)

        # multi-index df for reserves (reserves x time intervals over generator columns)
        self.Res = pd.DataFrame(index=pd.MultiIndex.from_product([plexos.reserves, self.fullIntervals]),columns=plexos.gens)

        # multi-index df for emissions (emission type x time intervals over generator columns)
        self.GenEmit = pd.DataFrame(index=pd.MultiIndex.from_product([plexos.emissions, self.fullIntervals]),columns=plexos.gens)

        # PLEXOS solver output
        self.PlexosOutput = pd.Series()

        # dictionary for storing results from Saint model
        if saveSaintResults:
            self.SaintResults = {'consumpByNodeTs': None,
                                 'invByNodeTs': None,
                                 'consumpByNode': pd.DataFrame(),
                                 'consumpByNodeType':pd.DataFrame(),
                                 'gnetResults':pd.DataFrame()}

    # these functions allow result data frame objects to be stored in dictionaries
    def __hash__(self):
        return hash((self.market))
    def __eq__(self, other):
        return (self.market) == (other.market)
