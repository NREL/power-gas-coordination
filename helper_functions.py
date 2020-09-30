# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:42:25 2019

@author: Brian Sergi (bsergi@nrel.gov)

functions need to run interleaved markets and to set up gas/power coordinations
"""

import os
from os.path import join
import time
import pandas as pd
import numpy as np
import math
import re

from shutil import copyfile,copytree,rmtree
from ctypes import *
from io import StringIO
import pdb
import warnings

## miscellaneous functions ####

# create new log file
def newLogFile(model, directory):
    os.chdir(directory.outputText)
    fileName = ''.join(["Scenario ", str(model.scenario), "_log.txt"])
    if fileName in os.listdir():
        os.remove(fileName)
    os.chdir(directory.baseDir)

# outputs program status to log file
def writeLog(updateString, directory):
    if directory is not None:
        os.chdir(directory.outputText)
        fileName = ''.join(["Scenario ", str(directory.name), "_log.txt"])
        log = open(fileName, "a")
        log.write(updateString + "\n")
        log.close()
        os.chdir(directory.baseDir)

# time-keeping function
def keepTime(updateString, directory, startTime=None, indent=None):
    if indent is not None:
        updateString = ''.join(['  ' * indent, updateString])
    if startTime is None:
        print(''.join([updateString, "..."]))
        writeLog(updateString, directory)
        return time.time()
    else:
        elapsed = time.time() - startTime
        if elapsed > 3600:
            elapsed = elapsed / 3600 # divide by 3600 seconds per hour
            unit = "hours"
        elif elapsed > 120:
            elapsed = elapsed / 60 # divide by 60 seconds per minute
            unit = "minutes"
        else:
            unit = "seconds"

        print(''.join([updateString, " done (%.1f %s)." %(elapsed, unit)]))
        writeLog(''.join([updateString, " done (", str(round(elapsed, 1)), " ", unit + ")."]), directory)

# date parsing function (speeds up reading csv files)
def dateparse(text):
    for fmt in ['%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M %p',  '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M']:
        try:
            return pd.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')

# specific to PLEXOS style outputs for aggregated files
def dateparseAgg(text):
    return pd.datetime.strptime(text, '%Y-%m-%d %H:%M:%S')
## functions to label (un)coordinated markets ####

# helper functions to build run name for writing csvs
def getRunName(market, run, day=None):
    if market == "DA":
        return ''.join([market, str(run)])
    else:
        return ''.join([market, str(day), "-", str(run)])

# helper function to update runInfo
# **kwargs variables must correspond to values in runInfo
def updateRunInfo(runInfo, **kwargs):
    for key, value in kwargs.items():
        runInfo[key] = value


## functions to update csvs that feed into PLEXOS (see plexos.update()) ####

# use to update various input scripts to PLEXOS, as well as to save those
# inputs in the results folders for reference later
# relevant parameters include Load, HydroGen, Wind, Solar, FuelOfftake
# (other inputs have separate update functions )
def updateDataFile(parameter, runInfo, directory, intervalsFull, runPeriods, reScalar=None):

    # filenames for input and 2 outputs (first for PLEXOS, second to keep)
    fileIn = ''.join([parameter, "All_", runInfo["market"], ".csv"])
    fileOut = ''.join([parameter, "_", runInfo["market"], ".csv"])
    fileSave = ''.join([parameter, "_", runInfo["runName"], ".csv"])

    # BS? : what is the difference between the coordinated and uncoordinated hydro data?
    # according to Michael, nothing. leaving this in here for now.
    if parameter == "HydroGen":
        hydroCoord = runInfo["coordinationTag"].strip()
        fileIn = ''.join([parameter, hydroCoord, "All_", runInfo["market"], ".csv"])
        fileSave = ''.join([parameter, hydroCoord, "_", runInfo["runName"], ".csv"])

    os.chdir(directory.plexosInputs)

    # read specified data frame
    try:
        df = pd.read_csv(fileIn, index_col=0, parse_dates=True)
    except:
        print("Error reading PLEXOS input file.")
        pdb.set_trace()

    # subset data to relevant dates for optimization window (including look-ahead)
    dfRun = df.loc[intervalsFull]

    # if reading renewables, multiply by scalar factor
    if reScalar is not None:
        dfRun = dfRun * reScalar

    # re-index data using base run dates
    dfRun.index = runPeriods
    dfRun.index.name = 'Datetime'

    # save 2 copies: one where plexos will run, the other in results to preserve
    dfRun.to_csv(fileOut)
    os.chdir(directory.plexosSaveInputsDir)
    dfRun.to_csv(fileSave)
    os.chdir(directory.baseDir)

# Update CSVs w/ initial generator conditions passed into PLEXOS.
# note: adjustFloat flag will add a small value to all values greater than zero;
# without this adder, initial generation values are often read as less than minimum load due to float error.
def updateGenInitCSV(parameter, runInfo, plexos, model, directory, intervals, resultsSaved, adjustFloat=False):

    # check if model is running on first market on first day
    firstFlag = runInfo["day"] + runInfo["run"] == 0

    # if this is the first run on the first day, initialize with zeros (all gen start off)
    if firstFlag:
        initDf = pd.DataFrame(0.0, columns=plexos.gens, index=[str(model.runStartDt)]) #str to force 0:00 to print out
    # otherwise read initial conditions from previous results
    else:
        # load corresponding data from previous RT run
        previousPeriod = [intervals[0] - pd.Timedelta(plexos.freq["RT"])]
        initDf = pd.DataFrame(resultsSaved.loc[previousPeriod])
        initDf.index = [str(model.runStartDt)]
        if adjustFloat: initDf[initDf>1E-4] += 1E-4

    # save 2 copies: one for PLEXOS and one to preserve in results
    initDf.index.name = 'Datetime'

    fileOut = ''.join([parameter, "_", runInfo["market"], ".csv"])
    os.chdir(directory.plexosInputs)
    initDf.to_csv(fileOut)

    fileSave = ''.join([parameter, runInfo["coordinationTag"], "_", runInfo["runName"], ".csv"])
    os.chdir(directory.plexosSaveInputsDir)
    initDf.to_csv(fileSave)

    os.chdir(directory.baseDir)


## Gas constraint functions ####

# function to take hourly PLEXOS offtakes and convert them into a ratable flow over the course of the optimization window
def convertToRatableGasOfftakes(model, directory, runInfo, saint):

    # folder for saving copy of ratable flows
    destinationName = ''.join(['Model Solution ', runInfo['runName'], runInfo["coordinationTag"]])
    savedResultsFolder = os.path.join(directory.plexosSaveSolutionsDir, destinationName)
    # go to interval results from previous PLEXOS run
    for f in os.listdir(directory.plexosIntervalFolder):
        # get id number from filename
        id = re.findall('\d+', f)
        if id != []:
            id = int(id[0])
        # read offtake files for plants in SAINT mapping
        if '.Offtake' in f and id in saint.mapping['Fuel'].values:
            # read in prior and current offtakes ('plexosIntervalFolderRT' refers to most recent RT solution)
            try:
                offtakes = pd.read_csv(os.path.join(directory.plexosIntervalFolder, f), index_col=0, parse_dates=['DATETIME'], date_parser=dateparse)
            except:
                print("Error reading offtakes from PLEXOS.")
                pdb.set_trace()
            # save original offtakes for post-processing and analysis
            oldOfftakes = "Nonratable Offtakes Fuel %d.csv" % id # create filename for saving non-ratable offtakes
            offtakes.to_csv(os.path.join(savedResultsFolder, "interval", oldOfftakes))

            # take averages for optimization window (24 hours for DA, 8 hours for ID) and
            # look-ahead window (24 hours for DA, 10 hours for ID)
            optHours = model.optHours[runInfo['market']]
            offtakes.iloc[0:optHours] = offtakes.iloc[0:optHours].mean()['VALUE'] # optimization window
            offtakes.iloc[optHours: ] = offtakes.iloc[optHours: ].mean()['VALUE'] # look-ahead window

            # re-write offtakes to interval folder, save copy in save folder so that downstream analysis uses these results
            offtakes.to_csv(os.path.join(directory.plexosIntervalFolder, f))
            offtakes.to_csv(os.path.join(savedResultsFolder, "interval", f))

# Function limits offtakes when SAINT max offtakes < PLEXOS requested offtakes, otherwise
# allows generators to increase fuel offtake (through arbitrary large offtake #). Offtakes are passed cumulatively, such that
# offtakes in the first iteration are included in subsequent iterations and so on. Returns
# df with max offtakes that other function writes into CSV for PLEXOS.
def incMaxOfftakesOfNonconstrainedGens(saintOfftakes, model, directory, offtakeData, runInfo):

    # read offtakes from any previous PLEXOS run
    plexosOfftakes = compileOfftakesFromPlexosRun(directory, runInfo)
    plexosOfftakes = plexosOfftakes[saintOfftakes.columns] # eliminate non-SAINT-gas offtakes

    # throw error for case with zero SAInt offtakes
    # note: this functionality doesn't work in cases of no offtakes; only use for debugging
    # if saintOfftakes.sum().sum() == 0 and plexosOfftakes.sum().sum() != 0:
    #     updateString = "SAInt Solve Error: " + runInfo["runName"] + " iter " + str(runInfo["iteration"])
    #     writeLog(updateString, directory)
    #     print("Zero offtakes...SAInt solver likely failed. Using offtakes from previous solution.")
    #     pdb.set_trace()
    #
    #     # if SAInt fails, use offtakes limits from last solve
    #     if offtakeData['cumulativeOfftakeLimits'] is not None:
    #         saintOfftakes = offtakeData['cumulativeOfftakeLimits']

    # throw error if there is a mismatch in column dimensions
    if saintOfftakes.shape[1] != plexosOfftakes.shape[1]:
        print("Error: PLEXOS and SAInt offtakes have different dimensions.")
        pdb.set_trace()

    # calculate difference between PLEXOS request and SAInt delivery
    diffs = plexosOfftakes - saintOfftakes # PLEXOS will always be >= SAINT offtakes, so always positive
    # seeing some negatives...is that just a float issue? or a SAINT precision issue?

    # save these differences
    diffsFilename = ''.join(['plexosSaintFuelOfftakeDiffs', runInfo['runName'], ' iter ', str(runInfo['iteration']), '.csv'])
    diffs.to_csv(join(directory.plexosSaveInputsDir, diffsFilename))

    print("    Offtake diffs: min %0.4f, max %0.4f" % (diffs.min().min(), diffs.max().max()))
    # if PLEXOS and SAINT offtakes are similar (i.e. diffs -> 0 ), then set future offtakes to
    # arbitrarily large value for unconstrained operations in future iterations.
    saintOfftakes[diffs < 1E-2] = 100000

    # indicator of whether to use prior offtakes (e.g. if after 1st COORDINATED PLEXOS run, i.e. after 2 iterations)
    # should only need this if statement logic if iterating more than twice (prevents oscillation)
    if runInfo['iteration'] > 1:
        # for any SAINT offtakes greater than cumulative offtake limits, set to lower value
        saintOfftakes[offtakeData['cumulativeOfftakeLimits'] < saintOfftakes] = offtakeData['cumulativeOfftakeLimits']
        # update cumulative limits (equiv. to copying saintOfftakes after having removed prior violations)
        offtakeData['cumulativeOfftakeLimits'][saintOfftakes < offtakeData['cumulativeOfftakeLimits']] = saintOfftakes
    else: # in first iter, don't have any prior offtakes, so skip above and save current offtake as limits
        offtakeData['cumulativeOfftakeLimits'] = saintOfftakes.copy()

    # Write plexos offtakes and difference values for each iteration
    exportFilename =  ''.join(['plexosFuelOfftakes', runInfo['runName'], ' iter ', str(runInfo['iteration']), '.csv'])
    plexosOfftakes.to_csv(join(directory.plexosSaveInputsDir, exportFilename))

    # track total gap between PLEXOS request and SAINT possible offtakes
    # modify this to (1) exclude look-ahead (2) deal with negative values
    offtakeData['offtakeDiffs'].append([runInfo['runName'], runInfo['iteration'], diffs.sum().sum()])

    # return current offtakes based on SAINT constraints
    return saintOfftakes


def compileOfftakesFromPlexosRun(directory, runInfo):
    # locate folder with saved PLEXOS results (these results unaltered by SAINT run, unlike current interval folder)
    os.chdir(directory.plexosSaveSolutionsDir)

    # called at the end of a PLEXOS --> SAINT iteration, so use run info for this current run
    priorSolutionFolder = ''.join(['Model Solution ', runInfo['runName'], runInfo['coordinationTag']])
    priorIntvlFolder = os.path.join(priorSolutionFolder,'interval')
    # list for storing offtake results from PLEXOS
    plexosOfftakes = list()
    # mapping of PLEXOS IDs to generator names
    try:
        id2name = pd.read_csv(join(priorSolutionFolder,'id2name.csv'), dtype={"class": str, "id": int , "name": str})
    except:
        print("Error reading PLEXOS Id values.")
        pdb.set_trace()

    # iterate across offtake files for each generator
    # read file, get gas generator name, and store in list
    # Note: reading ST Fuel(x).Offtake files (not offtakes by generator)
    for f in os.listdir(priorIntvlFolder):
        if '.Offtake' in f:
            try:
                priorOfftakes = pd.read_csv(os.path.join(priorIntvlFolder,f),index_col=0,parse_dates=True)['VALUE']
            except:
                print("Error reading offtakes.")
                pdb.set_trace()
            priorOfftakes.name = mapPlexosIdToGasName(f, id2name)
            plexosOfftakes.append(priorOfftakes)
    os.chdir(directory.baseDir)

    if not plexosOfftakes:    # verify that list is not empty
        print("Error: missing PLEXOS offtakes.")
        pdb.set_trace()
    # combine list of offtakes into 1 data frame and return with generator name as index
    plexosOfftakes = pd.concat(plexosOfftakes, axis=1)
    return plexosOfftakes

# takes PLEXOS IDs and return name gas generator
def mapPlexosIdToGasName(filename, id2name, idClass='Fuel', fieldFrom='id', fieldTo='name'):
    plexosId = int(filename[filename.index('(')+1:filename.index(')')])
    rows = id2name.loc[id2name['class']==idClass]
    mappedVal = rows.loc[rows[fieldFrom]==plexosId, fieldTo].values[0]
    return mappedVal

# updates files for generators whose values are fixed by earlier market runs
# commit called in ID run, both commit and fixedGen called in RT
# in RT, call with rtCoord var for second or greater iterations)
def updatePlexosFixedVal(filename, attribute, runInfo, plexos, model, directory, results, intervals, fixedGenIDs):

    # get run periods used by PLEXOS for this market level
    runPeriods = model.runPeriods[runInfo['market']]

    # set priorMarketLevel based on current level
    if runInfo['market'] == "ID":
        priorMarketLevel = "DA"
    elif runInfo['market'] == "RT":
        priorMarketLevel = "ID"
    else:
        print("Error: fixed value gens can only occur in lower markets!")

    # take commits from coordinated model if coordinating at previous level and currently in coord run
    # Michael uses 'parallel tracks' for commitment across coordination run (except in gen priors, etc.)
    # note: should align with dfs sent to updateGenInitCSV
    if model.coord[priorMarketLevel]:
        priorMarketLevel = ''.join([priorMarketLevel, runInfo["coordinationTag"]])

    # get appropriate results (onOff or gen)
    priorMarketVals = getattr(results[priorMarketLevel], attribute)

    # subselect values for gens that have constraints in appropriate timeframe
    intervalsRounded = intervals.floor(plexos.freq['DA'])
    fixedVals = np.array([priorMarketVals.loc[dt].values for dt in intervalsRounded])
    fixedVals = pd.DataFrame(fixedVals, index=runPeriods, columns=plexos.gens)
    fixedVals = fixedVals[fixedGenIDs]
    fixedVals.index.name = 'Datetime'

    # save in plexos location and also in saved inputs folder
    fullFilenamePlexos = ''.join([filename, "_", runInfo["market"], ".csv"])
    fullFilenameSaved = ''.join([filename, runInfo['coordinationTag'], "_", runInfo["runName"], ".csv"])

    fixedVals.to_csv(join(directory.plexosInputs, fullFilenamePlexos))
    fixedVals.to_csv(join(directory.plexosSaveInputsDir, fullFilenameSaved))


## SAINT helper files ####

# function removes events from dynamic SAINT output file (used on first run only)
def takeOutEventsFromDynSce(dynSceFile, netFolder):
    # Load original sce file
    with open(os.path.join(netFolder, dynSceFile + '.txt'), 'r') as myfile:
        sceText = myfile.read()
    #Separate out events by looking for + in first field of row
    nontimeEvents, timeEvents = list(), list()
    sceTextList = sceText.split('\n')
    for textLine in sceTextList:
        if len(textLine) > 0:
            if textLine[0]=='+': timeEvents.append(textLine)
            else: nontimeEvents.append(textLine)
    #Create 2 new text .sce files in same format as original one
    with open(os.path.join(netFolder,dynSceFile + 'NoTimeEvents.txt'),'w') as myfile:
        myfile.write('\n'.join(nontimeEvents))
    with open(os.path.join(netFolder,dynSceFile + 'TimeEvents.txt'),'w') as myfile:
        myfile.write('\n'.join(timeEvents))


#Create new dynamic .sce that includes timed events in given run
def createDynSceFileWithEvents(dynSceFile, dynSceFileForRun, runInfo, model, directory, useLookAhead):
    #Load .sce files with events & all other info
    with open(os.path.join(directory.netFile, dynSceFile + 'NoTimeEvents.txt'), 'r') as myfile:
        sceText=myfile.read()
    with open(os.path.join(directory.netFile, dynSceFile + 'TimeEvents.txt'), 'r') as myfile:
        sceEventsText=myfile.read()
    sceText, sceEventsText = sceText.split('\n'), sceEventsText.split('\n')
    if sceEventsText != ['']: #not empty
        #Create dict of time:event (full line of text)
        times = dict()
        for event in sceEventsText: times[int(event.split(':')[0][1:])] = event
        #Look for events in current optimization and add to list of other .sce info
        hourStart = runInfo['cumulativeRun']*model.optHours[runInfo['market']]
        hourEnd = (runInfo['cumulativeRun'] + 1) * model.optHours[runInfo['market']]

        hourEnd += model.optLookAhead[runInfo['market']] if useLookAhead else 0

        for t in times:
            if t in list(range(hourStart, hourEnd)):
                hourOfEvent = t - hourStart
                event = times[t]
                event = event[0] + str(hourOfEvent) + event[event.find(':'):]
                sceText.append(event)
    #Remove lines w/ .inv - this sets inventory of gas storage, only want in first run
    if runInfo['cumulativeRun'] > 0: sceText = [line for line in sceText if '.INV' not in line]
    #Write new .sce for run
    sceFile = os.path.join(directory.netFile, dynSceFileForRun + '.txt')
    with open(sceFile,'w') as myfile: myfile.write('\n'.join(sceText))
    fileName = dynSceFileForRun + runInfo['runName'] + '.txt'
    copyfile(sceFile, os.path.join(directory.SavedSaintInputs, fileName))

# sets up SAINT run
def createProfileForRun(model, directory, runInfo, prfFile, prfFileForRun, useLookAhead):

    # read in profile data
    # profile = pd.read_csv(os.path.join(directory.netFile, "Profile time series.csv"), index_col=0, parse_dates=True)

    with open(join(directory.netFile, 'Profile time series.txt'), 'r') as myfile:
        prfFull=myfile.read()
    profile = pd.read_csv(StringIO(prfFull), sep='\t', index_col=0)

    # define header
    header = '[DEF]\t\t\t\t\t\n%ProfileName\tProfileType\tTimeStep[h]\tInterpolation\tDistribution\tDuration\n'

    entries = [] # empty list for each demand profile
    for colName in profile.columns:
        header += ('\t'.join([colName, 'DETERMINISTIC', '0.01', 'Linear', 'UNIFORM', 'PERIODIC']) + '\n')
        entry = '\n[' + colName + ']' + '\t\t\t\t\t\n%MEAN\tDEVIATION\t\t\t\t\n'

        # select appropriate time range here
        # Note: profiles given in increments of 0.01 hours (defined in header), so multiple hour by 100
        # to get the appropriate row from the profile file
        profileIncrement = 100
        startTime = model.optHours[runInfo['market']] * runInfo['cumulativeRun'] * profileIncrement
        if useLookAhead:
            window = model.optHours[runInfo['market']] + model.optLookAhead[runInfo['market']]
        else:
            window = model.optHours[runInfo['market']]

        # adjust for extra time period for initial condition past first run
        if runInfo['cumulativeRun'] == 0:
            simStart, simEnd = startTime, startTime + window * profileIncrement
        else:
            simStart, simEnd = startTime-profileIncrement, startTime + window * profileIncrement     # account for .con soln period

        # multiplier for high gas demand scenario (only multiply for offtake node)
        values = profile[colName]
        gasDemandMultiplier = 0             # zero out gas demand in summer weeks
        if model.startDate in ['6/02/2024', '7/14/2024'] and colName == "DRENNAN_MMSCFD":
            values = values * gasDemandMultiplier

        # select appropriate values using date range
        values = values.astype(str)
        values = values[simStart:simEnd]

        # convert to list and join values
        # note: "0" refers to standard deviation of profile. adjust
        # if not running deterministically
        values = values.tolist()
        # print("Profile length: " + str(len(values)))
        entry = entry + '\t0\t\t\t\t\n'.join(values) + '\t0\t\t\t\t\n'
        entries.append(entry)

    # create profile from all entries
    newProfile = header + ''.join(entries)

    # save profile in gas folder and also save a copy to SAInt inputs folder
    prfFileFullName = join(directory.netFile, ''.join([prfFileForRun, '.txt']))
    with open(prfFileFullName,'w') as myfile: myfile.write(newProfile)
    fileName = ''.join([prfFileForRun, runInfo['runName'], '.txt'])
    copyfile(prfFileFullName, join(directory.SavedSaintInputs, fileName))


# saves fuel offtakes from SAINT to CSVs for later coordination with PLEXOS
def createFuelOfftakeCSVs(saint, directory, runInfo):

    offtakeBaseFilename = 'FuelOfftakeSaint_'
    offtakeFullFilename = ''.join([offtakeBaseFilename, runInfo['market'], '.csv'])
    baseOfftakeFile = os.path.join(directory.netFile, offtakeFullFilename)
    saint.saintDll.writeFuelOfftake(baseOfftakeFile)
    # read offtakes back in from SAINT output
    try:
        offtakes = pd.read_csv(baseOfftakeFile, index_col=0, parse_dates=['Datetime'], date_parser=dateparse)
    except:
        print("Error reading offtakes.")
        pdb.set_trace()
    if runInfo['cumulativeRun'] > 0:
        offtakes = offtakes.iloc[1:] # drop first row soln since it refers to prior PLEXOS hour (needed by .con)
        offtakes.index -= pd.Timedelta(hours=1)
    # save two copies: one in SAINT directory and the other in the results fodler
    fileName1 = ''.join([offtakeBaseFilename, 'ForPLEXOS_', runInfo['market'], '.csv'])
    fileName2 = ''.join([offtakeBaseFilename, 'ForPLEXOS_', runInfo['runName'], " iter ", str(runInfo["iteration"]), '.csv'])
    offtakes.to_csv(os.path.join(directory.netFile, fileName1))
    offtakes.to_csv(os.path.join(directory.SavedSaintInputs, fileName2))

# saves SAINT gas node results in results data frame object (all results other than fuel offtakes at plants for coordination,
# which is saved separately above in createFuelOfftakeCSVs)
def saveGasConsumptions(results, saint, model, directory, dataStartDate, runInfo):
    # link to appropriate results and optimization window
    saintResults = results[''.join([runInfo['market'], runInfo['coordinationTag']])].SaintResults
    hoursOpt = model.optHours[runInfo['market']]
    # extra results from SAINT
    intermediateResults = extractDataFromDll(saint, hoursOpt, dataStartDate)
    # merge with existing data frames for timeseries results
    mergeSaintResultDfs(saintResults, intermediateResults, 'consumpByNodeTs')
    mergeSaintResultDfs(saintResults, intermediateResults, 'invByNodeTs')

    # save other results directly
    saintResults['consumpByNode'][dataStartDate] = intermediateResults['consumpByNode']
    saintResults['consumpByNodeType'][dataStartDate] = intermediateResults['consumpByNodeType']
    saintResults['gnetResults'][dataStartDate] = intermediateResults['gnetResults']

# Return dictionary of total consumption by each node type and gas network
# total outflow (consumption), inflow, and flow balance (FB = outflow - inflow).
# In dictionary, node types providing gas to system (CBI, UGS, LNG) have negative
# flow values, whereas gas network inflow is always positive.
# BS? : not sure I understand the sign nomenclature here
def extractDataFromDll(saint, hoursOpt, dataStartDate):
    # Get number of nodes
    nodeNums = saint.saintDll.evalInt("GNET.NNO")
    # Get final timestep and index of SAINT output at end of optimization horizon (so exclude LA)
    finalTimestep = math.ceil(saint.saintDll.evalFloat("gtmax"))
    timestepsList = convertSaintOutputToList(saint.saintDll.evalStr('gtimes'))
    optEndIdx = [v<=hoursOpt+1E-3 for v in timestepsList]
    if False in optEndIdx: optEndIdx = optEndIdx.index(False)
    else: optEndIdx = len(optEndIdx)
    timestepsList = timestepsList[:optEndIdx]
    # Set up result dicts
    consumpByNodeTs = pd.DataFrame(index=[dataStartDate + pd.Timedelta(hours=h) for h in timestepsList])
    invByNodeTs = consumpByNodeTs.copy()
    consumpByNodeType, consumpByNode, gnetResults = pd.Series(),pd.Series(),pd.Series()
    # Get GNET results
    gnetResults['QOUT'] = saint.saintDll.evalFloat(''.join(["intg(GNET.QOUT.[Msm3/h].(#),0,", str(hoursOpt), ")"]))
    # total gas consumption (NOT net consumption)
    gnetResults['QIN'] = saint.saintDll.evalFloat(''.join(["intg(GNET.QIN.[Msm3/h].(#),0,", str(hoursOpt), ")"]))
    # total gas inflow (positive value)
    gnetResults['FB'] = saint.saintDll.evalFloat(''.join(["intg(GNET.FB.[Msm3/h].(#),0,", str(hoursOpt), ")"]))
    # total flow balance (consump - inflow)

    # Iterate through nodes
    for nodeNum in range(nodeNums):
        #Get total gas consump over simulation for only optimization horizon
        consumpTotalExpr = (''.join(['intg(NO.{0}.Q.[Msm3/h].(#),0,', str(hoursOpt), ')'])).format(nodeNum)
        consump = saint.saintDll.evalFloat(consumpTotalExpr)
        #Get time series of gas consump for only optimization horizon
        consumpTsExpr = 'NO.{0}.Q.[Msm3/h].(#)'.format(nodeNum)
        consumpTs = convertSaintOutputToList(saint.saintDll.evalStr(consumpTsExpr)) # get list of consump ts
        consumpTs = consumpTs[:optEndIdx] #slim to optim horizon
        #Save total consumption by node and node type and save node time series
        nodeType = saint.saintDll.evalStr('NO.{0}.Owner'.format(nodeNum))
        if nodeType in consumpByNodeType: consumpByNodeType[nodeType] += consump
        else: consumpByNodeType[nodeType] = consump
        nodeId = saint.saintDll.evalStr('NO.{0}.ID'.format(nodeNum))
        consumpByNode[nodeId] = consump
        consumpByNodeTs[nodeId] = consumpTs
        #Get storage inventories for LNG & UGS facilities and save
        if nodeType == 'LNG' or nodeType == 'UGS':
            invExpr = 'NO.{0}.INV.[Msm3].(#)'.format(nodeNum)
            invTs = convertSaintOutputToList(saint.saintDll.evalStr(invExpr))
            invByNodeTs[nodeType + ' ' + nodeId] = invTs[:optEndIdx]

    # collect results
    intermediateSaintResults = dict({"consumpByNode": consumpByNode,
                                     "consumpByNodeTs": consumpByNodeTs,
                                     "consumpByNodeType": consumpByNodeType,
                                     "gnetResults": gnetResults,
                                     "invByNodeTs": invByNodeTs})
    return intermediateSaintResults


# helper function to convert SAINT raw output into lists for processing
def convertSaintOutputToList(saintOutput):
    return [float(v) for v in saintOutput.replace('\r','').split('\n')]

# helper function to merge time series of SAINT results
def mergeSaintResultDfs(saintResults, resultsToAdd, category):
    newResult = resultsToAdd[category]
    result = newResult.copy() if saintResults[category] is None else pd.concat([saintResults[category], newResult], axis=0)
    saintResults[category] = result[~ result.index.duplicated(keep='first')] #initial consump = final consump of prior run

## final export functions ##
def saveResultCSVs(results, directory):
    plexosParams = ["PlexosOutput", "gen", "onOff", "HUp", "HDown", "starts",
                    "stops", "offtake", "fuelOfftakes", "price", "RSRMC", "RUnserved", "RDump",
                    "flows", "Res", "ResShort", "fuelCost", "vomCost", "startCost", "fomCost", "GenEmit"]

    # iterate over three market levels
    for market in results:
        #print(market)
        currentDir = os.path.join(directory.plexosSaveResultsDir, market)
        if not os.path.isdir(currentDir): os.makedirs(currentDir)
        # PLEXOS results
        for param in plexosParams:
           # print(param)
            resultToSave = getattr(results[market], param)
            resultToSave.to_csv(os.path.join(currentDir, param + '.csv'))

        # SAINT results (save only after coordinated run)
        if "Coord" in market:
            saintDict = results[market].SaintResults
            for saintResultName in saintDict:
                if saintDict[saintResultName] is not None: # empty when run w/o ID coordination
                    saintDict[saintResultName].to_csv(join(currentDir, saintResultName +'.csv'))
