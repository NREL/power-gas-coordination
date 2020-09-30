# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:42:25 2019

@author: Brian Sergi (bsergi@nrel.gov)

functions governing model runs and interleaved market calls
DA, ID, and RT runs called recursively using 'runConnectedMarkets' and 'runModel'
"""

import os
from os.path import join
import time
import datetime
import pandas as pd
from shutil import copyfile,copytree,rmtree
from ctypes import *
import pdb
from math import *
import smtplib

from helper_functions import *
from classes import *

## execute model runs ####
def runScenario(model, baseDir, weekFolder):

    # make sure that models iterate at least once (else no coordination)
    model.checkIterations()

    # create file/directory structure
    directory = newDirectory(baseDir, weekFolder)
    # create new log file
    newLogFile(model, directory)

    # create results folder
    directory.createRunResultsDir(model)

    # save scenario information as textfile
    model.save(directory)

    # update log file with start time
    scenarioTime = keepTime("Running scenario %d" % model.scenario, directory)
    sHour = "Start: " + str(datetime.datetime.now())
    writeLog(sHour, directory)

    # load SAINT model details
    startTime = keepTime("Loading SAINT parameters", directory)
    saint = saintModel("SAInt-API.dll", directory)
    saint.setSaintParameters(model, directory)
    keepTime("Loading SAINT parameters", directory, startTime)

    # load PLEXOS model details (Note: it can take a long time to delete pre-existing files)
    startTime = keepTime("Loading PLEXOS parameters", directory)
    plexos = plexosModel(model, directory)
    plexos.getPlexosObjs(directory)
    model.addRunPeriods(plexos)
    #Create intraday WSLT forecasts based on % forecast improvement
    plexos.createIDForecasts(model, directory)
    # update natural gas price (currently fixed by month)
    plexos.updateGasPrice(model, directory)
    keepTime("Loading PLEXOS parameters", directory, startTime)

    # create results data frame (one for each market: DA, ID, RT)
    startTime = keepTime("Loading results dataframe", directory)
    results = dict()
    for market in ["DA", "ID", "RT"]:
        results.update({market: resultsDF(market, model, directory, plexos, True)})
        if model.coord[market]:
            results.update({market + " Coord": resultsDF(market, model, directory, plexos, True)})
    keepTime("Creating results dataframe", directory, startTime)

    # run optimization for connected markets
    runConnectedMarkets(model, directory, plexos, saint, results)
    keepTime("Running scenario %d" % model.scenario, directory, scenarioTime)

    # print scenario / directory information
    os.chdir(directory.baseDir)
    fHour = "Last update: " + str(datetime.datetime.now())
    writeLog(fHour, directory)
    model.saveUpdateTime(directory, fHour)

    return(results)

# main function for calling interleaved markets
def runConnectedMarkets(model, directory, plexos, saint, results):
    # set run periods (temporal structure over which the model actually runs; based on
    # optimization window + look-ahead for each market)

    # store information on current run across loops
    daRunStart, daIterStart = 0, 1
    runName = getRunName("DA", daRunStart)
    runInfo = dict({"market": "DA", "run": daRunStart, "iteration": daIterStart, "runName": runName,
                    "day": daRunStart, "rtRunNum": 0, "cumulativeRun": daRunStart,
                    "coordinationTag": '', "priorRunName": ''})

    # tracks (1) total difference between PLEXOS and SAINT offtakes by run, and (2) constraints
    # on gas offtakes based on most recent SAINT modeling run
    offtakeData = dict({"offtakeDiffs": list(), "cumulativeOffTakeLimits": None })

    # start first DA model (subsequent iterations, lower markets, and days called recursively within function)
    runModel('DA', 0, 1, model, directory, plexos, saint, results, offtakeData, runInfo)

    try:
        # after finishing all runs export relevant results to CSV
        if model.allowInc['DA'] or model.allowInc['ID'] or model.allowInc['RT']:
            offtakeFilename = join(directory.plexosSaveResultsDir, 'totalOfftakeDiffsByIter.csv')
            pd.DataFrame(offtakeData['offtakeDiffs'], columns=['RunName','IterNum','TotalDiff']).to_csv(offtakeFilename)
        saveResultCSVs(results, directory)
    except:
        print("Error saving final csvs.")

# function for running market (DA, ID, or RT)
# called RECURSIVELY across market levels and iterations within each level
def runModel(market, currentRun, currentIter, model, directory, plexos, saint, results, offtakeData, runInfo):  # pass on other data as needed

    # start date for analysis, as well as total hours (opt + look-ahead) and
    # optimization start date and hours
    if market == "DA":
        baseStartDate = model.dataStartDt
    else:
        baseStartDate = model.dataStartDt + pd.Timedelta(hours=runInfo['day']*model.optHours['DA'])
    start = baseStartDate + pd.Timedelta(hours=currentRun*model.optHours[market])
    # daIntvls includes look-ahead window; daOptIntvls only contains dates for optimization
    intervals = pd.date_range(start,  start + pd.Timedelta(hours=model.optHours[market] + model.optLookAhead[market]-1), freq=plexos.freq[market])
    # intervalsSave =  pd.date_range(start,  start + pd.Timedelta(hours=model.optHours[market]-1), freq=plexos.freq[market])

    # keep track of time in function
    currentMarketString = ''.join([market, " market %d (" % currentRun, '{:%Y-%m-%d %H}'.format(start), "), iter %d " % currentIter])
    startTime = keepTime(currentMarketString, directory)

    # update run information (keeps track of position within the model)
    if market == "DA":
        updateRunInfo(runInfo, day=currentRun, rtRunNum=0)   # reset day and RT run counter for a new day
    updateRunInfo(runInfo, market=market, run=currentRun, iteration=currentIter, runName=getRunName(market, currentRun, runInfo['day']))
    # coordinationTag updated in plexos.update

    # update plexos input csvs
    startTime2 = keepTime("Updating PLEXOS csvs", directory, indent=1)
    # update offtake flag to ensure future PLEXOS iterations use gas offtake limits from SAINT
    useOfftakes = currentIter > 1 and model.coord[market]
    plexos.update(runInfo, results, start, intervals, model, directory, saint, offtakeData, useOfftakes)
    # if in ID or RT markets, update inputs using fixed commitments from corresponding DA or ID
    if market == "ID":
        updatePlexosFixedVal('Commit', 'onOff', runInfo, plexos, model, directory, results, intervals, plexos.gensNotRecommitID)
    elif market == "RT":
        updatePlexosFixedVal('Commit', 'onOff', runInfo, plexos, model, directory, results, intervals, plexos.gensNotRecommitRT)
        updatePlexosFixedVal('FixedGen', 'gen', runInfo, plexos, model, directory, results, intervals, plexos.gensHydro)
    keepTime("Updating PLEXOS csvs", directory, startTime2, indent=1)

    # run PLEXOS (depending on its mood, PLEXOS seems to take 90-120 seconds to run)
    startTime2 = keepTime("Running PLEXOS", directory, indent=1)
    plexos.runPlexos(runInfo, start, results, directory)
    keepTime("Running PLEXOS", directory, startTime2, indent=1)

    # export PLEXOS results to inform future runs / coordinated market
    startTime2 = keepTime("Saving PLEXOS results", directory, indent=1)
    startTime3 = keepTime("Copy files", directory, indent=2)
    plexos.copyPlexosOutput(directory, runInfo)
    keepTime("Copy files", directory, startTime3, indent=2)


    # calculate ratable quantities from PLEXOS offtakes if using ratable flows for this market
    # do this only for natural gas plants that are in gas network
    if model.ratable[runInfo['market']]:
        convertToRatableGasOfftakes(model, directory, runInfo, saint)

    # save PLEXOS results that are used to inform lower markets
    startTime3 = keepTime("Save results for next market", directory, indent=2)
    plexos.saveResults(results, model, directory, runInfo, intervals)
    keepTime("Save results for next market", directory, startTime3, indent=2)
    keepTime("Saving PLEXOS results", directory, startTime2, indent=1)

    # run SAINT if coordinating power and gas markets
    if model.coord[market]:
        startTime2 = keepTime("Running SAINT", directory, indent=1)

        saint.runSaint(model, directory, runInfo, start, results)

        # calc offtake diffs and save offtake files for next PLEXOS run
        saint.updateMaxFuelOfftake(runInfo, model, directory, offtakeData)

        # NOTE: in RT need to run SAINT twice:
            #1) Run SAINT for opt + LA horizon to get fuel offtakes for PLEXOS coordinated run.
            #2) Run SAINT for only opt horizon to get .con file for next SAINT DA/ID/RT runs (only on final iter in RT).
        if market == "RT" and currentIter == model.iters[market]:
            keepTime("Running SAINT", directory, startTime2, indent=1)
            startTime2 = keepTime("Running 2nd round SAINT for RT", directory, indent=1)
            # note: don't need to write fuel offtakes here, and run model without look-ahead
            saint.runSaint(model, directory, runInfo, start, results, writeFuelOfftakes=False, useLookAhead=False)
            keepTime("Running 2nd round SAINT for RT", directory, startTime2, indent=1)
        else:
            keepTime("Running SAINT", directory, startTime2, indent=1)

    # even without coordination, run SAINT in RT model to get .con file for future SAINT runs
    elif market == "RT" and currentIter == model.iters[market]:
        startTime2 = keepTime("Running SAINT (uncoordinated RT market)", directory, indent=1)
        saint.runSaint(model, directory, runInfo, start, results, writeFuelOfftakes=True, useLookAhead=False)
        keepTime("Running SAINT (uncoordinated RT market)", directory, startTime2, indent=1)

    # control for passing to next iteration or if done iterations, to the next market level
    if currentIter < model.iters[market]:  # more iterations needed
        # re-rerun model at current market level with one higher iteration
        keepTime(currentMarketString, directory, startTime)
        runModel(market, currentRun, currentIter+1, model, directory, plexos, saint, results, offtakeData, runInfo)

    # only do the rest of the operations after final iteration
    else:
        keepTime(currentMarketString, directory, startTime)
        if market == "DA":
            runModel("ID", 0, 1, model, directory, plexos, saint, results, offtakeData, runInfo)
        elif market == "ID":
            runModel("RT", runInfo['rtRunNum'], 1, model, directory, plexos, saint, results, offtakeData, runInfo)

        # change prior run name at end of run being used for initial conditions (usually RT)
        if market == model.mktForInitConds:
            runInfo['priorRunName'] = runInfo['runName']
        if market == "RT":
            runInfo['rtRunNum'] += 1

        # after finishing iterations and any all rounds of any lower level market runs, move to next time period within market
        # first if statement ensures that RT market loops back to ID after ID optimization hour timeframe has passed
        # e.g. if ID optimization windo is 6 hours, RT runs 0-5 and then passes back to next ID run
        if market == "RT" and ((currentRun + 1) % model.optHours['ID'] == 0):
            pass  # pass up to next ID level
        elif currentRun < model.runs[market]:
            runModel(market, currentRun+1, 1, model, directory, plexos, saint, results, offtakeData, runInfo)
