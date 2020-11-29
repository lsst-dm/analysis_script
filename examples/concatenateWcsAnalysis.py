import os
import pandas as pd

import lsst.daf.persistence as dafPersist
from lsst.pipe.tasks.parquetTable import ParquetTable

doDecam = False
subFieldNameList = [""]
if doDecam:
    filterList = ["g", "i"]
    fieldName = "decam"
else:
    fieldName = "RC2"
    subFieldName = ""
    # fieldName = "WIDE"
    # subFieldName = ""
    # fieldName = "DEEP"
    # subFieldNameList = ["DUD_ELAIS", "DUD_XMM", "DUD_COSMOS", "DUD_DEEP"]
    filterList = ["HSC-I", "HSC-I2", "HSC-R", "HSC-R2", "HSC-G", "HSC-Z", "HSC-Y",
                  "NB0921", "NB0387", "NB0816"]
    # rootDir = "/datasets/hsc/repo/rerun/private/lauren/DM-24024/" + fieldName + "/" + subFieldName

for subFieldName in subFieldNameList:
    rootDir = "/datasets/hsc/repo/rerun/private/lauren/DM-24024/" + fieldName + "/" + subFieldName
    if doDecam:
        rootDir = "/home/lauren/tickets/DM-24024/decam"
    rootDir = rootDir.rstrip("/")
    butler = dafPersist.Butler(rootDir)

    for filterName in filterList:
        inputDir = rootDir + "/plots/" + filterName
        if os.path.exists(inputDir):
            print("Reading wcsAnalysis tables from {}".format(inputDir))
            tractNames = os.listdir(inputDir)
            tractNames = [tractName for tractName in tractNames if "parq" not in tractName]
            tractIdList = [int(tractName[tractName.find("-") + 1:]) for tractName in tractNames]
            visitNamesPerTractDict = {tractName: os.listdir(inputDir + "/" + tractName)
                                      for tractName in tractNames}
            visitIdsPerTractDict = {}
            for i, (k, v) in enumerate(visitNamesPerTractDict.items()):
                visitIdsPerTractDict[str(tractIdList[i])] = [int(visitName[visitName.find("-") + 1:])
                                                             for visitName in v if "parq" not in visitName]
            wcsDfList = []
            for tract, visitList in visitIdsPerTractDict.items():
                tract = int(tract)
                for visit in visitList:
                    wcsTable = butler.get("wcsAnalysisVisitTable", tract=tract, visit=visit, filter=filterName,
                                          subdir="", immediate=True)
                    wcsDfList.append(wcsTable.toDataFrame())
            wcsDf = pd.concat(wcsDfList, axis=0, ignore_index=True)
            print("Writing output dataset for {}".format(filterName))
            butler.put(ParquetTable(dataFrame=wcsDf), "wcsAnalysisAllVisitsPerFilterTable", filter=filterName)
        else:
            print("NOTE: directory {} does not exist...skipping".format(inputDir))
