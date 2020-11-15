# This file is part of pipe_analysis.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import matplotlib
matplotlib.use("Agg")  # noqa 402
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

import copy
import pandas as pd
import numpy as np
from collections import defaultdict

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom
import lsst.geom as geom

from lsst.daf.persistence import NoResults
from lsst.pex.config import Config, Field
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct
from lsst.pipe.tasks.parquetTable import ParquetTable
from lsst.pipe.tasks.postprocess import VisitDataIdContainer
from .plotUtils import bboxToXyCoordLists
from .utils import findCcdKey


class WcsAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    doPlotWcsOutlines = Field(dtype=bool, default=True, doc="Make wcs outline plots?")


class WcsAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["tract"] = parsedCmd.tract
        if len(parsedCmd.id.refList) < 1:
            raise RuntimeWarning("refList from parsedCmd is empty...")
        visits = defaultdict(list)
        for refVisit in parsedCmd.id.refList:
            for ref in refVisit:
                ref.dataId["tract"] = int(parsedCmd.tract)  # update with tract for jointcal results
                ref.dataId["subdir"] = parsedCmd.subdir
                visits[ref.dataId["visit"]].append(ref)
        return [(visits[key], kwargs) for key in visits.keys()]


class WcsAnalysisTask(CmdLineTask):
    """Assess the accuracy of the raw WCS compared with the calibrated one.
    """
    _DefaultName = "wcsAnalysis"
    ConfigClass = WcsAnalysisConfig
    RunnerClass = WcsAnalysisRunner

    inputDataset = "raw"
    outputDataset = "wcsAnalysisVisitTable"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", cls.inputDataset, help="data ID, e.g. --id visit=12345",
                               ContainerClass=VisitDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/visit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList, **kwargs):
        self.log.info("dataRefList size: {:d}".format(len(dataRefList)))
        dataRefExistsList = [
            dataRef for dataRef in dataRefList if dataRef.datasetExists(self.inputDataset)]
        self.log.info("dataRefExistsList size: {:d}".format(len(dataRefExistsList)))
        butler = dataRefExistsList[0].getButler()
        rerun = list(butler.storage.repositoryCfgs)[0]
        camera = butler.get("camera")
        skyMap = dataRefExistsList[0].get(self.config.coaddName + "Coadd_skyMap")
        tract = int(kwargs["tract"])
        tractInfo = skyMap[tract]
        tractWcs = tractInfo.getWcs()
        visit = dataRefExistsList[0].dataId["visit"]
        filterName = dataRefExistsList[0].dataId["filter"]
        outputDataList = []
        tractCornersList = []
        focalPlaneCornersList = []
        tractCenterList = []
        focalPlaneCenterList = []
        dataRefStructTemplate = None
        # Get central 5% detectors for consideration for template dataRef
        distanceToFpCenterSquaredList = []
        cameraFpCenter = camera.getFpBBox().getCenter()
        for iCcd, ccd in enumerate(camera):
            ccdCenterFp = ccd.getCenter(cameraGeom.FOCAL_PLANE)
            distanceToFpCenterSquaredList.append(
                cameraFpCenter.distanceSquared(ccd.getCenter(cameraGeom.FOCAL_PLANE)))
        distanceToFpCenterSquaredList.sort()
        nCentralCcds = max(1, int(0.05*len(camera)))
        nthDistanceToFpCenter = distanceToFpCenterSquaredList[nCentralCcds - 1]
        for dataRef in dataRefExistsList:
            dataRefStruct = getDataRefInfo(dataRef, camera, log=self.log)
            if dataRefStruct is not None:
                ccdCenterFp = ccd.getCenter(cameraGeom.FOCAL_PLANE)
                distanceToFpCenterSquared = cameraFpCenter.distanceSquared(
                    dataRefStruct.ccd.getCenter(cameraGeom.FOCAL_PLANE))
                if distanceToFpCenterSquared <= nthDistanceToFpCenter:
                    if dataRefStructTemplate is None:
                        dataRefStructTemplate = dataRefStruct
                    else:
                        if dataRefStructTemplate.calexpWcs is None and dataRefStruct.calexpWcs is not None:
                            dataRefStructTemplate = dataRefStruct
                        if dataRefStructTemplate.jointcalWcs is None and dataRefStruct.jointcalWcs is not None:
                            dataRefStructTemplate = dataRefStruct
                rowEntry, tractCorners, focalPlaneCorners, tractCenter, focalPlaneCenter = (
                    self.run(camera, dataRefStruct, tract, tractWcs))
                outputDataList.append(rowEntry)
                tractCornersList.append(tractCorners)
                tractCenterList.append(tractCenter)
                focalPlaneCornersList.append(focalPlaneCorners)
                focalPlaneCenterList.append(focalPlaneCenter)
        if dataRefStructTemplate is None:  # Fall back to last dataRef read (needed for, e.g. ccd subsets)
            dataRefStructTemplate = dataRefStruct
        if dataRefStructTemplate is None:
            raise RuntimeError("No dataRefStructTemplate set from central 5%...")
        dfColumns = ["tract", "filter", "visit", "ccdId",
                     "rawTractCenterX", "rawTractCenterY",
                     "calexpTractCenterX", "calexpTractCenterY",
                     "calexpFpCenterVectorLength", "calexpFpCenterVectorTheta",
                     "jointcalTractCenterX", "jointcalTractCenterY",
                     "jointcalFpCenterVectorLength", "jointcalFpCenterVectorTheta",
                     "maxDiffTractPixCalexpX", "maxDiffTractPixCalexpY",
                     "maxDiffTractPixJointcalX", "maxDiffTractPixJointcalY",
                     "rawFocalPlaneCenterX", "rawFocalPlaneCenterY",
                     "calexpFocalPlaneCenterX", "calexpFocalPlaneCenterY",
                     "jointcalFocalPlaneCenterX", "jointcalFocalPlaneCenterY",
                     "maxDiffFocalPlanePixCalexpX", "maxDiffFocalPlanePixCalexpY",
                     "maxDiffFocalPlanePixJointcalX", "maxDiffFocalPlanePixJointcalY",
                     "fpSkyOriginRawRaDeg", "fpSkyOriginRawDecDeg",
                     "fpSkyOriginCalexpRaDeg", "fpSkyOriginCalexpDecDeg",
                     "fpSkyOriginJointcalRaDeg", "fpSkyOriginJointcalDecDeg",
                     "rawVsCalexpOffsetArcsec",
                     "rawVsJointcalOffsetArcsec",
                     "anyBadFlag", "rawBadFlag", "calexpBadFlag", "jointcalBadFlag", "airmass"]
        maxDiffTractCalexpXs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffTractPixCalexpX")].astype(np.float)
        maxDiffTractCalexpYs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffTractPixCalexpY")].astype(np.float)
        maxDiffTractJointcalXs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffTractPixJointcalX")].astype(np.float)
        maxDiffTractJointcalYs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffTractPixJointcalY")].astype(np.float)
        maxDiffFocalPlaneCalexpXs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffFocalPlanePixCalexpX")].astype(np.float)
        maxDiffFocalPlaneCalexpYs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffFocalPlanePixCalexpY")].astype(np.float)
        maxDiffFocalPlaneJointcalXs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffFocalPlanePixJointcalX")].astype(np.float)
        maxDiffFocalPlaneJointcalYs = np.array(outputDataList).transpose()[
            dfColumns.index("maxDiffFocalPlanePixJointcalY")].astype(np.float)
        calexpFpCenterVectorLengths = np.array(outputDataList).transpose()[
            dfColumns.index("calexpFpCenterVectorLength")].astype(np.float)
        jointcalFpCenterVectorLengths = np.array(outputDataList).transpose()[
            dfColumns.index("jointcalFpCenterVectorLength")].astype(np.float)
        calexpFpCenterVectorThetas = np.array(outputDataList).transpose()[
            dfColumns.index("calexpFpCenterVectorTheta")].astype(np.float)
        jointcalFpCenterVectorThetas = np.array(outputDataList).transpose()[
            dfColumns.index("jointcalFpCenterVectorTheta")].astype(np.float)

        anyBadFlags = np.array(outputDataList, dtype=bool).transpose()[dfColumns.index("anyBadFlag")]

        maxDiffTractCalexpXsGood = maxDiffTractCalexpXs[~anyBadFlags]
        maxDiffTractCalexpYsGood = maxDiffTractCalexpYs[~anyBadFlags]
        maxDiffTractJointcalXsGood = maxDiffTractJointcalXs[~anyBadFlags]
        maxDiffTractJointcalYsGood = maxDiffTractJointcalYs[~anyBadFlags]
        maxDiffFocalPlaneCalexpXsGood = maxDiffFocalPlaneCalexpXs[~anyBadFlags]
        maxDiffFocalPlaneCalexpYsGood = maxDiffFocalPlaneCalexpYs[~anyBadFlags]
        maxDiffFocalPlaneJointcalXsGood = maxDiffFocalPlaneJointcalXs[~anyBadFlags]
        maxDiffFocalPlaneJointcalYsGood = maxDiffFocalPlaneJointcalYs[~anyBadFlags]
        calexpFpCenterVectorLengthGood = calexpFpCenterVectorLengths[~anyBadFlags]
        jointcalFpCenterVectorLengthGood = jointcalFpCenterVectorLengths[~anyBadFlags]
        calexpFpCenterVectorThetaGood = calexpFpCenterVectorThetas[~anyBadFlags]
        jointcalFpCenterVectorThetaGood = jointcalFpCenterVectorThetas[~anyBadFlags]

        if all(np.isnan(maxDiffTractCalexpXs)):
            raise RuntimeError("No calexp data found...")
        haveJointcal = True if not all(np.isnan(maxDiffTractJointcalXs)) else False
        wcsStr = "jointcal" if haveJointcal else "calexp"
        if haveJointcal:
            xMaxInd = np.nanargmax(abs(maxDiffTractJointcalXsGood))
            ccdMaxDiffTractX = outputDataList[xMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffTractPixX = maxDiffTractJointcalXsGood[xMaxInd]
            visitMeanDiffTractPixX = np.nanmean(maxDiffTractJointcalXsGood)
            visitMedianDiffTractPixX = np.nanmedian(maxDiffTractJointcalXsGood)
            visitStdDiffTractPixX = np.nanstd(maxDiffTractJointcalXsGood)
            yMaxInd = np.nanargmax(abs(maxDiffTractJointcalYsGood))
            ccdMaxDiffTractY = outputDataList[yMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffTractPixY = maxDiffTractJointcalYsGood[yMaxInd]
            visitMeanDiffTractPixY = np.nanmean(maxDiffTractJointcalYsGood)
            visitMedianDiffTractPixY = np.nanmedian(maxDiffTractJointcalYsGood)
            visitStdDiffTractPixY = np.nanstd(maxDiffTractJointcalYsGood)

            xMaxInd = np.nanargmax(abs(maxDiffFocalPlaneJointcalXsGood))
            ccdMaxDiffFocalPlaneX = outputDataList[xMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffFocalPlanePixX = maxDiffFocalPlaneJointcalXsGood[xMaxInd]
            visitMeanDiffFocalPlanePixX = np.nanmean(maxDiffFocalPlaneJointcalXsGood)
            visitMedianDiffFocalPlanePixX = np.nanmedian(maxDiffFocalPlaneJointcalXsGood)
            visitStdDiffFocalPlanePixX = np.nanstd(maxDiffFocalPlaneJointcalXsGood)
            yMaxInd = np.nanargmax(abs(maxDiffFocalPlaneJointcalYsGood))
            ccdMaxDiffFocalPlaneY = outputDataList[yMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffFocalPlanePixY = maxDiffFocalPlaneJointcalYsGood[
                np.nanargmax(abs(maxDiffFocalPlaneJointcalYsGood))]
            visitMeanDiffFocalPlanePixY = np.nanmean(maxDiffFocalPlaneJointcalYsGood)
            visitMedianDiffFocalPlanePixY = np.nanmedian(maxDiffFocalPlaneJointcalYsGood)
            visitStdDiffFocalPlanePixY = np.nanstd(maxDiffFocalPlaneJointcalYsGood)
            focalPlanePixelScale = dataRefStructTemplate.jointcalWcs.getPixelScale(
                dataRefStructTemplate.ccd.getCenter(cameraGeom.PIXELS)).asArcseconds()

            maxVecInd = np.nanargmax(jointcalFpCenterVectorLengthGood)
            visitMaxVecLen = jointcalFpCenterVectorLengthGood[maxVecInd]
            visitMaxVecTheta = jointcalFpCenterVectorThetaGood[maxVecInd]
            ccdMaxVec = outputDataList[maxVecInd][dfColumns.index("ccdId")]
        else:
            xMaxInd = np.nanargmax(abs(maxDiffTractCalexpXsGood))
            ccdMaxDiffTractX = outputDataList[xMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffTractPixX = maxDiffTractCalexpXsGood[np.nanargmax(abs(maxDiffTractCalexpXsGood))]
            visitMeanDiffTractPixX = np.nanmean(maxDiffTractCalexpXsGood)
            visitMedianDiffTractPixX = np.nanmedian(maxDiffTractCalexpXsGood)
            visitStdDiffTractPixX = np.nanstd(maxDiffTractCalexpXsGood)
            yMaxInd = np.nanargmax(abs(maxDiffTractCalexpYsGood))
            ccdMaxDiffTractY = outputDataList[yMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffTractPixY = maxDiffTractCalexpYsGood[np.nanargmax(abs(maxDiffTractCalexpYsGood))]
            visitMeanDiffTractPixY = np.nanmean(maxDiffTractCalexpYsGood)
            visitMedianDiffTractPixY = np.nanmedian(maxDiffTractCalexpYsGood)
            visitStdDiffTractPixY = np.nanstd(maxDiffTractCalexpYsGood)

            xMaxInd = np.nanargmax(abs(maxDiffFocalPlaneCalexpXsGood))
            ccdMaxDiffFocalPlaneX = outputDataList[xMaxInd][dfColumns.index("ccdId")]
            visitMaxDiffFocalPlanePixX = maxDiffFocalPlaneCalexpXsGood[
                np.nanargmax(abs(maxDiffFocalPlaneCalexpXsGood))]
            visitMeanDiffFocalPlanePixX = np.nanmean(maxDiffFocalPlaneCalexpXsGood)
            visitMedianDiffFocalPlanePixX = np.nanmedian(maxDiffFocalPlaneCalexpXsGood)
            visitStdDiffFocalPlanePixX = np.nanstd(maxDiffFocalPlaneCalexpXsGood)
            visitMaxDiffFocalPlanePixY = maxDiffFocalPlaneCalexpYsGood[
                np.nanargmax(abs(maxDiffFocalPlaneCalexpYsGood))]
            yMaxInd = np.nanargmax(abs(maxDiffFocalPlaneCalexpYsGood))
            ccdMaxDiffFocalPlaneY = outputDataList[yMaxInd][dfColumns.index("ccdId")]
            visitMeanDiffFocalPlanePixY = np.nanmean(maxDiffFocalPlaneCalexpYsGood)
            visitMedianDiffFocalPlanePixY = np.nanmedian(maxDiffFocalPlaneCalexpYsGood)
            visitStdDiffFocalPlanePixY = np.nanstd(maxDiffFocalPlaneCalexpYsGood)
            focalPlanePixelScale = dataRefStructTemplate.calexpWcs.getPixelScale(
                dataRefStructTemplate.ccd.getCenter(cameraGeom.PIXELS)).asArcseconds()
            maxVecInd = np.nanargmax(calexpFpCenterVectorLengthGood)
            visitMaxVecLen = calexpFpCenterVectorLengthGood[maxVecInd]
            visitMaxVecTheta = calexpFpCenterVectorThetaGood[maxVecInd]
            ccdMaxVec = outputDataList[maxVecInd][dfColumns.index("ccdId")]
        tractPixelScale = tractWcs.getPixelScale().asArcseconds()
        self.log.info("Maximum {}-raw pixel offset ({:.3f} arcsec/pixel) for tract: {} visit: {}: "
                      "filter: {}  xyDiffTract      Max [ccd]: {:8.2f} [{:3d}] {:8.2f} [{:3d}]  "
                      "Median: {:8.2f} {:8.2f}".
                      format(wcsStr, tractPixelScale, tract, visit, filterName,
                             visitMaxDiffTractPixX, ccdMaxDiffTractX,
                             visitMaxDiffTractPixY, ccdMaxDiffTractY,
                             visitMedianDiffTractPixX, visitMedianDiffTractPixY))
        self.log.info("Maximum {}-raw pixel offset ({:.3f} arcsec/pixel) for tract: {} visit: {}: "
                      "filter: {}  xyDiffFocalPlane Max [ccd]: {:8.2f} [{:3d}] {:8.2f} [{:3d}]  "
                      "Median: {:8.2f} {:8.2f}".
                      format(wcsStr, focalPlanePixelScale, tract, visit, filterName,
                             visitMaxDiffFocalPlanePixX, ccdMaxDiffFocalPlaneX,
                             visitMaxDiffFocalPlanePixY, ccdMaxDiffFocalPlaneY,
                             visitMedianDiffFocalPlanePixX, visitMedianDiffFocalPlanePixY))
        self.log.info("Center {}-raw Focal Plane pixel vector ({:.3f} arcsec/pixel) for tract: {} "
                      "visit: {}: filter: {}  length, theta (deg) [ccd]: {:8.2f} {:8.2f} [{:3d}]".
                      format(wcsStr, focalPlanePixelScale, tract, visit, filterName,
                             visitMaxVecLen, visitMaxVecTheta, ccdMaxVec))
        outputDf = pd.DataFrame(np.array(outputDataList, dtype=object), columns=dfColumns, dtype=object)
        dataRefList[0].put(ParquetTable(dataFrame=outputDf), self.outputDataset)
        if self.config.doPlotWcsOutlines:
            plotTitle = "camera: {:}  tract: {:}  visit: {:} filter: {:}".format(
                str(camera.getName()), str(tract), visit, filterName)
            plotMaxStr = "max {}-raw offset x, y (pix): {:7.2f} {:7.2f}".format(
                wcsStr, visitMaxDiffTractPixX, visitMaxDiffTractPixY)
            plotMedStr = "med {}-raw offset x, y (pix): {:7.2f} {:7.2f}".format(
                wcsStr, visitMedianDiffTractPixX, visitMedianDiffTractPixY)
            tractPlot = plotWcsOutlines(tractCornersList, tractCenterList, plotTitle=plotTitle, rerun=rerun,
                                        tractInfo=tractInfo, plotMaxStr=plotMaxStr, plotMedStr=plotMedStr,
                                        log=self.log, ccdMaxX=ccdMaxDiffTractX, ccdMaxY=ccdMaxDiffTractY)
            dataRefList[0].put(tractPlot, "plotWcsOutlines", description="wcsOutlines", style="tract")
            plt.close(tractPlot)

            plotMaxStr = "max {}-raw offset x, y (pix): {:7.2f} {:7.2f}".format(
                wcsStr, visitMaxDiffFocalPlanePixX, visitMaxDiffFocalPlanePixY)
            plotMedStr = "med {}-raw offset x, y (pix): {:7.2f} {:7.2f}".format(
                wcsStr, visitMedianDiffFocalPlanePixX, visitMedianDiffFocalPlanePixY)
            focalPlanePlot = plotWcsOutlinesFocalPlane(
                focalPlaneCornersList, focalPlaneCenterList, dataRefStructTemplate.rawWcs, plotTitle=plotTitle,
                rerun=rerun, tractInfo=tractInfo, plotMaxStr=plotMaxStr, plotMedStr=plotMedStr, log=self.log,
                ccdMaxX=ccdMaxDiffFocalPlaneX, ccdMaxY=ccdMaxDiffFocalPlaneY, ccdMaxVec=ccdMaxVec)
            dataRefList[0].put(focalPlanePlot, "plotWcsOutlines", description="wcsOutlines",
                               style="focalPlane")
            plt.close(focalPlanePlot)

    def run(self, camera, dataRefStruct, tract, tractWcs):
        self.log.debug("Processing visit {:} ccd {:}".format(dataRefStruct.visit, dataRefStruct.ccdId))
        offsetsStruct = computeWcsOffsets(camera, dataRefStruct.ccd, dataRefStruct.bbox, tractWcs,
                                          dataRefStruct.rawWcs, calexpWcs=dataRefStruct.calexpWcs,
                                          jointcalWcs=dataRefStruct.jointcalWcs)
        rowEntry = [tract, dataRefStruct.filterName, dataRefStruct.visit, dataRefStruct.ccdId,
                    offsetsStruct.rawTractCenter[0], offsetsStruct.rawTractCenter[1],
                    offsetsStruct.calexpTractCenter[0], offsetsStruct.calexpTractCenter[1],
                    offsetsStruct.calexpFpCenterVectorLength, offsetsStruct.calexpFpCenterVectorTheta,
                    offsetsStruct.jointcalTractCenter[0], offsetsStruct.jointcalTractCenter[1],
                    offsetsStruct.jointcalFpCenterVectorLength, offsetsStruct.jointcalFpCenterVectorTheta,
                    offsetsStruct.maxDiffTractPixCalexpX, offsetsStruct.maxDiffTractPixCalexpY,
                    offsetsStruct.maxDiffTractPixJointcalX, offsetsStruct.maxDiffTractPixJointcalY,

                    offsetsStruct.rawFocalPlaneCenter[0], offsetsStruct.rawFocalPlaneCenter[1],
                    offsetsStruct.calexpFocalPlaneCenter[0], offsetsStruct.calexpFocalPlaneCenter[1],
                    offsetsStruct.jointcalFocalPlaneCenter[0], offsetsStruct.jointcalFocalPlaneCenter[1],
                    offsetsStruct.maxDiffFocalPlanePixCalexpX, offsetsStruct.maxDiffFocalPlanePixCalexpY,
                    offsetsStruct.maxDiffFocalPlanePixJointcalX, offsetsStruct.maxDiffFocalPlanePixJointcalY,

                    offsetsStruct.fpSkyOriginRaw.getRa().asDegrees(),
                    offsetsStruct.fpSkyOriginRaw.getDec().asDegrees(),
                    offsetsStruct.fpSkyOriginCalexp.getRa().asDegrees(),
                    offsetsStruct.fpSkyOriginCalexp.getDec().asDegrees(),
                    offsetsStruct.fpSkyOriginJointcal.getRa().asDegrees(),
                    offsetsStruct.fpSkyOriginJointcal.getDec().asDegrees(),
                    # offsetsStruct.rawVsCalexpOffsetRadians, offsetsStruct.rawVsCalexpOffsetDegrees,
                    offsetsStruct.rawVsCalexpOffsetArcsec,
                    # offsetsStruct.rawVsJointcalOffsetRadians, offsetsStruct.rawVsJointcalOffsetDegrees,
                    offsetsStruct.rawVsJointcalOffsetArcsec,
                    offsetsStruct.anyBadFlag, offsetsStruct.rawBadFlag, offsetsStruct.calexpBadFlag,
                    offsetsStruct.jointcalBadFlag, dataRefStruct.airmass]
        tractCorners = [dataRefStruct.ccdId, offsetsStruct.rawTractCorners,
                        offsetsStruct.calexpTractCorners, offsetsStruct.jointcalTractCorners]
        focalPlaneCorners = [dataRefStruct.ccdId, offsetsStruct.rawFocalPlaneCorners,
                             offsetsStruct.calexpFocalPlaneCorners, offsetsStruct.jointcalFocalPlaneCorners]
        if offsetsStruct.anyBadFlag:
            self.log.warn("Crazy polygon for tract: {:} filter: {:} visit: {:} ccd: {:} "
                          "...flag and bypass this one!".format(tract, dataRefStruct.filterName,
                                                                dataRefStruct.visit, dataRefStruct.ccdId))
        return (rowEntry, tractCorners, focalPlaneCorners, offsetsStruct.rawTractCenter,
                offsetsStruct.rawFocalPlaneCenter)

    def writeMetadata(self, dataRef):
        """No metadata to write.
        """
        pass

    def writeConfig(self, butler, clobber=False, doBackup=True):
        """No config to write.
        """
        pass


def getDataRefInfo(dataRef, camera, log=None):
    ccdKey = findCcdKey(dataRef.dataId)
    ccdId = dataRef.dataId[ccdKey]
    ccd = camera[ccdId]
    if ccd.getType() != cameraGeom.DetectorType.SCIENCE:
        return None
    tract = dataRef.dataId["tract"]
    visit = dataRef.dataId["visit"]
    filterName = dataRef.dataId["filter"]
    bbox = geom.Box2D(ccd.getBBox())
    raw = dataRef.get("raw")
    rawWcs = raw.getWcs()
    airmass = raw.getInfo().getVisitInfo().getBoresightAirmass()
    try:
        calexpWcs = dataRef.get("calexp_wcs")
    except NoResults:
        if log is not None:
            log.info("No calexp wcs found for tract: {} visit: {} filter: {} ccd: {}".
                     format(tract, visit, filterName, ccdId))
        calexpWcs = None
    try:
        jointcalWcs = dataRef.get("jointcal_wcs")
    except NoResults:
        if log is not None:
            log.info("No jointcal wcs found for tract: {} visit: {} filter: {} ccd: {}".
                     format(tract, visit, filterName, ccdId))
        jointcalWcs = None
    return Struct(
        visit=visit,
        filterName=filterName,
        ccdId=ccdId,
        ccd=ccd,
        airmass=airmass,
        bbox=bbox,
        rawWcs=rawWcs,
        calexpWcs=calexpWcs,
        jointcalWcs=jointcalWcs,
    )


def computePointingOffset(ccd, rawWcs, calexpWcs=None, jointcalWcs=None):
    fpSkyOriginRaw = computeFpSkyOrigin(ccd, rawWcs)
    if calexpWcs is not None:
        fpSkyOriginCalexp = computeFpSkyOrigin(ccd, calexpWcs)
        rawVsCalexpOffsetRadians = computeAngularDistance(fpSkyOriginRaw.getRa().asRadians(),
                                                          fpSkyOriginRaw.getDec().asRadians(),
                                                          fpSkyOriginCalexp.getRa().asRadians(),
                                                          fpSkyOriginCalexp.getDec().asRadians())
        rawVsCalexpOffsetDegrees = geom.radToDeg(rawVsCalexpOffsetRadians)
        rawVsCalexpOffsetArcsec = geom.radToArcsec(rawVsCalexpOffsetRadians)
    else:
        fpSkyOriginCalexp = geom.SpherePoint(np.nan*geom.degrees, np.nan*geom.degrees)
        rawVsCalexpOffsetRadians = np.nan
        rawVsCalexpOffsetDegrees = np.nan
        rawVsCalexpOffsetArcsec = np.nan
    if jointcalWcs is not None:
        fpSkyOriginJointcal = computeFpSkyOrigin(ccd, jointcalWcs)
        rawVsJointcalOffsetRadians = computeAngularDistance(fpSkyOriginRaw.getRa().asRadians(),
                                                            fpSkyOriginRaw.getDec().asRadians(),
                                                            fpSkyOriginJointcal.getRa().asRadians(),
                                                            fpSkyOriginJointcal.getDec().asRadians())
        rawVsJointcalOffsetDegrees = geom.radToDeg(rawVsJointcalOffsetRadians)
        rawVsJointcalOffsetArcsec = geom.radToArcsec(rawVsJointcalOffsetRadians)
    else:
        fpSkyOriginJointcal = geom.SpherePoint(np.nan*geom.degrees, np.nan*geom.degrees)
        rawVsJointcalOffsetRadians = np.nan
        rawVsJointcalOffsetDegrees = np.nan
        rawVsJointcalOffsetArcsec = np.nan
    return Struct(
        fpSkyOriginRaw=fpSkyOriginRaw,
        fpSkyOriginCalexp=fpSkyOriginCalexp,
        fpSkyOriginJointcal=fpSkyOriginJointcal,
        rawVsCalexpOffsetRadians=rawVsCalexpOffsetRadians,
        rawVsCalexpOffsetDegrees=rawVsCalexpOffsetDegrees,
        rawVsCalexpOffsetArcsec=rawVsCalexpOffsetArcsec,
        rawVsJointcalOffsetRadians=rawVsJointcalOffsetRadians,
        rawVsJointcalOffsetDegrees=rawVsJointcalOffsetDegrees,
        rawVsJointcalOffsetArcsec=rawVsJointcalOffsetArcsec,
    )


def computeFpSkyOrigin(ccd, wcs):
    focalPlaneToPixels = ccd.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    fpPixelOrigin = focalPlaneToPixels.applyForward(geom.Point2D(0.0, 0.0))
    fpSkyOrigin = wcs.pixelToSky(fpPixelOrigin)
    return fpSkyOrigin

def computeSkyToFpPixel(ccd, wcs, skyCoordList):
    fpToSky = ccd.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS).then(wcs.getTransform())
    fpMm = fpToSky.applyInverse(skyCoordList)
    fpPixel = copy.deepcopy(fpMm)
    if not isinstance(fpPixel, list):
        fpPixel = [fpPixel]
    for point in fpPixel:
        point[0] = point[0]/ccd.getPixelSize().getX()
        point[1] = point[1]/ccd.getPixelSize().getY()
    fpPixel = fpPixel[0] if len(fpPixel) == 1 else fpPixel
    return fpPixel

def computeCoordToDistortedFpPixel(camera, ccd, coordList, wcs=None):
    focalPlaneToFieldAngle = camera.getTransformMap().getTransform(cameraGeom.FOCAL_PLANE,
                                                                   cameraGeom.FIELD_ANGLE)
    pixelToFocalPlane = ccd.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    pixelToDistortedPixel = afwGeom.wcsUtils.computePixelToDistortedPixel(
        pixelToFocalPlane=pixelToFocalPlane, focalPlaneToFieldAngle=focalPlaneToFieldAngle)

    if wcs is not None:
        fpToSky = ccd.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS).then(wcs.getTransform())
        fpMm = fpToSky.applyInverse(coordList)
        pixels = pixelToFocalPlane.applyInverse(fpMm)
    else:
        pixels = coordList

    distortedPixels = pixelToDistortedPixel.applyForward(pixels)
    distortedFpMm = pixelToFocalPlane.applyForward(distortedPixels)
    distortedFpPixel = copy.deepcopy(distortedFpMm)
    if not isinstance(distortedFpPixel, list):
        distortedFpPixel = [distortedFpPixel]
    for point in distortedFpPixel:
        point[0] = point[0]/ccd.getPixelSize().getX()
        point[1] = point[1]/ccd.getPixelSize().getY()
    distortedFpPixel = distortedFpPixel[0] if len(distortedFpPixel) == 1 else distortedFpPixel
    return distortedFpPixel

def computePixelToFpPixel(ccd, pixCoordList):
    pixelToFocalPlane = ccd.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpMm = pixelToFocalPlane.applyForward(pixCoordList)
    fpPixel = copy.deepcopy(fpMm)
    if not isinstance(fpPixel, list):
        fpPixel = [fpPixel]
    for point in fpPixel:
        point[0] = point[0]/ccd.getPixelSize().getX()
        point[1] = point[1]/ccd.getPixelSize().getY()
    fpPixel = fpPixel[0] if len(fpPixel) == 1 else fpPixel
    return fpPixel


def computeWcsOffsets(camera, ccd, bbox, tractWcs, rawWcs, calexpWcs=None, jointcalWcs=None):
    rawBadFlag, calexpBadFlag, jointcalBadFlag = False, False, False
    ccdCorners = bbox.getCorners()
    ccdCenter = bbox.getCenter()
    # First compute corner differences in Tract pixel coordinates
    maxDiffTractPixCalexpX, maxDiffTractPixCalexpY = 1e-12, 1e-12  # in pixels
    maxDiffTractPixJointcalX, maxDiffTractPixJointcalY = 1e-12, 1e-12  # in pixels
    rawTractCenter = tractWcs.skyToPixel(rawWcs.pixelToSky(ccdCenter))
    rawTractCorners = tractWcs.skyToPixel(rawWcs.pixelToSky(ccdCorners))
    rawTractCorners, rawBadFlag = orderCoordsToCc(rawTractCorners)
    calexpTractCenter = geom.Point2D(np.nan, np.nan)
    jointcalTractCenter = geom.Point2D(np.nan, np.nan)
    calexpTractCorners = [geom.Point2D(np.nan, np.nan)]*4
    jointcalTractCorners = [geom.Point2D(np.nan, np.nan)]*4
    if calexpWcs is not None or jointcalWcs is not None:
        if calexpWcs is not None:
            calexpTractCenter = tractWcs.skyToPixel(calexpWcs.pixelToSky(ccdCenter))
            calexpTractCorners = tractWcs.skyToPixel(calexpWcs.pixelToSky(ccdCorners))
            calexpTractCorners, calexpBadFlag = orderCoordsToCc(calexpTractCorners, ccdId=ccd.getId())
            for rawPoint, calexpPoint in zip(rawTractCorners, calexpTractCorners):
                diffTractRawCalexp = calexpPoint - rawPoint
                if not calexpBadFlag:
                    maxDiffTractPixCalexpX = (diffTractRawCalexp[0] if
                                              abs(diffTractRawCalexp[0]) > abs(maxDiffTractPixCalexpX)
                                              else maxDiffTractPixCalexpX)
                    maxDiffTractPixCalexpY = (diffTractRawCalexp[1] if
                                              abs(diffTractRawCalexp[1]) > abs(maxDiffTractPixCalexpY)
                                              else maxDiffTractPixCalexpY)
        else:
            maxDiffTractPixCalexpX, maxDiffTractPixCalexpY = np.nan, np.nan
        if jointcalWcs is not None:
            jointcalTractCenter = tractWcs.skyToPixel(jointcalWcs.pixelToSky(ccdCenter))
            jointcalTractCorners = tractWcs.skyToPixel(jointcalWcs.pixelToSky(ccdCorners))
            jointcalTractCorners, jointcalBadFlag = orderCoordsToCc(jointcalTractCorners, ccdId=ccd.getId())
            for rawPoint, jointcalPoint in zip(rawTractCorners, jointcalTractCorners):
                diffTractRawJointcal = jointcalPoint - rawPoint
                if not jointcalBadFlag:
                    maxDiffTractPixJointcalX = (diffTractRawJointcal[0] if
                                                abs(diffTractRawJointcal[0]) > abs(maxDiffTractPixJointcalX)
                                                else maxDiffTractPixJointcalX)
                    maxDiffTractPixJointcalY = (diffTractRawJointcal[1] if
                                                abs(diffTractRawJointcal[1]) > abs(maxDiffTractPixJointcalY)
                                                else maxDiffTractPixJointcalY)
        else:
            maxDiffTractPixJointcalX, maxDiffTractPixJointcalY = np.nan, np.nan
    else:
        maxDiffTractPixCalexpX, maxDiffTractPixCalexpY = np.nan, np.nan
        maxDiffTractPixJointcalX, maxDiffTractPixJointcalY = np.nan, np.nan

    # Now compute corner differences in Focal Plane pixel coordinates
    maxDiffFocalPlanePixCalexpX, maxDiffFocalPlanePixCalexpY = 1e-12, 1e-12  # in pixels
    maxDiffFocalPlanePixJointcalX, maxDiffFocalPlanePixJointcalY = 1e-12, 1e-12  # in pixels
    rawSkyCenter = rawWcs.pixelToSky(ccd.getCenter(cameraGeom.PIXELS))
    rawSkyCorners = rawWcs.pixelToSky(ccd.getCorners(cameraGeom.PIXELS))

    ccdFocalPlaneCenter = computePixelToFpPixel(ccd, ccd.getCenter(cameraGeom.PIXELS))
    ccdFocalPlaneCorners = computePixelToFpPixel(ccd, ccd.getCorners(cameraGeom.PIXELS))
    ccdFocalPlaneCorners, rawBadFlag = orderCoordsToCc(ccdFocalPlaneCorners)
    ccdDistortedFpCenter = computeCoordToDistortedFpPixel(camera, ccd, ccd.getCenter(cameraGeom.PIXELS),
                                                          wcs=None)
    ccdDistortedCornersFp = computeCoordToDistortedFpPixel(camera, ccd, ccd.getCorners(cameraGeom.PIXELS),
                                                           wcs=None)
    # rawFocalPlaneCenter = computeSkyToFpPixel(ccd, rawWcs, rawSkyCenter)
    # rawFocalPlaneCorners = computeSkyToFpPixel(ccd, rawWcs, rawSkyCorners)
    # rawFocalPlaneCorners, rawBadFlag = orderCoordsToCc(rawFocalPlaneCorners)
    rawDistortedFpCenter = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCenter, wcs=rawWcs)
    rawDistortedFpCorners = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCorners, wcs=rawWcs)
    rawDistortedFpCorners, rawBadFlag = orderCoordsToCc(rawDistortedFpCorners)
    calexpDistortedFpCenter = geom.Point2D(np.nan, np.nan)
    jointcalDistortedFpCenter = geom.Point2D(np.nan, np.nan)
    calexpDistortedFpCorners = [geom.Point2D(np.nan, np.nan)]*4
    jointcalDistortedFpCorners = [geom.Point2D(np.nan, np.nan)]*4
    calexpFpCenterVectorLength, calexpFpCenterVectorTheta = np.nan, np.nan
    jointcalFpCenterVectorLength, jointcalFpCenterVectorTheta = np.nan, np.nan

    if calexpWcs is not None or jointcalWcs is not None:
        if calexpWcs is not None:
            # calexpPixelCenter = calexpWcs.skyToPixel(rawSkyCenter)
            # calexpFocalPlaneCenter = computePixelToFpPixel(ccd, calexpPixelCenter)
            # calexpPixelCorners = calexpWcs.skyToPixel(rawSkyCorners)
            # calexpFocalPlaneCorners = computePixelToFpPixel(ccd, calexpPixelCorners)
            # calexpFocalPlaneCorners, calexpBadFlag = orderCoordsToCc(calexpFocalPlaneCorners,
            #                                                          ccdId=ccd.getId())
            calexpDistortedFpCenter = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCenter, wcs=calexpWcs)
            diffFpCenterX = calexpDistortedFpCenter[0] - rawDistortedFpCenter[0]
            diffFpCenterY = calexpDistortedFpCenter[1] - rawDistortedFpCenter[1]
            calexpFpCenterVectorLength = np.sqrt(diffFpCenterX**2.0 + diffFpCenterY**2.0)
            calexpFpCenterVectorTheta = np.rad2deg(np.math.atan2(diffFpCenterX, diffFpCenterY))
            calexpDistortedFpCorners = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCorners,
                                                                      wcs=calexpWcs)
            calexpDistortedFpCorners, calexpBadFlag = orderCoordsToCc(calexpDistortedFpCorners,
                                                                      ccdId=ccd.getId())

            for rawPoint, calexpPoint in zip(rawDistortedFpCorners, calexpDistortedFpCorners):
                diffFocalPlaneRawCalexp = calexpPoint - rawPoint
                if not calexpBadFlag:
                    maxDiffFocalPlanePixCalexpX = (
                        diffFocalPlaneRawCalexp[0] if
                        abs(diffFocalPlaneRawCalexp[0]) > abs(maxDiffFocalPlanePixCalexpX)
                        else maxDiffFocalPlanePixCalexpX)
                    maxDiffFocalPlanePixCalexpY = (
                        diffFocalPlaneRawCalexp[1] if
                        abs(diffFocalPlaneRawCalexp[1]) > abs(maxDiffFocalPlanePixCalexpY)
                        else maxDiffFocalPlanePixCalexpY)

        else:
            maxDiffFocalPlanePixCalexpX, maxDiffFocalPlanePixCalexpY = np.nan, np.nan
        if jointcalWcs is not None:
            # jointcalPixelCenter = jointcalWcs.skyToPixel(rawSkyCenter)
            # jointcalFocalPlaneCenter = computePixelToFpPixel(ccd, jointcalPixelCenter)
            # jointcalPixelCorners = jointcalWcs.skyToPixel(rawSkyCorners)
            # jointcalFocalPlaneCorners = computePixelToFpPixel(ccd, jointcalPixelCorners)
            # jointcalFocalPlaneCorners, jointcalBadFlag = orderCoordsToCc(jointcalFocalPlaneCorners,
            #                                                             ccdId=ccd.getId())
            jointcalDistortedFpCenter = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCenter,
                                                                       wcs=jointcalWcs)
            diffFpCenterX = jointcalDistortedFpCenter[0] - rawDistortedFpCenter[0]
            diffFpCenterY = jointcalDistortedFpCenter[1] - rawDistortedFpCenter[1]
            jointcalFpCenterVectorLength = np.sqrt(diffFpCenterX**2.0 + diffFpCenterY**2.0)
            jointcalFpCenterVectorTheta = np.rad2deg(np.math.atan2(diffFpCenterX, diffFpCenterY))
            jointcalDistortedFpCorners = computeCoordToDistortedFpPixel(camera, ccd, rawSkyCorners,
                                                                        wcs=jointcalWcs)
            jointcalDistortedFpCorners, jointcalBadFlag = orderCoordsToCc(jointcalDistortedFpCorners,
                                                                          ccdId=ccd.getId())

            for rawPoint, jointcalPoint in zip(rawDistortedFpCorners, jointcalDistortedFpCorners):
                diffFocalPlaneRawJointcal = jointcalPoint - rawPoint
                if not jointcalBadFlag:
                    maxDiffFocalPlanePixJointcalX = (
                        diffFocalPlaneRawJointcal[0] if
                        abs(diffFocalPlaneRawJointcal[0]) > abs(maxDiffFocalPlanePixJointcalX)
                        else maxDiffFocalPlanePixJointcalX)
                    maxDiffFocalPlanePixJointcalY = (
                        diffFocalPlaneRawJointcal[1] if
                        abs(diffFocalPlaneRawJointcal[1]) > abs(maxDiffFocalPlanePixJointcalY)
                        else maxDiffFocalPlanePixJointcalY)

        else:
            maxDiffFocalPlanePixJointcalX, maxDiffFocalPlanePixJointcalY = np.nan, np.nan
    else:
        maxDiffFocalPlanePixCalexpX, maxDiffFocalPlanePixCalexpY = np.nan, np.nan
        maxDiffFocalPlanePixJointcalX, maxDiffFocalPlanePixJointcalY = np.nan, np.nan

    pointingOffsetStruct = computePointingOffset(ccd, rawWcs, calexpWcs=calexpWcs, jointcalWcs=jointcalWcs)
    anyBadFlag = rawBadFlag or calexpBadFlag or jointcalBadFlag

    return Struct(
        rawTractCorners=rawTractCorners,
        rawTractCenter=rawTractCenter,
        calexpTractCenter=calexpTractCenter,
        calexpTractCorners=calexpTractCorners,
        jointcalTractCenter=jointcalTractCenter,
        jointcalTractCorners=jointcalTractCorners,
        maxDiffTractPixCalexpX=maxDiffTractPixCalexpX,
        maxDiffTractPixCalexpY=maxDiffTractPixCalexpY,
        maxDiffTractPixJointcalX=maxDiffTractPixJointcalX,
        maxDiffTractPixJointcalY=maxDiffTractPixJointcalY,
        rawFocalPlaneCorners=rawDistortedFpCorners,
        rawFocalPlaneCenter=rawDistortedFpCenter,
        calexpFocalPlaneCenter=calexpDistortedFpCenter,
        calexpFocalPlaneCorners=calexpDistortedFpCorners,
        jointcalFocalPlaneCenter=jointcalDistortedFpCenter,
        jointcalFocalPlaneCorners=jointcalDistortedFpCorners,
        maxDiffFocalPlanePixCalexpX=maxDiffFocalPlanePixCalexpX,
        maxDiffFocalPlanePixCalexpY=maxDiffFocalPlanePixCalexpY,
        maxDiffFocalPlanePixJointcalX=maxDiffFocalPlanePixJointcalX,
        maxDiffFocalPlanePixJointcalY=maxDiffFocalPlanePixJointcalY,
        fpSkyOriginRaw=pointingOffsetStruct.fpSkyOriginRaw,
        fpSkyOriginCalexp=pointingOffsetStruct.fpSkyOriginCalexp,
        fpSkyOriginJointcal=pointingOffsetStruct.fpSkyOriginJointcal,
        rawVsCalexpOffsetRadians=pointingOffsetStruct.rawVsCalexpOffsetRadians,
        rawVsCalexpOffsetDegrees=pointingOffsetStruct.rawVsCalexpOffsetDegrees,
        rawVsCalexpOffsetArcsec=pointingOffsetStruct.rawVsCalexpOffsetArcsec,
        rawVsJointcalOffsetRadians=pointingOffsetStruct.rawVsJointcalOffsetRadians,
        rawVsJointcalOffsetDegrees=pointingOffsetStruct.rawVsJointcalOffsetDegrees,
        rawVsJointcalOffsetArcsec=pointingOffsetStruct.rawVsJointcalOffsetArcsec,
        calexpFpCenterVectorLength=calexpFpCenterVectorLength,
        calexpFpCenterVectorTheta=calexpFpCenterVectorTheta,
        jointcalFpCenterVectorLength=jointcalFpCenterVectorLength,
        jointcalFpCenterVectorTheta=jointcalFpCenterVectorTheta,
        anyBadFlag=anyBadFlag,
        rawBadFlag=rawBadFlag,
        calexpBadFlag=calexpBadFlag,
        jointcalBadFlag=jointcalBadFlag,
    )


def orderCoordsToCc(corners, ccdId=None):
    """Return list of pixel corners in counter clockwise from LL order
    """
    badFlag = False
    xs, ys = zip(*corners)
    xMid = np.min(xs) + 0.5*(np.max(xs) - np.min(xs))
    yMid = np.min(ys) + 0.5*(np.max(ys) - np.min(ys))
    xLL, yLL, xUL, yUL, xLR, yLR, xUR, yUR = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    for xi, yi in zip(xs, ys):
        if xi < xMid and yi < yMid:
            xLL, yLL = xi, yi
        if xi < xMid and yi > yMid:
            xUL, yUL = xi, yi
        if xi > xMid and yi < yMid:
            xLR, yLR = xi, yi
        if xi > xMid and yi > yMid:
            xUR, yUR = xi, yi
    if any(np.isnan([xLL, yLL, xUL, yUL, xLR, yLR, xUR, yUR])):
        badFlag = True
    orderedCorners = [geom.Point2D(xLL, yLL), geom.Point2D(xLR, yLR), geom.Point2D(xUR, yUR),
                      geom.Point2D(xUL, yUL)]
    return orderedCorners, badFlag


def computeAngularDistance(ra1, dec1, ra2, dec2):
    """Calculate the Haversine angular distance between two points.

    The Haversine formula, which determines the great-circle distance between
    two points on a sphere given their longitudes (ra) and latitudes (dec), is
    given by:
    distance =
    2*arcsin(
       sqrt(sin**2((dec2-dec1)/2) + cos(del1)cos(del2)sin**2((ra1-ra2)/2)))

    Parameters
    ----------
    ra1, dec1 : `float`
       The RA and Dec (in radians) of the first point.
    ra1, dec1 : `float`
       The RA and Dec (in radians) of the second point.

    Returns
    -------
    angularDistance : `float`
       The Haversine angular distance (in radians) between the points.
    """
    deltaRa = ra1 - ra2
    deltaDec = dec1 - dec2
    haverDeltaRa = np.sin(deltaRa/2.00)
    haverDeltaDec = np.sin(deltaDec/2.00)
    haverAlpha = np.sqrt(np.square(haverDeltaDec) + np.cos(dec1)*np.cos(dec2)*np.square(haverDeltaRa))
    angularDistance = 2.0*np.arcsin(haverAlpha)
    return angularDistance


def plotWcsOutlines(cornersList, centerList, plotTitle=None, rerun=None, tractInfo=None,
                    plotMaxStr=None, plotMedStr=None, log=None, ccdMaxX=-99, ccdMaxY=-99):
    """Plot the outlines of the ccds in ccdList for various wcs.

    The plots are in Tract pixels.
    """
    xaxLimMin, xaxLimMax = 1e12, -1e12
    yaxLimMin, yaxLimMax = 1e12, -1e12
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)
    ccdIds = []
    rawCorners = []
    calexpCorners = []
    jointcalCorners = []
    wcsList = ["rawWcs", "calexpWcs", "jointcalWcs"]
    lineColorDict = {"rawWcs": "palevioletred", "calexpWcs": "teal", "jointcalWcs": "black"}
    legendPatches = []

    def convertCorners(corners):
        minX, maxX = 1e12, -1e12
        minY, maxY = 1e12, -1e12
        xyList = []
        if corners is not None:
            for corner in corners:
                maxX = corner.x if corner.x > maxX else maxX
                maxY = corner.y if corner.y > maxY else maxY
                minX = corner.x if corner.x < minX else minX
                minY = corner.y if corner.y < minY else minY
                xyList.append((corner.x, corner.y))
        else:
            xyList = [(np.nan, np.nan)]*4
        return xyList, minX, maxX, minY, maxY

    for i, corners in enumerate(cornersList):
        ccdIds.append(corners[0])
        convertedCornersRaw, xMinRaw, xMaxRaw, yMinRaw, yMaxRaw = convertCorners(corners[1])
        rawCorners.append(convertedCornersRaw)
        if "HSC" in plotTitle and int(corners[0]) == 9:
            if log is not None:
                log.info("Skipping plotting of CCD 9 calexp WCS (despite its presence...)")
        else:
            convertedCornersCalexp, xMinCalexp, xMaxCalexp, yMinCalexp, yMaxCalexp = convertCorners(
                corners[2])
            calexpCorners.append(convertedCornersCalexp)
            convertedCornersJointcal, xMinJointcal, xMaxJointcal, yMinJointcal, yMaxJointcal = convertCorners(
                corners[3])
            jointcalCorners.append(convertedCornersJointcal)

        xaxLimMin = np.nanmin([xMinRaw, xMinCalexp, xMinJointcal, xaxLimMin])
        xaxLimMax = np.nanmax([xMaxRaw, xMaxCalexp, xMaxJointcal, xaxLimMax])
        yaxLimMin = np.nanmin([yMinRaw, yMinCalexp, yMinJointcal, yaxLimMin])
        yaxLimMax = np.nanmax([yMaxRaw, yMaxCalexp, yMaxJointcal, yaxLimMax])

    padPixels = abs(int(0.07*max(xaxLimMax - xaxLimMin, yaxLimMax - yaxLimMin)))
    xTractLims = (xaxLimMin - padPixels, xaxLimMax + padPixels)
    yTractLims = (yaxLimMin - padPixels, yaxLimMax + padPixels)
    if tractInfo is not None:
        tractWcs = tractInfo.getWcs()
        # Get RA and Dec of tract plot limits to add to axis labels
        tract00 = tractWcs.pixelToSky(geom.Point2D(xTractLims[0], yTractLims[0]))
        tractN0 = tractWcs.pixelToSky(geom.Point2D(xTractLims[1], yTractLims[0]))
        tract0N = tractWcs.pixelToSky(geom.Point2D(xTractLims[0], yTractLims[1]))
        xax2 = ax.twiny()
        yax2 = ax.twinx()
        xax2.set_xlim(tract00.getRa().asDegrees(), tractN0.getRa().asDegrees())
        yax2.set_ylim(tract00.getDec().asDegrees(), tract0N.getDec().asDegrees())
        xax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        yax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        twinColor = "steelblue"
        xax2.xaxis.label.set_color(twinColor)
        yax2.yaxis.label.set_color(twinColor)
        tickKwargs = dict(which="both", direction="in", labelsize=7, labelcolor=twinColor, color=twinColor)
        xax2.tick_params(axis="x", **tickKwargs)
        yax2.tick_params(axis="y", **tickKwargs)
        xax2.set_xlabel("Ra (deg)", labelpad=3, fontsize=7, color=twinColor)
        yax2.set_ylabel("Dec (deg)", labelpad=9, fontsize=7, color=twinColor, rotation=270)
        tractXs, tractYs = bboxToXyCoordLists(tractInfo.getBBox(), wcs=None, close=True)
        ax.plot(tractXs, tractYs, color="tab:orange", linestyle="dotted", linewidth=1, alpha=0.9,
                label="tract {}".format(str(tractInfo.getId())))
        legendPatches.append(Patch(facecolor="white", edgecolor="tab:orange", alpha=0.9, linestyle="dotted",
                                   linewidth=0.6, label="tract {}".format(str(tractInfo.getId()))))

    ax.set_xlim(xTractLims)
    ax.set_ylim(yTractLims)
    ax.tick_params(which="both", direction="in", labelsize=8)
    ax.set_xlabel("Tract x (pixels)")
    ax.set_ylabel("Tract y (pixels)")
    if plotTitle is not None:
        ax.set_title(plotTitle, fontsize=10, pad=3)
    annotateKwargs = dict(xycoords="axes fraction", fontsize=7, color="mediumblue", alpha=1.0)
    if plotMaxStr is not None:
        ax.annotate(plotMaxStr, xy=(0.025, 0.95), **annotateKwargs)
    if plotMedStr is not None:
        ax.annotate(plotMedStr, xy=(0.025, 0.03), **annotateKwargs)
    if rerun is not None:
        ax.annotate("rerun: " + rerun, xy=(0.5, 1.15), xycoords="axes fraction", ha="center", fontsize=7,
                    color="purple")
    for iWcs, wcsStr in enumerate(wcsList):
        patches = []
        color = lineColorDict[wcsList[iWcs]]
        faceColor = matplotlib.colors.to_rgba(color, alpha=0.1)
        edgeColor = matplotlib.colors.to_rgba(color, alpha=1.0)
        polyKwargs = dict(closed=True, facecolor=faceColor, edgecolor=edgeColor, linewidth=0.6)
        if wcsStr == "rawWcs":
            lineStyle = "solid"
            for ic, rawCorner in enumerate(rawCorners):
                rawPatch = Polygon(rawCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(rawPatch)
        if wcsStr == "calexpWcs":
            lineStyle = (0, (5, 2))
            for calexpCorner in calexpCorners:
                calexpPatch = Polygon(calexpCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(calexpPatch)
        if wcsStr == "jointcalWcs":
            lineStyle = (0, (4, 3))
            for jointcalCorner in jointcalCorners:
                jointcalPatch = Polygon(jointcalCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(jointcalPatch)
        patchCollection = PatchCollection(patches, match_original=True)
        ax.add_collection(patchCollection)
        legendPatches.append(Patch(facecolor=faceColor, edgecolor=edgeColor, linestyle=lineStyle,
                                   linewidth=0.6, label=wcsStr))
    ax.legend(handles=legendPatches, fontsize=6, loc="upper right")

    for ic, center in enumerate(centerList):
        ccdInt = int(ccdIds[ic])
        ccdColor = "black"
        if ccdMaxX == ccdMaxY:
            ccdColor = "mediumblue" if (ccdInt == ccdMaxX)  else ccdColor
        else:
            ccdColor = "mediumblue" if (ccdInt == ccdMaxX and ccdInt != ccdMaxY) else ccdColor
            ccdColor = "darkviolet" if (ccdInt != ccdMaxX and ccdInt == ccdMaxY) else ccdColor
        fontWeight = "normal" if ccdColor == "black" else "heavy"
        ax.annotate(ccdIds[ic], xy=(center[0], center[1]), ha="center", va="center", fontsize=7,
                    fontweight=fontWeight, color=ccdColor)

    fig.set_dpi(120)
    return plt.gcf()

def plotWcsOutlinesFocalPlane(cornersList, centerList, rawWcs, plotTitle=None, rerun=None, tractInfo=None,
                              plotMaxStr=None, plotMedStr=None, log=None, ccdMaxX=-99, ccdMaxY=-99,
                              ccdMaxVec=-99):
    """Plot the outlines of the ccds in ccdList for various wcs in Focal Plane
    coordinates.
    """
    xaxLimMin, xaxLimMax = 1e12, -1e12
    yaxLimMin, yaxLimMax = 1e12, -1e12
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.85)
    ccdIds = []
    rawCorners = []
    calexpCorners = []
    jointcalCorners = []
    wcsList = ["rawWcs", "calexpWcs", "jointcalWcs"]
    lineColorDict = {"rawWcs": "palevioletred", "calexpWcs": "teal", "jointcalWcs": "black"}
    legendPatches = []

    def convertCorners(corners):
        minX, maxX = 1e12, -1e12
        minY, maxY = 1e12, -1e12
        xyList = []
        if corners is not None:
            for corner in corners:
                maxX = corner.x if corner.x > maxX else maxX
                maxY = corner.y if corner.y > maxY else maxY
                minX = corner.x if corner.x < minX else minX
                minY = corner.y if corner.y < minY else minY
                xyList.append((corner.x, corner.y))
        else:
            xyList = [(np.nan, np.nan)]*4
        return xyList, minX, maxX, minY, maxY

    for i, corners in enumerate(cornersList):
        ccdIds.append(corners[0])
        convertedCornersRaw, xMinRaw, xMaxRaw, yMinRaw, yMaxRaw = convertCorners(corners[1])
        rawCorners.append(convertedCornersRaw)
        if "HSC" in plotTitle and int(corners[0]) == 9:
            if log is not None:
                log.info("Skipping plotting of CCD 9 calexp WCS (despite its presence...)")
        else:
            convertedCornersCalexp, xMinCalexp, xMaxCalexp, yMinCalexp, yMaxCalexp = convertCorners(
                corners[2])
            calexpCorners.append(convertedCornersCalexp)
            convertedCornersJointcal, xMinJointcal, xMaxJointcal, yMinJointcal, yMaxJointcal = convertCorners(
                corners[3])
            jointcalCorners.append(convertedCornersJointcal)

        xaxLimMin = np.nanmin([xMinRaw, xMinCalexp, xMinJointcal, xaxLimMin])
        xaxLimMax = np.nanmax([xMaxRaw, xMaxCalexp, xMaxJointcal, xaxLimMax])
        yaxLimMin = np.nanmin([yMinRaw, yMinCalexp, yMinJointcal, yaxLimMin])
        yaxLimMax = np.nanmax([yMaxRaw, yMaxCalexp, yMaxJointcal, yaxLimMax])

    padPixels = abs(int(0.07*max(xaxLimMax - xaxLimMin, yaxLimMax - yaxLimMin)))
    xFpLims = (xaxLimMin - padPixels, xaxLimMax + padPixels)
    yFpLims = (yaxLimMin - padPixels, yaxLimMax + padPixels)
    if rawWcs is not None:
        # Get RA and Dec of focal plane plot limits to add to axis labels
        fp00 = rawWcs.pixelToSky(geom.Point2D(xFpLims[0], yFpLims[0]))
        fpN0 = rawWcs.pixelToSky(geom.Point2D(xFpLims[1], yFpLims[0]))
        fp0N = rawWcs.pixelToSky(geom.Point2D(xFpLims[0], yFpLims[1]))
        xax2 = ax.twiny()
        yax2 = ax.twinx()
        xax2.set_xlim(fp00.getRa().asDegrees(), fpN0.getRa().asDegrees())
        yax2.set_ylim(fp00.getDec().asDegrees(), fp0N.getDec().asDegrees())
        xax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        yax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        twinColor = "steelblue"
        xax2.xaxis.label.set_color(twinColor)
        yax2.yaxis.label.set_color(twinColor)
        tickKwargs = dict(which="both", direction="in", labelsize=7, labelcolor=twinColor, color=twinColor)
        xax2.tick_params(axis="x", **tickKwargs)
        yax2.tick_params(axis="y", **tickKwargs)
        xax2.set_xlabel("Ra (deg)", labelpad=3, fontsize=7, color=twinColor)
        yax2.set_ylabel("Dec (deg)", labelpad=9, fontsize=7, color=twinColor, rotation=270)

    ax.set_xlim(xFpLims)
    ax.set_ylim(yFpLims)
    ax.tick_params(which="both", direction="in", labelsize=8)
    ax.set_xlabel("Focal Plane x (pixels)")
    ax.set_ylabel("Focal Plane y (pixels)")
    if plotTitle is not None:
        ax.set_title(plotTitle, fontsize=10, pad=3)
    annotateKwargs = dict(xycoords="axes fraction", fontsize=7, color="mediumblue", alpha=1.0)
    if plotMaxStr is not None:
        ax.annotate(plotMaxStr, xy=(0.025, 0.95), **annotateKwargs)
    if plotMedStr is not None:
        ax.annotate(plotMedStr, xy=(0.025, 0.03), **annotateKwargs)
    if rerun is not None:
        ax.annotate("rerun: " + rerun, xy=(0.5, 1.15), xycoords="axes fraction", ha="center", fontsize=7,
                    color="purple")
    for iWcs, wcsStr in enumerate(wcsList):
        patches = []
        color = lineColorDict[wcsList[iWcs]]
        faceColor = matplotlib.colors.to_rgba(color, alpha=0.1)
        edgeColor = matplotlib.colors.to_rgba(color, alpha=1.0)
        polyKwargs = dict(closed=True, facecolor=faceColor, edgecolor=edgeColor, linewidth=0.6)
        if wcsStr == "rawWcs":
            lineStyle = "solid"
            for ic, rawCorner in enumerate(rawCorners):
                rawPatch = Polygon(rawCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(rawPatch)
        if wcsStr == "calexpWcs":
            lineStyle = (0, (5, 2))
            for calexpCorner in calexpCorners:
                calexpPatch = Polygon(calexpCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(calexpPatch)
        if wcsStr == "jointcalWcs":
            lineStyle = (0, (4, 3))
            for jointcalCorner in jointcalCorners:
                jointcalPatch = Polygon(jointcalCorner, linestyle=lineStyle, **polyKwargs)
                patches.append(jointcalPatch)
        patchCollection = PatchCollection(patches, match_original=True)
        ax.add_collection(patchCollection)
        legendPatches.append(Patch(facecolor=faceColor, edgecolor=edgeColor, linestyle=lineStyle,
                                   linewidth=0.6, label=wcsStr))
    ax.legend(handles=legendPatches, fontsize=6, loc="upper right")

    for ic, center in enumerate(centerList):
        ccdInt = int(ccdIds[ic])
        ccdColor = "black"
        if ccdMaxX == ccdMaxY:
            ccdColor = "mediumblue" if ccdInt == ccdMaxX else ccdColor
        else:
            ccdColor = "mediumblue" if (ccdInt == ccdMaxX and ccdInt != ccdMaxY) else ccdColor
            ccdColor = "darkviolet" if (ccdInt != ccdMaxX and ccdInt == ccdMaxY) else ccdColor
        if ccdInt == ccdMaxVec:
            ccdColor = "red"
        fontWeight = "normal" if ccdColor == "black" else "heavy"
        ax.annotate(ccdIds[ic], xy=(center[0], center[1]), ha="center", va="center", fontsize=7,
                    fontweight=fontWeight, color=ccdColor)

    fig.set_dpi(120)
    return plt.gcf()
