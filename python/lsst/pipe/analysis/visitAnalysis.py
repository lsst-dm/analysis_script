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

import copy
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import defaultdict

import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler
from lsst.daf.persistence.butler import Butler
from lsst.pex.config import Field, ChoiceField
from lsst.pipe.base import ArgumentParser, TaskRunner, TaskError, Struct
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from .analysis import Analysis
from .coaddAnalysis import (FLAGCOLORS, CoaddAnalysisConfig, CoaddAnalysisTask, CompareCoaddAnalysisConfig,
                            CompareCoaddAnalysisTask)
from .utils import (matchAndJoinCatalogs, makeBadArray, calibrateSourceCatalogMosaic,
                    calibrateSourceCatalogPhotoCalib, calibrateSourceCatalog, andCatalog, writeParquet,
                    getRepoInfo, findCcdKey, getCcdNameRefList, getDataExistsRefList, addPreComputedColumns,
                    updateVerifyJob, savePlots, getSchema, computeAreaDict)
from .plotUtils import annotateAxes, labelVisit, labelCamera, plotText, getPlotInfo
from .fakesAnalysis import (addDegreePositions, matchCatalogs, addNearestNeighbor, fakesPositionCompare,
                            calcFakesAreaDepth, plotFakesAreaDepth, fakesMagnitudeCompare,
                            fakesMagnitudeNearestNeighbor, fakesMagnitudeBlendedness, fakesCompletenessPlot,
                            fakesMagnitudePositionError)

matplotlib.use("Agg")
np.seterr(all="ignore")


class CcdAnalysis(Analysis):
    def plotAll(self, description, plotInfoDict, areaDict, log, enforcer=None, matchRadius=None,
                matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, uberCalLabel=None, postFix="",
                plotRunStats=True, highlightList=None, haveFpCoords=None, doPrintMedian=False):
        stats = self.stats
        if self.config.doPlotCcdXy:
            yield from self.plotCcd(self.shortName, style="ccd" + postFix, stats=self.stats,
                                    matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                    zpLabel=zpLabel)
        if self.config.doPlotFP and haveFpCoords:
            yield from self.plotFocalPlane(self.shortname, plotInfoDict, style="fpa" + postFix, stats=stats,
                                           matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                           zpLabel=zpLabel, forcedStr=forcedStr)

        yield from Analysis.plotAll(self, description, plotInfoDict, areaDict, log, enforcer=enforcer,
                                    matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                    zpLabel=zpLabel, forcedStr=forcedStr, postFix=postFix,
                                    plotRunStats=plotRunStats, highlightList=highlightList,
                                    doPrintMedian=doPrintMedian)

    def plotFP(self, description, plotInfoDict, log, enforcer=None, matchRadius=None, matchRadiusUnitStr=None,
               zpLabel=None, forcedStr=None):
        yield from self.plotFocalPlane(self.shortName, plotInfoDict, style="fpa", stats=self.stats,
                                       matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                       zpLabel=zpLabel, forcedStr=forcedStr)

    def plotCcd(self, description, plotInfoDict, centroid="base_SdssCentroid", cmap=plt.cm.nipy_spectral,
                idBits=32, visitMultiplier=200, stats=None, matchRadius=None,
                matchRadiusUnitStr=None, zpLabel=None, doPrintMedian=False, style="ccd"):
        """Plot quantity as a function of CCD x,y.
        """
        xx = self.catalog[self.prefix + centroid + "_x"]
        yy = self.catalog[self.prefix + centroid + "_y"]
        ccd = (self.catalog[self.prefix + "id"] >> idBits) % visitMultiplier
        vMin, vMax = ccd.min(), ccd.max()
        if vMin == vMax:
            vMin, vMax = vMin - 2, vMax + 2
            self.log.info("Only one CCD ({0:d}) to analyze: setting vMin ({1:d}), vMax ({2:d})".format(
                          ccd.min(), vMin, vMax))
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        ptSize = None
        for name, data in self.data.items():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            if not ptSize:
                ptSize = min(12, max(4, int(25/np.log10(len(data.mag)))))
            selection = data.selection & good
            quantity = data.quantity[good[data.selection]]
            kwargs = {"s": ptSize, "marker": "o", "lw": 0, "alpha": 0.5, "cmap": cmap,
                      "vmin": vMin, "vmax": vMax}
            axes[0].scatter(xx[selection], quantity, c=ccd[selection], **kwargs)
            axes[1].scatter(yy[selection], quantity, c=ccd[selection], **kwargs)

        axes[0].set_xlabel("x_ccd", labelpad=-1)
        axes[1].set_xlabel("y_ccd")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")
        if stats is not None:
            annotateAxes(description, plt, axes[0], stats, "star", self.config.magThreshold, x0=0.03,
                         yOff=0.07, hscRun=plotInfoDict["hscRun"], matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian, matchRadius=matchRadius)
            annotateAxes(description, plt, axes[1], stats, "star", self.config.magThreshold, x0=0.03,
                         yOff=0.07, hscRun=plotInfoDict["hscRun"], matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian, matchRadius=matchRadius)
        axes[0].set_xlim(-100, 2150)
        axes[1].set_xlim(-100, 4300)
        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []  # fake up the array of the scalar mappable. Urgh...
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.83, 0.15, 0.04, 0.7])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("CCD index", rotation=270, labelpad=15)
        labelVisit(description, plt, axes[0], 0.5, 1.1)
        if zpLabel:
            plotText(zpLabel, plt, axes[0], 0.08, -0.11, prefix="zp: ", color="green")
        yield Struct(fig=fig, description=description, style=style, stats=stats)

    def plotFocalPlane(self, description, plotInfoDict, cmap=plt.cm.Spectral, stats=None, matchRadius=None,
                       matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, fontSize=8, style="fpa"):
        """Plot quantity colormaped on the focal plane.
        """
        xFp = self.catalog[self.prefix + "base_FPPosition_x"]
        yFp = self.catalog[self.prefix + "base_FPPosition_y"]
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                np.ones(len(self.mag), dtype=bool))
        if "galaxy" in self.data and "calib_psf_used" not in self.goodKeys:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        else:
            vMin, vMax = self.qMin, self.qMax
        # Set limits to ccd pixel ranges when plotting the centroids (which are
        # in pixel units).
        if description.find("Centroid") > -1:
            cmap = plt.cm.pink
            vMin = min(0, np.round(self.data["star"].quantity.min() - 10))
            vMax = np.round(self.data["star"].quantity.max() + 50, -2)
        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.7"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=fontSize)
        for name, data in self.data.items():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            axes.scatter(xFp[selection], yFp[selection], s=2, marker="o", lw=0,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)
        axes.set_xlabel("x_fpa (pixels)")
        axes.set_ylabel("y_fpa (pixels)")

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []  # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName, rotation=270, labelpad=15)
        if plotInfoDict["hscRun"]:
            axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")
        labelVisit(plotInfoDict, fig, axes, 0.5, 1.04)
        if plotInfoDict["camera"]:
            labelCamera(plotInfoDict, fig, axes, 0.5, 1.09)
        if zpLabel:
            plotText(zpLabel, fig, axes, 0.08, -0.1, prefix="zp: ", color="green")
        if forcedStr:
            plotText(forcedStr, fig, axes, 0.86, -0.1, prefix="cat: ", color="green")
        yield Struct(fig=fig, description=description, stats=stats, style=style)


class VisitAnalysisConfig(CoaddAnalysisConfig):
    doApplyExternalPhotoCalib = Field(dtype=bool, default=True,
                                      doc=("Whether to apply external photometric calibration (e.g. "
                                           "fgcmcal/jointcal/meas_mosaic) via an `lsst.afw.image.PhotoCalib` "
                                           "object?  If `True`, uses ``externalPhotoCalibName`` field to "
                                           "determine which calibration to load.  If `False`, the "
                                           "instrumental flux corresponding to 0th magnitude, fluxMag0, "
                                           "from SFM is applied."))
    externalPhotoCalibName = ChoiceField(dtype=str, default="jointcal",
                                         allowed={"jointcal": "Use jointcal_photoCalib",
                                                  "fgcm": "Use fgcm_photoCalib",
                                                  "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                         doc=("Type of external `lsst.afw.image.PhotoCalib` if "
                                              "``doApplyExternalPhotoCalib`` is `True`."))
    doApplyExternalSkyWcs = Field(dtype=bool, default=True,
                                  doc=("Whether to apply external astrometric calibration via an "
                                       "`lsst.afw.geom.SkyWcs` object.  Uses ``externalSkyWcsName`` field "
                                       "to determine which calibration to load."))
    externalSkyWcsName = ChoiceField(dtype=str, default="jointcal",
                                     allowed={"jointcal": "Use jointcal_wcs"},
                                     doc=("Type of external `lsst.afw.geom.SkyWcs` if "
                                          "``doApplyExternalSkyWcs`` is `True`."))
    useMeasMosaic = Field(dtype=bool, default=False,
                          doc=("Use meas_mosaic's applyMosaicResultsExposure to apply meas_mosaic "
                               "calibration results to catalog (i.e. as opposed to using the photoCalib "
                               "object)?"))

    hasFakes = Field(dtype=bool, default=False, doc="Include the analysis of the added fake sources?")
    inputFakesRaCol = Field(
        dtype=str,
        doc="RA column name used in the fake source catalog.",
        default="raJ2000"
    )
    inputFakesDecCol = Field(
        dtype=str,
        doc="Dec. column name used in the fake source catalog.",
        default="decJ2000"
    )
    catalogRaCol = Field(
        dtype=str,
        doc="RA column name used in the source catalog.",
        default="coord_ra",
    )
    catalogDecCol = Field(
        dtype=str,
        doc="Dec. column name used in the source catalog.",
        default="coord_dec",
    )

    def setDefaults(self):
        CoaddAnalysisConfig.setDefaults(self)
        self.analysis.fluxColumn = "base_PsfFlux_instFlux"
        self.analysisAstromMatches.fluxColumn = "base_PsfFlux_instFlux"
        self.analysisPhotomMatches.fluxColumn = "base_PsfFlux_instFlux"

    def validate(self):
        CoaddAnalysisConfig.validate(self)


class VisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        if len(parsedCmd.id.refList) < 1:
            raise RuntimeWarning("refList from parsedCmd is empty...")
        kwargs["tract"] = parsedCmd.tract
        kwargs["subdir"] = parsedCmd.subdir

        idParser = parsedCmd.id.__class__(parsedCmd.id.level)
        idParser.idList = parsedCmd.id.idList
        idParser.datasetType = parsedCmd.id.datasetType
        idParser.makeDataRefList(parsedCmd)

        if parsedCmd.collection is not None:
            # For some reason the gen3 butler is needing the filter in the
            # dataId, so it must be supplied on the command line --id.
            if "filter" not in idParser.idList[0]:
                raise RuntimeError("Must in include filter in the input --id")
            repoRootDir = "/repo/dc2" if parsedCmd.instrument == "LSSTCam-imSim" else "/repo/main"
            if parsedCmd.instrument is None:
                raise RuntimeError("Must provide --instrument command line option for gen3 repos.")
            butlerGen3 = dafButler.Butler(repoRootDir, collections=parsedCmd.collection,
                                          instrument=parsedCmd.instrument)
            butlerGen2 = parsedCmd.butler
            parsedCmd.butler = butlerGen3
            kwargs["butlerGen2"] = butlerGen2

        ccdList = []
        gen3RefList = []
        ccdKey = findCcdKey(parsedCmd.id.refList[0].dataId)
        for pId in parsedCmd.id.refList:
            if int(pId.dataId["tract"]) == int(parsedCmd.tract):
                ccdList.append(pId.dataId[ccdKey])

        visits = defaultdict(list)
        if parsedCmd.collection is None:
            for ref in parsedCmd.id.refList:
                if int(ref.dataId["tract"]) == int(parsedCmd.tract):
                    visits[ref.dataId["visit"]].append(ref)
        else:
            pIdList = copy.deepcopy(idParser.idList)
            if len(pIdList) == 1 and len(pIdList) < len(ccdList):
                pIdList = pIdList*len(ccdList)
            for pId, rerunCcdId in zip(pIdList, ccdList):
                pIdCopy = copy.deepcopy(pId)
                if "ccd" in pIdCopy:
                    pIdCopy["detector"] = pIdCopy.pop("ccd")
                else:
                    pIdCopy["detector"] = rerunCcdId
                try:
                    butlerGen3.getURI("calexp", dataId=pIdCopy)
                    gen3RefList.append({"butler": butlerGen3, "camera": parsedCmd.instrument,
                                        "dataId": pIdCopy})
                except LookupError:
                    continue
            for ref in gen3RefList:
                visits[ref["dataId"]["visit"]].append(ref)

        return [(visits[key], kwargs) for key in visits.keys()]


class VisitAnalysisTask(CoaddAnalysisTask):
    _DefaultName = "visitAnalysis"
    ConfigClass = VisitAnalysisConfig
    RunnerClass = VisitAnalysisRunner
    AnalysisClass = CcdAnalysis

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--collection", required=False, default=None,
                            help="Collection for run if it is Gen3.  NOTE: must still point to a gen2 "
                            "input to get a valid parsed data reference list and a gen2-stlye rerun for "
                            "plot persistence. E.g. "
                            "/datasets/hsc/repo/ --rerun RC/w_2021_NN/DM-NNNNN-sfm:private/username/outDir "
                            "--collection HSC/runs/RC2/w_2021_NN/DM-NNNNN --instrument HSC or"
                            "/datasets/DC2/repoRun2.2i "
                            "--rerun w_2021_NN/DM-NNNNN/sfm:private/username/outDir "
                            " --collection 2.2i/runs/test-med-1/w_2021_NN/DM-NNNNN "
                            "--instrument LSSTCam-imSim")
        parser.add_argument("--instrument", required=False, default=None,
                            help="Instrument for run if it is Gen3")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/visit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList, tract=None, subdir="", butlerGen2=None):
        plotList = []
        dataset = "source" if self.config.doReadParquetTables else "src"
        self.log.info("dataRefList size: {:d}".format(len(dataRefList)))
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split("^")]
        dataRefListPerTract = [None]*len(tractList)
        for i, tract in enumerate(tractList):
            dataRefListPerTract[i] = []
            for dataRef in dataRefList:
                if isinstance(dataRef, dict):
                    if "dataId" in dataRef:
                        dataRef["dataId"].update({"tract": int(tract)})
                dataRefListPerTract[i].append(dataRef)

        commonZpDone = False
        self.catLabel = ""

        for i, dataRefListTract in enumerate(dataRefListPerTract):
            if not dataRefListTract:
                if len(dataRefListPerTract) <= 1:
                    raise RuntimeError("No data found for tract: {:d}".format(tractList[i]))
                else:
                    self.log.info("No data found for tract: {:d}".format(tractList[i]))
                    continue
            _, ccdListPerTract = getDataExistsRefList(dataRefListTract, dataset)
            repoInfo = getRepoInfo(dataRefListTract[ccdListPerTract[0]], catDataset=dataset,
                                   doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib,
                                   externalPhotoCalibName=self.config.externalPhotoCalibName,
                                   doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs,
                                   externalSkyWcsName=self.config.externalSkyWcsName)
            self.log.info("dataId: {!s:s}".format(repoInfo.dataId))

            plotInfoDict = getPlotInfo(repoInfo)
            subdir = "ccd-" + str(ccdListPerTract[0]) if len(ccdListPerTract) == 1 else subdir
            repoInfo.dataId["subdir"] = "/" + subdir
            # Dict of all parameters common to plot* functions
            plotInfoDict.update(dict(plotType="plotVisit", subdir=subdir, ccdList=ccdListPerTract,
                                     hscRun=repoInfo.hscRun, tractInfo=repoInfo.tractInfo,
                                     dataId=repoInfo.dataId))

            if not ccdListPerTract:
                raise RuntimeError("No datasets found for datasetType = {:s}".format(repoInfo.catDataset))
            if self.config.doApplyExternalPhotoCalib:
                _, ccdPhotoCalibListPerTract = getDataExistsRefList(dataRefListTract,
                                                                    repoInfo.photoCalibDataset)
                if not ccdPhotoCalibListPerTract:
                    self.log.fatal(f"No data found for {repoInfo.photoCalibDataset} dataset...are you sure "
                                   "you ran the external photometric calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib=False")
            if self.config.doApplyExternalSkyWcs:
                # Check for wcs for compatibility with old dataset naming
                _, ccdSkyWcsListPerTract = getDataExistsRefList(dataRefListTract, repoInfo.skyWcsDataset)
                if not ccdSkyWcsListPerTract:
                    _, ccdSkyWcsListPerTract = getDataExistsRefList(dataRefListTract, "wcs")
                    if ccdSkyWcsListPerTract:
                        repoInfo.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo.skyWcsDataset} dataset...are you sure "
                                       "you ran the external astrometric calibration?  If not, run with "
                                       "--config doApplyExternalSkyWcs=False")
            self.log.info(f"Existing {repoInfo.catDataset} data for tract {tractList[i]}: "
                          f"ccdListPerTract = \n{ccdListPerTract}")
            if self.config.doApplyExternalPhotoCalib:
                self.log.info(f"Existing {repoInfo.photoCalibDataset} data for tract {tractList[i]}: "
                              f"ccdPhotoCalibListPerTract = \n{ccdPhotoCalibListPerTract}")
            if self.config.doApplyExternalSkyWcs:
                self.log.info(f"Existing {repoInfo.skyWcsDataset} data for tract {tractList[i]}: "
                              f"ccdSkyWcsListPerTract = \n{ccdSkyWcsListPerTract}")
            if self.config.doApplyExternalPhotoCalib and not ccdPhotoCalibListPerTract:
                raise RuntimeError(f"No {repoInfo.photoCalibDataset} datasets were found...are you sure "
                                   "you ran the specified external photometric calibration?  If no "
                                   "photometric external calibrations are to be applied, run with "
                                   "--config doApplyExternalPhotoCalib=False")
            if self.config.doApplyExternalSkyWcs and not ccdSkyWcsListPerTract:
                raise RuntimeError(f"No {repoInfo.skyWcsDataset} datasets were found...are you sure "
                                   "you ran the specified external astrometric calibration?  If no "
                                   "astrometric external calibrations are to be applied, run with "
                                   "--config doApplyExternalSkywcs=False")
            if self.config.doApplyExternalPhotoCalib:
                if set(ccdListPerTract) != set(ccdPhotoCalibListPerTract):
                    self.log.warn(f"Did not find {repoInfo.photoCalibDataset} external calibrations for "
                                  f"all dataIds that do have {repoInfo.catDataset} catalogs.")
            if self.config.doApplyExternalSkyWcs:
                if set(ccdListPerTract) != set(ccdSkyWcsListPerTract):
                    self.log.warn(f"Did not find {repoInfo.skyWcsDataset} external calibrations for "
                                  f"all dataIds that do have {repoInfo.catDataset} catalogs.")

            # Create list of alias mappings for differing schema naming
            # conventions (if any).
            aliasDictList = [self.config.flagsToAlias, ]
            if repoInfo.hscRun and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0, "turquoise")]
            for ih, flagName in enumerate(list(self.config.analysis.flags)):
                if not any(flagName in highlight for highlight in highlightList):
                    highlightList += [(flagName, 0, FLAGCOLORS[ih%len(FLAGCOLORS)]), ]
            # Dict of all parameters common to plot* functions
            plotInfoDict.update(dict(ccdList=ccdListPerTract, hscRun=repoInfo.hscRun,
                                     tractInfo=repoInfo.tractInfo, dataId=repoInfo.dataId))

            if any(doPlot for doPlot in
                   [self.config.doPlotPsfFluxSnHists, self.config.doPlotSkyObjects,
                    self.config.doPlotFootprintArea, self.config.doPlotRhoStatistics,
                    self.config.doPlotQuiver, self.config.doPlotMags, self.config.doPlotStarGalaxy,
                    self.config.doPlotSizes, self.config.doPlotCentroids, self.config.doPlotRhoStatistics,
                    self.config.doWriteParquetTables]) and not self.config.plotMatchesOnly:
                if self.config.hasFakes:
                    inputFakes = repoInfo.butler.get("deepCoadd_fakeSourceCat", dataId=repoInfo.dataId)
                    inputFakes = inputFakes.toDataFrame()
                    datasetType = "fakes_" + dataset
                else:
                    inputFakes = None
                    datasetType = dataset

                externalCalKwargs = dict(doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib,
                                         doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs,
                                         useMeasMosaic=self.config.useMeasMosaic)
                if self.config.doReadParquetTables:
                    catalog, commonZpCat = self.readParquetTables(dataRefListTract, datasetType, repoInfo,
                                                                  **externalCalKwargs)
                    areaDict, _ = computeAreaDict(repoInfo, dataRefListTract, dataset="", fakeCat=None)
                else:
                    catStruct = self.readCatalogs(dataRefListTract, datasetType, repoInfo,
                                                  aliasDictList=aliasDictList, fakeCat=inputFakes,
                                                  readFootprintsAs=self.config.readFootprintsAs,
                                                  **externalCalKwargs)
                    commonZpCat = catStruct.commonZpCatalog
                    catalog = catStruct.catalog
                    # Convert to pandas DataFrames
                    commonZpCat = commonZpCat.asAstropy().to_pandas().set_index("id", drop=False)
                    catalog = catalog.asAstropy().to_pandas().set_index("id", drop=False)
                    areaDict = catStruct.areaDict
                schema = getSchema(catalog)
                xFp = catalog["base_FPPosition_x"].array  # Double check for compatibility with older repos
                self.haveFpCoords = False if len(xFp[np.where(np.isfinite(xFp))]) <= 0 else True
                # Make sub-catalog of sky sources before flag culling as many
                # of these will have flags set due to measurement difficulties
                # in regions that are really blank sky.
                skySrcCat = None
                if self.config.doPlotSkyObjects:
                    if "sky_source" in schema:
                        baseGoodSky = catalog["sky_source"]
                        if "detect_isDeblendedSource" in schema:
                            baseGoodSky &= catalog["detect_isDeblendedSource"]
                        skySrcCat = catalog[baseGoodSky].copy(deep=True)
                    else:
                        self.log.warn("doPlotSkyObjects is True, but the \"sky_source\" "
                                      "column does not exist in the catalog schema.  Skipping "
                                      "skyObjects plot.")

                # Set boolean arrays indicating sources deemed unsuitable for
                # qa analyses.
                bad = makeBadArray(catalog, flagList=self.config.analysis.flags,
                                   onlyReadStars=self.config.onlyReadStars)
                badCommonZp = makeBadArray(commonZpCat, flagList=self.config.analysis.flags,
                                           onlyReadStars=self.config.onlyReadStars)
                self.catLabel = "isPrimary"
                if self.config.doBackoutApCorr:
                    self.log.info("Backing out aperture corrections from all fluxes")
                    self.catLabel += "\n     (noApCorr)"

                # Create and write parquet tables
                if self.config.doWriteParquetTables:
                    # Add pre-computed columns for parquet tables
                    catalog = addPreComputedColumns(catalog, fluxToPlotList=self.config.fluxToPlotList,
                                                    toMilli=self.config.toMilli)
                    commonZpCat = addPreComputedColumns(commonZpCat,
                                                        fluxToPlotList=self.config.fluxToPlotList,
                                                        toMilli=self.config.toMilli)
                    if repoInfo.isGen3:
                        self.log.info("Not writing parquet files for gen3 repo...")
                    else:
                        dataRef_catalog = repoInfo.butler.dataRef("analysisVisitTable",
                                                                  dataId=repoInfo.dataId)
                        writeParquet(dataRef_catalog, catalog, badArray=bad)
                        dataRef_commonZp = repoInfo.butler.dataRef("analysisVisitTable_commonZp",
                                                                   dataId=repoInfo.dataId)
                        writeParquet(dataRef_commonZp, commonZpCat, badArray=badCommonZp)
                        if self.config.writeParquetOnly and not self.config.doPlotMatches:
                            self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                            return

                # purge the catalogs of flagged sources
                catalog = catalog[~bad].copy(deep=True)
                commonZpCat = commonZpCat[~badCommonZp].copy(deep=True)

                if self.config.hasFakes:
                    processedFakes = catalog
                    inputFakes = catStruct.fakeCat
                    inputFakes = addDegreePositions(inputFakes, self.config.inputFakesRaCol,
                                                    self.config.inputFakesDecCol)
                    processedFakes = addDegreePositions(processedFakes, self.config.catalogRaCol,
                                                        self.config.catalogDecCol)
                    processedFakes = addNearestNeighbor(processedFakes, self.config.catalogRaCol + "_deg",
                                                        self.config.catalogDecCol + "_deg")
                    inputFakesMatched, processedFakesMatched, inputFakes = matchCatalogs(
                        inputFakes, self.config.inputFakesRaCol + "_deg",
                        self.config.inputFakesDecCol + "_deg", processedFakes,
                        self.config.catalogRaCol + "_deg", self.config.catalogDecCol + "_deg")

                    areaDepthStruct = calcFakesAreaDepth(inputFakesMatched, processedFakesMatched, areaDict,
                                                         numSigmas=100.0)

                    plotList.append(plotFakesAreaDepth(inputFakesMatched, processedFakesMatched,
                                                       plotInfoDict, areaDict))
                    plotList.append(plotFakesAreaDepth(inputFakesMatched, processedFakesMatched,
                                                       plotInfoDict, areaDict, numSigmas=100.0))

                    plotInfoDict["magLim"] = areaDepthStruct.medMagsToLimit
                    plotList.append(fakesPositionCompare(inputFakesMatched, processedFakesMatched,
                                                         plotInfoDict))
                    plotList.append(fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched,
                                                          plotInfoDict, verifyJob=self.verifyJob))
                    plotList.append(fakesMagnitudeCompare(inputFakesMatched, processedFakesMatched,
                                                          plotInfoDict, verifyJob=self.verifyJob,
                                                          magCol="base_CircularApertureFlux_12_0_mag"))
                    plotList.append(fakesMagnitudeNearestNeighbor(inputFakesMatched, processedFakesMatched,
                                                                  plotInfoDict))
                    plotList.append(fakesMagnitudeBlendedness(inputFakesMatched, processedFakesMatched,
                                                              plotInfoDict))
                    plotList.append(fakesCompletenessPlot(inputFakes, inputFakesMatched,
                                                          processedFakesMatched, plotInfoDict, areaDict))
                    plotList.append(fakesMagnitudePositionError(inputFakesMatched, processedFakesMatched,
                                                                plotInfoDict, areaDict))

                if self.config.doPlotSkyObjects and skySrcCat is not None:
                    plotList.append(self.plotSkyObjects(skySrcCat, "skySources", plotInfoDict, areaDict))
                if self.config.doPlotPsfFluxSnHists:
                    plotList.append(self.plotPsfFluxSnHists(commonZpCat, "base_PsfFlux_raw",
                                                            plotInfoDict, areaDict, zpLabel="raw"))
                    plotList.append(self.plotPsfFluxSnHists(catalog, "base_PsfFlux_cal",
                                                            plotInfoDict, areaDict, zpLabel=self.zpLabel))
                plotKwargs = dict(zpLabel=self.zpLabel, forcedStr=self.catLabel)
                if self.config.doPlotFootprintArea:
                    plotList.append(self.plotFootprintHist(catalog, "footArea", plotInfoDict, **plotKwargs))
                    plotList.append(self.plotFootprint(catalog, plotInfoDict, areaDict, plotRunStats=False,
                                                       highlightList=highlightList
                                                       + [("parent", 0, "yellow"), ], **plotKwargs))

                if self.config.doPlotQuiver:
                    plotList.append(self.plotQuiver(catalog, "ellipResids", plotInfoDict, areaDict, scale=2,
                                                    **plotKwargs))

                if self.config.doPlotRhoStatistics:
                    plotList.append(self.plotRhoStatistics(catalog, plotInfoDict,
                                                           **plotKwargs))

                plotKwargs.update(dict(highlightList=highlightList))
                # Create mag comparison plots using common ZP
                if self.config.doPlotMags and not commonZpDone:
                    zpLabel = "common (%s)" % self.config.analysis.commonZp
                    plotKwargs.update(dict(zpLabel=zpLabel))
                    plotList.append(self.plotMags(commonZpCat, plotInfoDict, areaDict,
                                                  fluxToPlotList=["base_GaussianFlux",
                                                                  "base_CircularApertureFlux_12_0"],
                                                  postFix="_commonZp", **plotKwargs))
                    commonZpDone = True
                # Now calibrate the source catalg to either the instrumental
                # flux corresponding to 0th magnitude, fluxMag0, from SFM or
                # the external-calibration solution (from jointcal, fgcm, or
                # meas_mosaic) for remainder of plots.
                plotKwargs.update(dict(zpLabel=self.zpLabel))
                if self.config.doPlotMags:
                    plotList.append(self.plotMags(catalog, plotInfoDict, areaDict, **plotKwargs))
                if self.config.doPlotStarGalaxy:
                    if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
                        plotList.append(self.plotStarGal(catalog, plotInfoDict, areaDict, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotStarGal: "
                                      "ext_shapeHSM_HsmSourceMoments_xx not in the catalog schema")
                if self.config.doPlotSizes:
                    if "base_SdssShape_psf_xx" in schema:
                        plotList.append(self.plotSizes(catalog, plotInfoDict, areaDict, **plotKwargs))
                    else:
                        self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in the catalog schema")
                if self.config.doPlotCentroids and self.haveFpCoords:
                    plotList.append(self.plotCentroidXY(catalog, plotInfoDict, areaDict, **plotKwargs))

            # Haven't got reference catalog loading working for a gen3 repo...
            if (self.config.doPlotMatches or self.config.doWriteParquetTables) and not repoInfo.isGen3:
                # Read in and unpack just the persisted srcMatch from SFM
                # (which still uses ps1 for astrometric calibration).
                sfmUnpackedMatches, _ = self.readSrcMatches(
                    repoInfo, dataRefListTract, dataset="src", refObjLoader=self.config.photomRefObjLoader,
                    aliasDictList=aliasDictList, readPackedMatchesOnly=True,
                    doApplyExternalPhotoCalib=False, doApplyExternalSkyWcs=False)
                self.zpLabelPacked = self.zpLabel  # not doing external calibration for this sample
                self.zpLabel = None
                if sfmUnpackedMatches is not None:
                    self.unpackedMatchLabel = ("unpackedMatches: \n"
                                               + self.config.photomRefObjLoader.ref_dataset_name)
                else:
                    self.unpackedMatchLabel = ("matches also used in SFM astrom: \n"
                                               + self.config.astromRefObjLoader.ref_dataset_name)

                matchAreaDict = {}
                externalCalKwargs = dict(doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib,
                                         doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs,
                                         useMeasMosaic=self.config.useMeasMosaic)
                astromMatches, astromMatchAreaDict = self.readSrcMatches(
                    repoInfo, dataRefListTract, dataset="src", refObjLoader=self.config.astromRefObjLoader,
                    aliasDictList=aliasDictList, **externalCalKwargs)
                photomMatches, photomMatchAreaDict = self.readSrcMatches(
                    repoInfo, dataRefListTract, dataset="src", refObjLoader=self.config.photomRefObjLoader,
                    aliasDictList=aliasDictList, **externalCalKwargs)
                if self.config.doWriteParquetTables:
                    for calibType, calibMatches in [("Astrom", astromMatches), ("Photom", photomMatches)]:
                        matchesDataRef = repoInfo.butler.dataRef(
                            "analysis" + calibType + "MatchFullRefVisitTable", dataId=repoInfo.dataId)
                        writeParquet(matchesDataRef, calibMatches, badArray=None, prefix="src_")
                if self.config.writeParquetOnly:
                    self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                    return

                if self.config.doPlotMatches:
                    plotKwargs = dict(zpLabel=self.zpLabel)
                    # Dict of all parameters common to plot* functions
                    matchHighlightList = [
                        ("src_" + self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0,
                         "turquoise"), ("src_deblend_nChild", 0, "lime")]
                    for ih, flagName in enumerate(list(self.config.analysis.flags)):
                        flagName = "src_" + flagName
                        if not any(flagName in highlight for highlight in matchHighlightList):
                            matchHighlightList += [(flagName, 0, FLAGCOLORS[ih%len(FLAGCOLORS)]), ]
                    plotKwargs.update(dict(highlightList=matchHighlightList, matchRadius=self.matchRadius,
                                           matchRadiusUnitStr=self.matchRadiusUnitStr))
                    matchLabel = "matched to\n" + self.config.astromRefObjLoader.ref_dataset_name
                    matchLabel = (matchLabel + "\n     (noApCorr)" if self.config.doBackoutApCorr
                                  else matchLabel)
                    plotList.append(self.plotAstromMatches(
                        astromMatches, plotInfoDict, astromMatchAreaDict, self.config.astromRefObjLoader,
                        unpackedMatches=sfmUnpackedMatches, forcedStr=matchLabel, **plotKwargs))
                    matchLabel = "matched to\n" + self.config.photomRefObjLoader.ref_dataset_name
                    matchLabel = (matchLabel + "\n     (noApCorr)" if self.config.doBackoutApCorr
                                  else matchLabel)
                    plotList.append(self.plotPhotomMatches(photomMatches, plotInfoDict, photomMatchAreaDict,
                                                           self.config.photomRefObjLoader,
                                                           forcedStr=matchLabel, **plotKwargs))

                for cat in self.config.externalCatalogs:
                    if self.config.photoCatName not in cat:
                        with andCatalog(cat):
                            matches = self.matchCatalog(catalog, plotInfoDict["filter"],
                                                        self.config.externalCatalogs[cat])
                            matchLabel = "matched to\n" + self.config.externalCatalogs[cat]
                            plotList.append(self.plotMatches(matches, plotInfoDict, matchAreaDict,
                                                             **plotKwargs))
        metaDict = {"tract": plotInfoDict["tract"], "visit": plotInfoDict["visit"],
                    "filter": plotInfoDict["filter"]}
        if plotInfoDict["cameraName"]:
            metaDict.update({"camera": plotInfoDict["cameraName"]})
        self.verifyJob = updateVerifyJob(self.verifyJob, metaDict=metaDict)
        if not repoInfo.isGen3:
            verifyJobFilename = repoInfo.butler.get("visitAnalysis_verify_job_filename",
                                                    dataId=repoInfo.dataId)[0]
        else:
            verifyJobFilename = butlerGen2.get("visitAnalysis_verify_job_filename", dataId=repoInfo.dataId)[0]
        if plotList:
            if repoInfo.isGen3:
                savePlots(plotList, "plotVisit", repoInfo.dataId, butlerGen2, subdir=subdir)
            else:
                savePlots(plotList, "plotVisit", repoInfo.dataId, repoInfo.butler, subdir=subdir)

        # TODO: DM-26758 (or DM-14768) should make the following line a proper
        # butler.put by directly persisting json files.
        self.verifyJob.write(verifyJobFilename)

    def calibrateCatalogs(self, dataRef, catalog, fluxMag0, repoInfo, doApplyExternalPhotoCalib,
                          doApplyExternalSkyWcs, useMeasMosaic, iCat=None):
        """Determine and apply appropriate flux calibration to the catalog.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            If applying meas_mosaic calibrations, a dataRef is needed for call
            to meas_mosaic's applyMosaicResultsCatalog() in in utils'
            calibrateSourceCatalogMosaic().  It is also needed to distinguish
            between jointcal vs. meas_mosaic when calibrating through
            a `lsst.afw.image.PhotoCalib` object.
        catalog : `lsst.afw.table.SourceCatalog`
            The catalog to which the calibration is applied in place.
        fluxMag0 : `float`
            The instrumental flux corresponding to 0 magnitude from Single
            Frame Measurement for the catalog.
        repoInfo : `lsst.pipe.base.Struct`
            A struct containing relevant information about the repository under
            study.  Elements used here include the dataset names for any
            external calibrations to be applied.
        doApplyExternalPhotoCalib : `bool`
            If `True`: Apply the external photometric calibrations specified by
                      ``repoInfo.photoCalibDataset`` to the catalog.
            If `False`: Apply the ``fluxMag0`` photometric calibration from
                        Single Frame Measuerment to the catalog.
        doApplyExternalSkyWcs : `bool`
            If `True`: Apply the external astrometric calibrations specified by
                       ``repoInfo.skyWcsDataset`` the catalog.
            If `False`: Retain the WCS from Single Frame Measurement.
        useMeasMosaic : `bool`
            Use meas_mosaic's applyMosaicResultsCatalog for the external
            calibration (even if photoCalib object exists).  For testing
            implementations.
        iCat : `int` or None, optional
            Integer representing whether this is comparison catalog 0 or 1.

        Returns
        -------
        calibrated : `lsst.afw.table.SourceCatalog`
            The calibrated source catalog.
        zpLabel : `str`
            A label indicating the external calibration applied (currently
            either "jointcal", "fgcm", "fgcm_tract", or "meas_mosaic", but the
            latter is effectively retired)
        """
        self.zp = 0.0
        if iCat is None:
            try:
                self.zpLabel = self.zpLabel
            except Exception:
                self.zpLabel = None
        else:
            self.zpLabel = None
        zpLabel = self.zpLabel
        if doApplyExternalPhotoCalib:
            if not useMeasMosaic:
                # i.e. the processing was post-photoCalib output generation
                # AND you want the photoCalib flux object used for the
                # calibration (as opposed to meas_mosaic's fcr object).
                if not self.zpLabel:
                    if not repoInfo.isGen3:
                        zpStr = ("MMphotoCalib" if dataRef.datasetExists("fcr_md")
                                 else repoInfo.photoCalibDataset.split("_")[0].upper())
                    else:
                        zpStr = repoInfo.photoCalibDataset.split("_")[0].upper()
                    if (iCat is None or (iCat == 0 and self.zpLabel1 is None)
                            or (iCat == 1 and self.zpLabel2 is None)):  # Suppress superfluous logging
                        msg = "Applying {0:} photoCalib calibration to catalog".format(zpStr)
                        msg = msg + str(iCat + 1) if iCat is not None else msg
                        self.log.info(msg)
                    zpLabel = zpStr
                    zpLabel = zpLabel + "_" + str(iCat + 1) if iCat is not None else zpLabel
                calibrated = calibrateSourceCatalogPhotoCalib(dataRef, catalog, repoInfo.photoCalibDataset,
                                                              repoInfo.isGen3, zp=self.zp)
            else:
                # If here, the data were processed pre-photoCalib output
                # generation, so must use old method OR old method was
                # explicitly requested via useMeasMosaic.
                try:
                    import lsst.meas.mosaic  # noqa : F401
                except ImportError:
                    raise ValueError("Cannot apply calibrations because meas_mosaic could not "
                                     "be imported. \nEither setup meas_mosaic or run with "
                                     "--config doApplyExternalPhotoCalib=False doApplyExternalSkyWcs=False")
                if not self.zpLabel:
                    self.log.info("Applying meas_mosaic calibration to catalog")
                zpLabel = "MEAS_MOSAIC"
                calibrated = calibrateSourceCatalogMosaic(dataRef, catalog, zp=self.zp)
        else:
            # Scale fluxes to measured zeropoint
            self.zp = 2.5*np.log10(fluxMag0)
            if ((iCat is None and self.zpLabel is None) or (iCat == 0 and self.zpLabel1 is None)
                    or (iCat == 1 and self.zpLabel2 is None)):  # Suppress superfluous logging
                msg = ("Using 2.5*log10(fluxMag0) (zeropoint of order ~{0:.2f}) from SFM for catalog".
                       format(self.zp))
                msg = msg + str(iCat + 1) if iCat is not None else msg
                self.log.info(msg)
            zpLabel = "FLUXMAG0"
            zpLabel = zpLabel + "_" + str(iCat + 1) if iCat is not None else zpLabel
            calibrated = calibrateSourceCatalog(catalog, self.zp)

        if doApplyExternalSkyWcs:
            if not repoInfo.isGen3:
                wcs = dataRef.get(repoInfo.skyWcsDataset)
            else:
                dataId = dataRef["dataId"]
                externalSkyWcsCatalog = dataRef["butler"].get(repoInfo.skyWcsDataset, dataId=dataId)
                row = externalSkyWcsCatalog.find(dataId["detector"])
                wcs = row.getWcs()
            if wcs is None:
                self.log.warn("No wcs found for dataset {} dataId{}".format(repoInfo.skyWcsDataset, dataId))
            if isinstance(calibrated, pd.DataFrame):
                xPixelArray = np.array(calibrated["slot_Centroid_x"])
                yPixelArray = np.array(calibrated["slot_Centroid_y"])
                updatedRaDec = wcs.pixelToSkyArray(xPixelArray, yPixelArray)
                calibrated["coord_ra"] = updatedRaDec[0]
                calibrated["coord_dec"] = updatedRaDec[1]
            else:
                afwTable.updateSourceCoords(wcs, calibrated)
            if "wcs" not in zpLabel:
                if iCat is None:
                    zpLabel += "\nwcs: " + repoInfo.skyWcsDataset.split("_")[0].upper()
                if iCat is not None:
                    zpLabel += (" wcs: " + repoInfo.skyWcsDataset.split("_")[0].upper() + "_"
                                + str(iCat + 1))
        else:
            if "wcs" not in zpLabel:
                if iCat is None:
                    zpLabel += "\nwcs: SFM"
                if iCat is not None:
                    zpLabel += " wcs: SFM_" + str(iCat + 1)
        if ((iCat is None and self.zpLabel is None) or (iCat == 0 and self.zpLabel1 is None)
                or (iCat == 1 and self.zpLabel2 is None)):  # Suppress superfluous logging
            msg = "Applying WCS from {0:}".format(zpLabel[zpLabel.find("wcs:") + 5:])
            msg = msg.replace("Applying", "Using") if "SFM" in msg else msg
            msg = msg.replace("_", " for catalog") if "_" in msg else msg
            self.log.info(msg)
        if iCat is None:
            self.zpLabel = zpLabel
        else:
            self.zpLabel1 = zpLabel if iCat == 0 and not self.zpLabel1 else self.zpLabel1
            self.zpLabel2 = zpLabel if iCat == 1 and not self.zpLabel2 else self.zpLabel2
        return calibrated


class CompareVisitAnalysisConfig(VisitAnalysisConfig, CompareCoaddAnalysisConfig):
    doApplyExternalPhotoCalib1 = Field(dtype=bool, default=True,
                                       doc=("Whether to apply external photometric calibration (e.g. "
                                            "fgcmcal/jointcal/meas_mosaic) via an "
                                            "`lsst.afw.image.PhotoCalib` object to input1.  If `True`, "
                                            "uses ``externalPhotoCalibName1`` field to determine which "
                                            "calibration to load.  If `False`, the instrumental flux "
                                            "corresponding to 0th magnitude, fluxMag0, from SFM is "
                                            "applied."))
    externalPhotoCalibName1 = ChoiceField(dtype=str, default="jointcal",
                                          allowed={"jointcal": "Use jointcal_photoCalib",
                                                   "fgcm": "Use fgcm_photoCalib",
                                                   "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                          doc=("Type of external `lsst.afw.image.PhotoCalib` to apply to "
                                               "input1 if ``doApplyExternalPhotoCalib1`` is `True`."))
    doApplyExternalSkyWcs1 = Field(dtype=bool, default=True,
                                   doc=("Whether to apply external astrometric calibration via an "
                                        "`lsst.afw.geom.SkyWcs` object to input1.  Uses "
                                        "``externalSkyWcsName1`` field to determine which calibration "
                                        "to load."))
    externalSkyWcsName1 = ChoiceField(dtype=str, default="jointcal",
                                      allowed={"jointcal": "Use jointcal_wcs"},
                                      doc=("Type of external `lsst.afw.geom.SkyWcs` to apply to input1 if "
                                           "``doApplyExternalSkyWcs1`` is ``True``."))
    doApplyExternalPhotoCalib2 = Field(dtype=bool, default=True,
                                       doc=("Whether to apply external photometric calibration (e.g. "
                                            "fgcmcal/jointcal/meas_mosaic) via an "
                                            "`lsst.afw.image.PhotoCalib` object to input2.  If `True`, "
                                            "uses ``externalPhotoCalibName2`` field to determine which "
                                            "calibration to load.  If `False`, the instrumental flux "
                                            "corresponding to 0th magnitude, fluxMag0, from SFM is "
                                            "applied."))
    externalPhotoCalibName2 = ChoiceField(dtype=str, default="jointcal",
                                          allowed={"jointcal": "Use jointcal_photoCalib",
                                                   "fgcm": "Use fgcm_photoCalib",
                                                   "fgcm_tract": "Use fgcm_tract_photoCalib"},
                                          doc=("Type of external `lsst.afw.image.PhotoCalib` to apply to "
                                               "input2 if ``doApplyExternalPhotoCalib2`` is `True`."))
    doApplyExternalSkyWcs2 = Field(dtype=bool, default=True,
                                   doc=("Whether to apply external astrometric calibration via an "
                                        "`lsst.afw.geom.SkyWcs` object to input2.  Uses "
                                        "``externalSkyWcsName2`` field to determine which calibration "
                                        "to load."))
    externalSkyWcsName2 = ChoiceField(dtype=str, default="jointcal",
                                      allowed={"jointcal": "Use jointcal_wcs"},
                                      doc=("Type of external `lsst.afw.geom.SkyWcs` to apply to input2 if "
                                           "``doApplyExternalSkyWcs2`` is `True`."))
    useMeasMosaic1 = Field(dtype=bool, default=False,
                           doc=("Use meas_mosaic's applyMosaicResultsExposure to apply meas_mosaic "
                                "calibration results to input1 (i.e. as opposed to using the "
                                "photoCalib object)?"))
    useMeasMosaic2 = Field(dtype=bool, default=False,
                           doc=("Use meas_mosaic's applyMosaicResultsExposure to apply meas_mosaic "
                                "calibration results to input2 (i.e. as opposed to using the "
                                "photoCalib object)?"))

    def setDefaults(self):
        VisitAnalysisConfig.setDefaults(self)
        # If matching on Ra/Dec, use a tighter match radius for comparing runs:
        # they are calibrated and we want to avoid mis-matches.
        self.matchRadiusRaDec = 0.2
        self.matchRadiusXy = 1.0e-5  # has to be bigger than absolute zero
        if "base_PsfFlux" not in self.fluxToPlotList:
            self.fluxToPlotList.append("base_PsfFlux")  # Add PSF flux to default list for comparison scripts

    def validate(self):
        super(CoaddAnalysisConfig, self).validate()
        if not self.doApplyExternalPhotoCalib and (self.doApplyExternalPhotoCalib1
                                                   and self.doApplyExternalPhotoCalib2):
            raise ValueError("doApplyExternalPhotoCalib is set to False, but doApplyExternalPhotoCalib1 and "
                             "doApplyExternalPhotoCalib2, the appropriate settings for the "
                             "compareVisitAnalysis.py scirpt, are both True, so external calibrations would "
                             "be applied.  Try running without setting doApplyExternalPhotoCalib (which is "
                             "only appropriate for the visitAnalysis.py script).")
        if (not self.doApplyExternalSkyWcs
                and (self.doApplyExternalSkyWcs1 and self.doApplyExternalSkyWcs2)):
            raise ValueError("doApplyExternalSkyWcs is set to False, but doApplyExternalSkyWcs1 and "
                             "doApplyExternalSkyWcs2, the appropriate settings for the "
                             "compareVisitAnalysis.py scirpt, are both True, so external calibrations would "
                             "be applied.  Try running without setting doApplyExternalSkyWcs (which is "
                             "only appropriate for the visitAnalysis.py script).")
        if not self.useMeasMosaic and (self.useMeasMosaic1 and self.useMeasMosaic2):
            raise ValueError("useMeasMosaic is set to False, but useMeasMosaic1 and useMeasMosaic2, "
                             "the appropriate settings for the compareVisitAnalysis.py scirpt, are both "
                             "True, so PhotoCalib-derived calibrations would be applied.  Try running "
                             "without setting useMeasMosaic (which is only appropriate for the "
                             "visitAnalysis.py script).")


class CompareVisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        rootDir = parsedCmd.input.split("rerun")[0] if len(parsedCmd.rerun) == 2 else parsedCmd.input
        kwargs["tract"] = parsedCmd.tract
        kwargs["subdir"] = parsedCmd.subdir
        # New butler requires identical RepositoryArgs and RepositoryCfg and
        # mapperArgs={} is NOT considered equivalent to
        # mapperArgs={"calibRoot": None}, so only use if pasedCmd.calib is not
        # None.
        if parsedCmd.collection is not None:
            if parsedCmd.instrument is None:
                raise RuntimeError("Must provide --instrument command line option for gen3 repos.")
            butler2 = dafButler.Butler(parsedCmd.rerun2, collections=parsedCmd.collection,
                                       instrument=parsedCmd.instrument)
        else:
            butlerArgs = dict(root=os.path.join(rootDir, "rerun", parsedCmd.rerun2))
            if parsedCmd.calib is not None:
                butlerArgs["calibRoot"] = parsedCmd.calib
            butler2 = Butler(**butlerArgs)
            parsedCmd.butler = butler2
        idParser = parsedCmd.id.__class__(parsedCmd.id.level)
        idParser.idList = parsedCmd.id.idList
        idParser.datasetType = parsedCmd.id.datasetType
        idParser.makeDataRefList(parsedCmd)

        butler = parsedCmd.butler
        rerunRefList = []
        rerunCcdList = []
        rerun2RefList = []
        ccdKey = findCcdKey(parsedCmd.id.refList[0].dataId)
        for pId in parsedCmd.id.refList:
            if int(pId.dataId["tract"]) == int(parsedCmd.tract):
                rerunRefList.append(pId)
                rerunCcdList.append(pId.dataId[ccdKey])
        if parsedCmd.collection is None:
            for pId in idParser.refList:
                if int(pId.dataId["tract"]) == int(parsedCmd.tract):
                    rerun2RefList.append(pId)
        else:
            pIdList = copy.deepcopy(idParser.idList)
            if len(pIdList) == 1 and len(pIdList) < len(rerunCcdList):
                pIdList = pIdList*len(rerunCcdList)
            for pId, rerunCcdId in zip(pIdList, rerunCcdList):
                pIdCopy = copy.deepcopy(pId)
                if "ccd" in pIdCopy:
                    pIdCopy["detector"] = pIdCopy.pop("ccd")
                else:
                    pIdCopy["detector"] = rerunCcdId
                try:
                    butler2.getURI("calexp", dataId=pIdCopy)
                    rerun2RefList.append({"butler": butler2, "camera": parsedCmd.instrument,
                                          "dataId": pIdCopy})
                except LookupError:
                    continue

        if len(rerunRefList) > len(rerun2RefList):  # Need to pad for missing patches in rerun2
            dummyId = copy.deepcopy(pIdCopy)
            dummyId["detector"] = 999
            for iPad in range(0, len(rerunRefList) - len(rerun2RefList)):
                rerun2RefList.append({"butler": butler2, "camera": parsedCmd.instrument,
                                      "dataId": dummyId})
        parsedCmd.butler = butler
        visits1 = defaultdict(list)
        visits2 = defaultdict(list)
        for ref1, ref2 in zip(rerunRefList, rerun2RefList):
            visits1[ref1.dataId["visit"]].append(ref1)
            if parsedCmd.collection is None:
                visits2[ref2.dataId["visit"]].append(ref2)
            else:
                visits2[ref2["dataId"]["visit"]].append(ref2)
        return [(refs1, dict(dataRefList2=refs2, **kwargs)) for
                refs1, refs2 in zip(visits1.values(), visits2.values())]


class CompareVisitAnalysisTask(VisitAnalysisTask, CompareCoaddAnalysisTask):
    ConfigClass = CompareVisitAnalysisConfig
    RunnerClass = CompareVisitAnalysisRunner
    _DefaultName = "compareVisitAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "src", help="data ID with raw CCD keys, "
                               "e.g. --id visit=12345 ccd=6^8..11", ContainerClass=PerTractCcdDataIdContainer)
        parser.add_argument("--tract", type=str, default=None,
                            help="Tract(s) to use (do one at a time for overlapping) e.g. 1^5^0")
        parser.add_argument("--collection", required=False, default=None,
                            help="Collection for rerun2 if it is Gen3")
        parser.add_argument("--instrument", required=False, default=None,
                            help="Instrument for run if it is Gen3")
        parser.add_argument("--subdir", type=str, default="",
                            help=("Subdirectory below plots/filter/tract-NNNN/visit-NNNN (useful "
                                  "for, e.g., subgrouping of CCDs.  Ignored if only one CCD is "
                                  "specified, in which case the subdir is set to ccd-NNN"))
        return parser

    def runDataRef(self, dataRefList1, dataRefList2, tract=None, subdir=""):
        # This is for the commonZP plots (i.e. all ccds regardless of tract)
        if tract is None:
            tractList = [0, ]
        else:
            tractList = [int(tractStr) for tractStr in tract.split("^")]
        self.log.debug("tractList = {}".format(tractList))
        dataRefListPerTract1 = [None]*len(tractList)
        dataRefListPerTract2 = [None]*len(tractList)
        for dataRef1 in dataRefList1:
            if isinstance(dataRef1, dict):
                if "dataId" in dataRef1:
                    dataRef1["dataId"].update({"tract": int(tract)})
        for dataRef2 in dataRefList2:
            if isinstance(dataRef2, dict):
                if "dataId" in dataRef2:
                    dataRef2["dataId"].update({"tract": int(tract)})
        for i, tract in enumerate(tractList):
            dataRefListPerTract1[i] = [dataRef1 for dataRef1 in dataRefList1 if
                                       dataRef1.dataId["tract"] == tract]
            try:
                dataRefListPerTract2[i] = [dataRef2 for dataRef2 in dataRefList2 if
                                           dataRef2.dataId["tract"] == tract]
            except AttributeError:
                dataRefListPerTract2[i] = [dataRef2 for dataRef2 in dataRefList2]
        if len(dataRefListPerTract1) != len(dataRefListPerTract2):
            raise TaskError("Lengths of comparison dataRefLists do not match!")
        commonZpDone = False
        dataset1 = "source" if self.config.doReadParquetTables1 else "src"
        dataset2 = "source" if self.config.doReadParquetTables2 else "src"
        self.zpLabel1 = None
        self.zpLabel2 = None
        i = -1
        for dataRefListTract1, dataRefListTract2 in zip(dataRefListPerTract1, dataRefListPerTract2):
            i += 1
            if not dataRefListTract1:
                self.log.info("No data found in --rerun for tract: {:d}".format(tractList[i]))
                continue
            if not dataRefListTract2:
                self.log.info("No data found in --rerun2 for tract: {:d}".format(tractList[i]))
                continue
            # Get a butler and dataId for each dataset.  Needed for feeding a
            # butler and camera into the plotting functions (for labelling the
            # camera and plotting ccd outlines) in addition to determining if
            # the data were processed with the HSC stack.  We assume all
            # processing in a given rerun is self-consistent, so only need one
            # valid dataId per comparison rerun.
            repoInfo1 = getRepoInfo(dataRefListTract1[0], catDataset=dataset1,
                                    doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib1,
                                    externalPhotoCalibName=self.config.externalPhotoCalibName1,
                                    doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs1,
                                    externalSkyWcsName=self.config.externalSkyWcsName1)
            repoInfo2 = getRepoInfo(dataRefListTract2[0], catDataset=dataset2,
                                    doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib2,
                                    externalPhotoCalibName=self.config.externalPhotoCalibName2,
                                    doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs2,
                                    externalSkyWcsName=self.config.externalSkyWcsName2)
            fullCameraCcdList1 = getCcdNameRefList(dataRefListTract1)

            existsRefList1, existsCcdList1 = getDataExistsRefList(dataRefListTract1, repoInfo1.catDataset)
            existsRefList2, existsCcdList2 = getDataExistsRefList(dataRefListTract2, repoInfo2.catDataset)
            if not existsCcdList1:
                raise RuntimeError(f"No datasets found for datasetType = {repoInfo1.catDataset}")
            if not existsCcdList2:
                raise RuntimeError(f"No datasets found for datasetType = {repoInfo2.catDataset}")

            if self.config.doApplyExternalPhotoCalib1:
                _, ccdPhotoCalibListPerTract1 = getDataExistsRefList(dataRefListTract1,
                                                                     repoInfo1.photoCalibDataset)
                if not ccdPhotoCalibListPerTract1:
                    self.log.fatal(f"No data found for {repoInfo1.photoCalibDataset} dataset...are you "
                                   "sure you ran the external calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib1=False")
            if self.config.doApplyExternalPhotoCalib2:
                _, ccdPhotoCalibListPerTract2 = getDataExistsRefList(dataRefListTract2,
                                                                     repoInfo2.photoCalibDataset)
                if not ccdPhotoCalibListPerTract2:
                    self.log.fatal(f"No data found for {repoInfo2.photoCalibDataset} dataset...are you "
                                   "sure you ran the external calibration?  If not, run with "
                                   "--config doApplyExternalPhotoCalib2=False")
            if self.config.doApplyExternalSkyWcs1:
                # Check for wcs for compatibility with old dataset naming
                _, ccdSkyWcsListPerTract1 = getDataExistsRefList(dataRefListTract1, repoInfo1.skyWcsDataset)
                if not ccdSkyWcsListPerTract1:
                    _, ccdSkyWcsListPerTract1 = getDataExistsRefList(dataRefListTract1, "wcs")
                    if ccdSkyWcsListPerTract1:
                        repoInfo1.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo1.wcsSkyDataset} dataset...are you "
                                       "sure you ran the external astrometric calibration?  If not, run "
                                       "with --config doApplyExternalSkyWcs1=False")
            if self.config.doApplyExternalSkyWcs2:
                # Check for wcs for compatibility with old dataset naming
                _, ccdSkyWcsListPerTract2 = getDataExistsRefList(dataRefListTract2, repoInfo2.skyWcsDataset)
                if not ccdSkyWcsListPerTract2:
                    _, ccdSkyWcsListPerTract2 = getDataExistsRefList(dataRefListTract2, "wcs")
                    if ccdSkyWcsListPerTract2:
                        repoInfo2.skyWcsDataset = "wcs"
                        self.log.info("Old meas_mosaic dataset naming: wcs (new name is jointcal_wcs)")
                    else:
                        self.log.fatal(f"No data found for {repoInfo2.wcsSkyDataset} dataset...are you "
                                       "sure you ran the external astrometric calibration?  If not, run "
                                       "with --config doApplyExternalSkyWcs2=False")

            ccdIntersectList = list(set(existsCcdList1).intersection(set(existsCcdList2)))
            self.log.info("tract: {:d}".format(repoInfo1.dataId["tract"]))
            self.log.info(f"existsCcdList1: \n{existsCcdList1}")
            self.log.info(f"existsCcdList2: \n{existsCcdList2}")
            self.log.info(f"ccdIntersectList: \n{ccdIntersectList}")
            if self.config.doApplyExternalPhotoCalib1:
                self.log.info(f"ccdPhotoCalibListPerTract1: \n{ccdPhotoCalibListPerTract1}")
            if self.config.doApplyExternalPhotoCalib2:
                self.log.info(f"ccdPhotoCalibListPerTract2: \n{ccdPhotoCalibListPerTract2}")
            if self.config.doApplyExternalSkyWcs1:
                self.log.info(f"ccdSkyWcsListPerTract1: \n{ccdSkyWcsListPerTract1}")
            if self.config.doApplyExternalSkyWcs2:
                self.log.info(f"ccdSkyWcsListPerTract2: \n{ccdSkyWcsListPerTract2}")

            # Set some aliases for differing schema naming conventions
            aliasDictList = [self.config.flagsToAlias, ]
            if (repoInfo1.hscRun or repoInfo2.hscRun) and self.config.srcSchemaMap is not None:
                aliasDictList += [self.config.srcSchemaMap]

            externalCalKwargs1 = dict(doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib1,
                                      doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs1,
                                      useMeasMosaic=self.config.useMeasMosaic1, iCat=0)
            externalCalKwargs2 = dict(doApplyExternalPhotoCalib=self.config.doApplyExternalPhotoCalib2,
                                      doApplyExternalSkyWcs=self.config.doApplyExternalSkyWcs2,
                                      useMeasMosaic=self.config.useMeasMosaic2, iCat=1)
            if self.config.doReadParquetTables1 or self.config.doReadParquetTables2:
                if self.config.doReadParquetTables1:
                    catalog1, commonZpCat1 = self.readParquetTables(existsRefList1, repoInfo1.catDataset,
                                                                    repoInfo1, **externalCalKwargs1)
                    areaDict1, _ = computeAreaDict(repoInfo1, existsRefList1, dataset="", fakeCat=None)

                if self.config.doReadParquetTables2:
                    catalog2, commonZpCat2 = self.readParquetTables(existsRefList2, repoInfo2.catDataset,
                                                                    repoInfo2, **externalCalKwargs2)
            if not self.config.doReadParquetTables1 or not self.config.doReadParquetTables2:
                if not self.config.doReadParquetTables1:
                    catStruct1 = self.readCatalogs(existsRefList1, dataset1, repoInfo1,
                                                   aliasDictList=aliasDictList, fakeCat=None,
                                                   readFootprintsAs=self.config.readFootprintsAs,
                                                   **externalCalKwargs1)
                    commonZpCat1 = catStruct1.commonZpCatalog
                    catalog1 = catStruct1.catalog
                    areaDict1 = catStruct1.areaDict
                    # Convert to pandas DataFrames
                    commonZpCat1 = commonZpCat1.asAstropy().to_pandas().set_index("id", drop=False)
                    catalog1 = catalog1.asAstropy().to_pandas().set_index("id", drop=False)
                if not self.config.doReadParquetTables2:
                    catStruct2 = self.readCatalogs(existsRefList2, dataset2, repoInfo2,
                                                   aliasDictList=aliasDictList, fakeCat=None,
                                                   readFootprintsAs=self.config.readFootprintsAs,
                                                   **externalCalKwargs2)
                    commonZpCat2 = catStruct2.commonZpCatalog
                    catalog2 = catStruct2.catalog
                    # Convert to pandas DataFrames
                    commonZpCat2 = commonZpCat2.asAstropy().to_pandas().set_index("id", drop=False)
                    catalog2 = catalog2.asAstropy().to_pandas().set_index("id", drop=False)

            self.zpLabel = self.zpLabel1 + "\nzp: " + self.zpLabel2

            # Set boolean arrays indicating sources deemed unsuitable for qa
            # analyses.
            self.catLabel = "isPrimary"
            bad1 = makeBadArray(catalog1, flagList=self.config.analysis.flags,
                                onlyReadStars=self.config.onlyReadStars)
            bad2 = makeBadArray(catalog2, flagList=self.config.analysis.flags,
                                onlyReadStars=self.config.onlyReadStars)
            badCommonZp1 = makeBadArray(commonZpCat1, flagList=self.config.analysis.flags,
                                        onlyReadStars=self.config.onlyReadStars)
            badCommonZp2 = makeBadArray(commonZpCat2, flagList=self.config.analysis.flags,
                                        onlyReadStars=self.config.onlyReadStars)

            # purge the catalogs of flagged sources
            catalog1 = catalog1[~bad1].copy(deep=True)
            catalog2 = catalog2[~bad2].copy(deep=True)
            commonZpCat1 = commonZpCat1[~badCommonZp1].copy(deep=True)
            commonZpCat2 = commonZpCat2[~badCommonZp2].copy(deep=True)
            self.log.info("\nNumber of sources in catalogs: first = {0:d} and second = {1:d}".format(
                          len(catalog1), len(catalog2)))

            commonZpCat = matchAndJoinCatalogs(commonZpCat1, commonZpCat2, matchRadius=self.matchRadius,
                                               matchXy=self.config.matchXy, camera1=repoInfo1,
                                               camera2=repoInfo2)
            catalog = matchAndJoinCatalogs(catalog1, catalog2, matchRadius=self.matchRadius,
                                           matchXy=self.config.matchXy, camera1=repoInfo1, camera2=repoInfo2)

            self.log.info("Number [fraction] of matches (maxDist = {0:.2f}{1:s}) = {2:d} [{3:d}%]".
                          format(self.matchRadius, self.matchRadiusUnitStr, len(catalog),
                                 int(100*len(catalog)/len(catalog1))))

            subdir = "ccd-" + str(existsCcdList1[0]) if len(ccdIntersectList) == 1 else subdir
            hscRun = repoInfo1.hscRun if repoInfo1.hscRun else repoInfo2.hscRun
            schema = getSchema(catalog)

            # Dict of all parameters common to plot* functions
            tractInfo1 = repoInfo1.tractInfo if self.config.doApplyExternalPhotoCalib1 else None
            # Always highlight points with x-axis flag set (for cases where
            # they do not get explicitly filtered out).
            highlightList = [(self.config.analysis.fluxColumn.replace("_instFlux", "_flag"), 0,
                              "turquoise"), ]
            plotKwargs1 = dict(matchRadius=self.matchRadius, matchRadiusUnitStr=self.matchRadiusUnitStr,
                               zpLabel=self.zpLabel, forcedStr=self.catLabel, highlightList=highlightList)

            plotInfoDict = getPlotInfo(repoInfo1)
            try:
                rerun2Str = list(repoInfo2.butler.storage.repositoryCfgs)[0]
            except AttributeError:
                rootDir = str(repoInfo2.butler.datastore.root)
                rootDir = rootDir.replace("file://", "")
                rerun2Str = rootDir + repoInfo2.butler.collections[0]
            plotInfoDict.update({"ccdList": ccdIntersectList, "allCcdList": fullCameraCcdList1,
                                 "plotType": "plotCompareVisit", "subdir": subdir,
                                 "hscRun1": repoInfo1.hscRun, "hscRun2": repoInfo2.hscRun,
                                 "hscRun": hscRun, "tractInfo": tractInfo1, "dataId": repoInfo1.dataId,
                                 "rerun2": rerun2Str})
            plotList = []
            if self.config.doPlotFootprintArea:
                plotList.append(self.plotFootprint(catalog, plotInfoDict, areaDict1, **plotKwargs1))

            # Create mag comparison plots using common ZP
            if not commonZpDone:
                zpLabel = "common (" + str(self.config.analysis.commonZp) + ")"
                plotKwargs1.update(dict(zpLabel=zpLabel))
                plotList.append(self.plotMags(commonZpCat, plotInfoDict, areaDict1,
                                              fluxToPlotList=["base_GaussianFlux",
                                                              "base_CircularApertureFlux_12_0"],
                                              postFix="_commonZp", **plotKwargs1))
                commonZpDone = True

            plotKwargs1.update(dict(zpLabel=self.zpLabel))
            if self.config.doPlotMags:
                plotList.append(self.plotMags(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotSizes:
                if ("first_base_SdssShape_psf_xx" in schema and "second_base_SdssShape_psf_xx" in schema):
                    plotList.append(self.plotSizes(catalog, plotInfoDict, areaDict1, **plotKwargs1))
                else:
                    self.log.warn("Cannot run plotSizes: base_SdssShape_psf_xx not in the catalog schema")
            if self.config.doApCorrs:
                plotList.append(self.plotApCorrs(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotCentroids:
                plotList.append(self.plotCentroids(catalog, plotInfoDict, areaDict1, **plotKwargs1))
            if self.config.doPlotStarGalaxy:
                plotList.append(self.plotStarGal(catalog, plotInfoDict, areaDict1, **plotKwargs1))

            self.allStats, self.allStatsHigh = savePlots(plotList, "plotCompareVisit", repoInfo1.dataId,
                                                         repoInfo1.butler, subdir=subdir)
