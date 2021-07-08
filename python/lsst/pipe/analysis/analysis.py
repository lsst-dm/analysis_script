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
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import re
import astropy.units as u

import lsst.afw.geom as afwGeom
import lsst.geom as geom
from lsst.pex.config import Config, Field, ListField, DictField
from lsst.pipe.base import Struct

from .utils import (Data, Stats, E1Resids, E2Resids, fluxToPlotString, computeMeanOfFrac,
                    calcQuartileClippedStats, RhoStatistics, measureRhoMetrics, addMetricMeasurement,
                    updateVerifyJob, getSchema)
from .plotUtils import (annotateAxes, AllLabeller, setPtSize, labelVisit, plotText, plotCameraOutline,
                        plotTractOutline, plotPatchOutline, plotCcdOutline, labelCamera, getQuiver,
                        plotRhoStats, getRaDecMinMaxPatchList, bboxToXyCoordLists, makeAlphaCmap)

matplotlib.use("Agg")
np.seterr(all="ignore")

__all__ = ["AnalysisConfig", "Analysis"]

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", "purple", "deeppink", "orange"]
# List of string replacement mappings to shorten text of labels
strMappingList = [("merge_measurement", "ref"), ("src_", ""), ("base_", ""), ("Flux", ""), ("_flag", "Flag"),
                  ("CircularAperture", "CircAp"), ("_12_0", "12"), ("modelfit_", ""), ("saturated", "sat"),
                  ("ClassificationExtendedness", "ClassnExt"), ("PixelFlags", "Pix"), ("Center", "Cent"),
                  ("_sat", "Sat"), ("slot_", "slot")]


class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "slot_Centroid_flag", "base_PsfFlux_flag",
                               "base_PixelFlags_flag_saturatedCenter",
                               "base_ClassificationExtendedness_flag"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    useSignalToNoiseThreshold = Field(dtype=bool, default=True, doc="Use a Signal-to-Noise threshold "
                                      "to set the limit for the statistics computation?  If True, the "
                                      "value set in signalToNoiseThreshold is used directly for single "
                                      "frame (visit) catalogs, but is scaled by sqrt(NumberOfVisits) for "
                                      "coadd catalogs (to ensure a similar effective magnitude cut for "
                                      "both).  If False, the cut will be based on magnitude using the "
                                      "value in magThreshold for both visit and coadd catalogs.")
    signalToNoiseThreshold = Field(dtype=float, default=100.0, doc="Signal-to-Noise threshold to apply")
    signalToNoiseHighThreshold = Field(dtype=float, default=500.0, doc="Signal-to-Noise threshold to apply "
                                       "as representative of a \"high\" S/N sample that is always computed "
                                       "(i.e. regardless of useSignalToNoiseThreshold).  The value is "
                                       "used directly for single frame (visit) catalogs, but is scaled by "
                                       "sqrt(NumberOfVisits) for coadd catalogs (to ensure a similar "
                                       "effective magnitude cut for both).")
    minHighSampleN = Field(dtype=int, default=20, doc="Minimum number of stars for the "
                           "signalToNoiseHighThreshold sample.  If too few objects classified as stars "
                           "exist with the configured value, decrease the S/N threshold by 10 until a "
                           "sample with N > minHighSampleN is achieved.")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=14.5, doc="Fallback minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=25.99, doc="Fallback maximum magnitude to plot")
    magPlotStarMin = DictField(
        keytype=str,
        itemtype=float,
        default={"HSC-G": 16.5, "HSC-R": 17.25, "HSC-I": 16.5, "HSC-Z": 15.5, "HSC-Y": 15.25,
                 "NB0921": 15.0, "u": 13.75, "g": 15.5, "r": 15.0, "i": 15.5, "z": 14.5, "y": 14.0},
        doc="Minimum magnitude to plot",
    )
    magPlotStarMax = DictField(
        keytype=str,
        itemtype=float,
        default={"HSC-G": 23.75, "HSC-R": 24.25, "HSC-I": 23.75, "HSC-Z": 23.0, "HSC-Y": 22.0,
                 "NB0921": 22.25, "u": 21.5, "g": 22.5, "r": 22.0, "i": 21.5, "z": 21.0, "y": 20.0},
        doc="Maximum magnitude to plot",
    )
    fluxColumn = Field(dtype=str, default="modelfit_CModel_instFlux",
                       doc="Column to use for flux/mag plotting")
    coaddZp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply for coadds")
    commonZp = Field(dtype=float, default=33.0, doc="Magnitude zero point to apply for common ZP plots")
    doPlotOldMagsHist = Field(dtype=bool, default=False, doc="Make older, separated, mag and hist plots?")
    doPlotRaDec = Field(dtype=bool, default=False, doc="Make delta vs. RA and Dec plots?")
    doPlotFP = Field(dtype=bool, default=False, doc="Make FocalPlane plots?")
    doPlotCcdXy = Field(dtype=bool, default=False, doc="Make plots as a function of CCD x and y?")
    doPlotTractOutline = Field(dtype=bool, default=True, doc="Plot tract outline (may be a bit slow)?")
    visitClassFluxRatio = Field(dtype=float, default=0.95,
                                doc="Flux ratio for visit level star/galaxy classifiaction")
    coaddClassFluxRatio = Field(dtype=float, default=0.985,
                                doc="Flux ratio for coadd level star/galaxy classifiaction")
    doLabelRerun = Field(dtype=bool, default=True, doc="Include label indicating rerun direcotry on plots?")


class Analysis(object):
    """Centralised base for plotting.
    """
    def __init__(self, catalog, func, quantityName, shortName, config, qMin=-0.2, qMax=0.2,
                 prefix="", flags=[], goodKeys=[], errFunc=None, labeller=AllLabeller(),
                 magThreshold=None, forcedMean=None, unitScale=1.0, compareCat=None, fluxColumn=None):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.shortName = shortName
        self.config = config
        self.forcedMean = forcedMean
        if magThreshold is None:
            self.magThreshold = self.config.magThreshold
        else:
            self.magThreshold = magThreshold
        self.unitScale = unitScale
        self.qMin = qMin*self.unitScale
        self.qMax = qMax*self.unitScale
        if "modelfit" in self.shortName:  # Cmodel has smaller mean/scatter from psf mag for most objects
            self.qMin /= 2.0
            self.qMax /= 2.0
        self.goodKeys = goodKeys  # include if goodKey = True
        self.calibUsedOnly = len([key for key in self.goodKeys if "used" in key])
        self.prefix = prefix
        self.errFunc = errFunc
        if func is not None:
            if isinstance(func, np.ndarray) or isinstance(func, pd.Series):
                self.quantity = func
            else:
                self.quantity = func(catalog)
        else:
            self.quantity = None

        self.quantityError = errFunc(catalog) if errFunc is not None else None
        schema = getSchema(catalog)
        self.fluxColumn = fluxColumn
        if not fluxColumn:
            if prefix + self.config.fluxColumn in schema:
                self.fluxColumn = self.config.fluxColumn
            else:
                self.fluxColumn = "flux_psf_flux"
        self.mag = -2.5*np.log10(catalog[prefix + self.fluxColumn])

        self.good = (np.isfinite(self.quantity) & np.isfinite(self.mag) if self.quantity is not None
                     else np.isfinite(self.mag))
        if any(ss in self.shortName for ss in ["skySources", "skyObjects"]) and self.quantity is not None:
            self.good = np.isfinite(self.quantity)
        if errFunc is not None:
            self.good &= np.isfinite(self.quantityError)

        # Skip flag culling on the macth and overlap catalogs: we want to look
        # at any/all matches found (objects with any notable flags set will be
        # highlighted in the plot), and the latter are already culled.  Also,
        # if sub-selecting a calib_*_used sample, we want to look at all
        # objects used in the visit-level calibrations, so do not cull on the
        # standard self.config.flags.  Rather, only cull on flags explicitly
        # set in the flags variable for calib_*_used subsamples.
        if not any(ss in self.shortName for ss in ["matches", "overlap", "quiver", "inputCounts",
                                                   "skyObjects", "skySources"]):
            flagsList = flags.copy()
            flagsList = flagsList + list(self.config.flags) if self.calibUsedOnly == 0 else flagsList
            for flagName in set(flagsList):
                if prefix + flagName in schema:
                    self.good &= ~catalog[prefix + flagName]
        for flagName in goodKeys:
            self.good &= catalog[prefix + flagName]

        # If the input catalog is a coadd, scale the S/N threshold by roughly
        # the sqrt of the number of input visits (actually the mean of the
        # upper 10% of the base_InputCount_value distribution).
        if prefix + "base_InputCount_value" in schema:
            inputCounts = catalog[prefix + "base_InputCount_value"]
            scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=10)
            if scaleFactor == 0.0:
                scaleFactor = computeMeanOfFrac(inputCounts, tailStr="upper", fraction=0.1, floorFactor=1)
            self.signalToNoiseThreshold = np.floor(
                np.sqrt(scaleFactor)*self.config.signalToNoiseThreshold/100 + 0.49)*100
            self.signalToNoiseHighThreshold = np.floor(
                np.sqrt(scaleFactor)*self.config.signalToNoiseHighThreshold/100 + 0.49)*100
        else:
            self.signalToNoiseThreshold = self.config.signalToNoiseThreshold
            self.signalToNoiseHighThreshold = self.config.signalToNoiseHighThreshold
        if "galacticExtinction" in self.shortName and self.magThreshold > 90.0:
            self.signalToNoiseThreshold = 0.0
            self.signalToNoiseHighThreshold = 0.0

        if any(ss in self.shortName for ss in ["skySources", "skyObjects"]):
            # We don't want to cull on S/N for the sky sources and they can be
            # negative, so set threshold to a very large negative number.
            self.signalToNoiseThreshold = -1e30
            self.signalToNoiseHighThreshold = -1e30
            if self.quantity is not None:
                self.good = np.isfinite(self.quantity)
        self.signalToNoise = catalog[prefix + self.fluxColumn]/catalog[prefix + self.fluxColumn + "Err"]
        self.signalToNoiseStr = None
        goodSn0 = np.isfinite(self.signalToNoise)
        if self.good is not None:
            goodSn0 = np.logical_and(self.good, goodSn0)
        if self.config.useSignalToNoiseThreshold:
            self.signalToNoiseStr = r"[S/N$\geqslant${0:}]".format(int(self.signalToNoiseThreshold))
            goodSn = np.logical_and(goodSn0, self.signalToNoise >= self.signalToNoiseThreshold)
            # Set self.magThreshold to represent approximately that which
            # corresponds to the S/N threshold.  Computed as the mean mag
            # of the lower 5% of the S/N > signalToNoiseThreshold
            # subsample.
            if self.magThreshold < 90.0:
                self.magThreshold = computeMeanOfFrac(self.mag[goodSn], tailStr="upper", fraction=0.05,
                                                      floorFactor=0.1)

        # Always compute stats for S/N > config.signalToNoiseHighThreshold.
        # If too few objects classified as stars exist with the configured
        # value, decrease the S/N threshold by 10 until a sample with
        # N > self.config.minHighSampleN is achieved.
        goodSnHigh = np.logical_and(goodSn0, self.signalToNoise >= self.signalToNoiseHighThreshold)
        if prefix + "base_ClassificationExtendedness_value" in schema:
            isStar = catalog[prefix + "base_ClassificationExtendedness_value"] < 0.5
        elif "numStarFlags" in schema:
            isStar = catalog["numStarFlags"] >= 3
        else:
            isStar = np.ones(len(self.mag), dtype=bool)
            print("Warning: No star/gal flag found")
        goodSnHighStars = np.logical_and(goodSnHigh, isStar)
        while(len(self.mag[goodSnHighStars]) < self.config.minHighSampleN
              and self.signalToNoiseHighThreshold > 0.0):
            self.signalToNoiseHighThreshold -= 10.0
            goodSnHigh = np.logical_and(goodSn0, self.signalToNoise >= self.signalToNoiseHighThreshold)
            goodSnHighStars = np.logical_and(goodSnHigh, isStar)
        self.magThresholdHigh = computeMeanOfFrac(self.mag[goodSnHigh], tailStr="upper", fraction=0.1,
                                                  floorFactor=0.1)
        self.signalToNoiseHighStr = r"[S/N$\geqslant${0:}]".format(int(self.signalToNoiseHighThreshold))

        # Select a sample for setting plot limits: "good" based on flags, S/N
        # is finite and >= 2.0.  Limits are the means of the bottom 1% and top
        # 5% of this sample with a 0.5 mag buffer on either side.
        goodSn0 &= self.signalToNoise >= 2.0
        self.magMin = (computeMeanOfFrac(self.mag[goodSn0], tailStr="lower", fraction=0.005, floorFactor=1)
                       - 1.5)
        self.magMax = (computeMeanOfFrac(self.mag[goodSn0], tailStr="upper", fraction=0.05, floorFactor=1)
                       + 0.5)

        if labeller is not None:
            labels = labeller(catalog, compareCat) if compareCat is not None else labeller(catalog)
            self.data = {name: Data(catalog, self.quantity, self.mag, self.signalToNoise,
                                    self.good & (labels == value),
                                    colorList[value], self.quantityError, name in labeller.plot) for
                         name, value in labeller.labels.items()}
            # Sort data dict by number of points in each data type.
            self.data = {k: self.data[k] for _, k in sorted(((len(v.mag), k) for (k, v) in self.data.items()),
                                                            reverse=True)}
            if self.config.useSignalToNoiseThreshold:
                self.stats = self.statistics(signalToNoiseThreshold=self.signalToNoiseThreshold,
                                             forcedMean=forcedMean)
            else:
                self.stats = self.statistics(magThreshold=self.magThreshold, forcedMean=forcedMean)
            self.statsHigh = self.statistics(signalToNoiseThreshold=self.signalToNoiseHighThreshold,
                                             forcedMean=forcedMean)
            # Ensure plot limits always encompass at least mean +/- 6.0*stdev,
            # at most mean +/- 20.0*stddev, and clipped stats range + 25%.
            dataType = "all" if "all" in self.data else "star"
            nStdevMin = 6.0 if "matches" not in self.shortName else 10.0
            nStdevMax = 20.0
            if self.stats[dataType].num > 0:
                if not any(ss in self.shortName for ss in ["footArea", "distance", "pStar", "resolution",
                                                           "race", "psfInst", "psfCal", "Rho", "hsmRho",
                                                           "Rho_calib_psf_used", "hsmRho_calib_psf_used",
                                                           "Rho_all_stars", "hsmRho_all_stars"]):
                    self.qMin = max(min(self.qMin,
                                        self.stats[dataType].mean - nStdevMin*self.stats[dataType].stdev,
                                        self.stats[dataType].median - 1.25*self.stats[dataType].clip),
                                    min(self.stats[dataType].mean - nStdevMax*self.stats[dataType].stdev,
                                        -0.005*self.unitScale))
                    if (abs(self.stats[dataType].mean) < 0.0005*self.unitScale
                            and abs(self.stats[dataType].stdev) < 0.0005*self.unitScale):
                        minmax = 2.0*max(abs(min(self.quantity[self.good])),
                                         abs(max(self.quantity[self.good])))
                        self.qMin = -minmax if minmax > 0 else self.qMin
                if not any(ss in self.shortName for ss in ["footArea", "pStar", "resolution", "race",
                                                           "psfInst", "psfCal", "Rho", "hsmRho",
                                                           "Rho_calib_psf_used", "hsmRho_calib_psf_used",
                                                           "Rho_all_stars", "hsmRho_all_stars"]):
                    self.qMax = min(max(self.qMax,
                                        self.stats[dataType].mean + nStdevMin*self.stats[dataType].stdev,
                                        self.stats[dataType].median + 1.25*self.stats[dataType].clip),
                                    max(self.stats[dataType].mean + nStdevMax*self.stats[dataType].stdev,
                                        0.005*self.unitScale))
                    if (abs(self.stats[dataType].mean) < 0.0005*self.unitScale
                            and abs(self.stats[dataType].stdev) < 0.0005*self.unitScale):
                        minmax = 2.0*max(abs(min(self.quantity[self.good])),
                                         abs(max(self.quantity[self.good])))
                        self.qMax = minmax if minmax > 0 else self.qMax

    def plotAgainstMag(self, description, plotInfoDict, stats=None, matchRadius=None, matchRadiusUnitStr=None,
                       zpLabel=None, forcedStr=None, doPrintMedian=False):
        """Plot quantity against magnitude.
        """
        fig, axes = plt.subplots(1, 1)
        plt.axhline(0, linestyle="--", color="0.4")
        if self.magThreshold > 90.0:
            magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        else:
            magMin, magMax = self.magMin, self.magMax
        dataPoints = []
        ptSize = None
        for name, data in self.data.items():
            if not data.mag.any():
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            dataPoints.append(axes.scatter(data.mag, data.quantity, s=ptSize, marker="o", lw=0,
                                           c=data.color, label=name, alpha=0.3))
        axes.set_xlabel("Mag from %s" % self.fluxColumn)
        axes.set_ylabel(self.quantityName)
        axes.set_ylim(self.qMin, self.qMax)
        axes.set_xlim(magMin, magMax)
        if stats is not None:
            annotateAxes(description, axes, stats, "star", self.magThreshold,
                         signalToNoiseStr=self.signalToNoiseStr, statsHigh=self.statsHigh,
                         magThresholdHigh=self.magThresholdHigh,
                         signalToNoiseHighStr=self.signalToNoiseHighStr,
                         matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian)
        axes.legend(handles=dataPoints, loc=1, fontsize=8)
        labelVisit(plotInfoDict, fig, axes, 0.5, 1.05)
        if zpLabel is not None:
            prefix = "" if "GalExt" in zpLabel else "zp: "
            plotText(zpLabel, fig, axes, 0.13, -0.09, prefix=prefix, color="green")
        if forcedStr is not None:
            plotText(forcedStr, fig, axes, 0.85, -0.10, prefix="cat: ", color="green")
        yield Struct(fig=fig, description=description, stats=self.stats, statsHigh=self.statsHigh, dpi=120,
                     style="psfMag")

    def plotAgainstMagAndHist(self, log, description, plotInfoDict, stats=None, matchRadius=None,
                              matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, plotRunStats=True,
                              highlightList=None, extraLabels=None, uberCalLabel=None, doPrintMedian=False):
        """Plot quantity against magnitude with side histogram.
        """
        if plotInfoDict["plotType"] != "plotColor":
            filterLabelStr = "[{}]".format(plotInfoDict["filter"])
            if "lsst" in plotInfoDict["cameraName"]:
                filterLabelStr = "[{}-{}]".format(plotInfoDict["cameraName"], plotInfoDict["filter"])
        else:
            filterLabelStr = ""

        nullfmt = NullFormatter()  # no labels for histograms
        # definitions for the axes
        left, width = 0.12, 0.62
        bottom, height = 0.10, 0.62
        left_h = left + width + 0.02
        bottom_h = bottom + width + 0.03
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.22]
        rect_histy = [left_h, bottom, 0.18, height]
        topRight = [left_h - 0.013, bottom_h, 0.22, 0.22]

        axScatter = plt.axes(rect_scatter)
        axScatter.axhline(0, linestyle="--", color="0.4")
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)
        axHistx.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        axHisty.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        axScatter.tick_params(which="both", direction="in", labelsize=9)

        if "Visit" in plotInfoDict["plotType"]:
            ccdList = plotInfoDict["ccdList"]
        if ("Visit" in plotInfoDict["plotType"] and plotInfoDict["camera"] is not None
                and ccdList is not None):
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotCameraOutline(axTopRight, plotInfoDict["camera"], ccdList)

        tractLevelPlot = "Coadd" in plotInfoDict["plotType"] or "Color" in plotInfoDict["plotType"]
        if self.config.doPlotTractOutline and tractLevelPlot:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotTractOutline(axTopRight, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"])

        dataType = "all" if "all" in self.data else "star"
        inLimits = self.data[dataType].quantity < self.qMax
        inLimits &= self.data[dataType].quantity > self.qMin
        if self.data[dataType].quantity.any():
            if len(self.data[dataType].quantity[inLimits]) < max(1.0, 0.35*len(self.data[dataType].quantity)):
                log.info("plotAgainstMagAndHist: No data within limits...decreasing/increasing qMin/qMax")
            while (len(self.data[dataType].quantity[inLimits])
                   < max(1.0, 0.35*len(self.data[dataType].quantity))):
                self.qMin -= 0.1*np.abs(self.qMin)
                self.qMax += 0.1*self.qMax
                inLimits = self.data[dataType].quantity < self.qMax
                inLimits &= self.data[dataType].quantity > self.qMin

        # Make sure plot limit extends low enough to show well below the
        # star/galaxy separation line.  Add delta as opposed to directly
        # changing self.qMin to not affect other plots.
        deltaMin = 0.0
        if "galaxy" in self.data and len(self.data["galaxy"].quantity) > 0 and "-mag_" in description:
            if "GaussianFlux" in description:
                galMin = np.round(2.5*np.log10(self.config.visitClassFluxRatio) - 0.08, 2)*self.unitScale
                deltaMin = max(0.0, self.qMin - galMin)
            if "CModel" in description:
                galMin = np.round(2.5*np.log10(self.config.coaddClassFluxRatio) - 0.08, 2)*self.unitScale
                deltaMin = max(0.0, self.qMin - galMin)

        if "matches" in self.shortName and not tractLevelPlot:
            magMin = self.magMin + 1.0  # the -1.5 shift above is a bit much in this case
            if "gaia" in forcedStr:
                magMax = min(self.config.magPlotStarMax[plotInfoDict["filter"]], self.magMax)
            else:
                magMax = max(self.config.magPlotStarMax[plotInfoDict["filter"]], self.magMax)
        elif self.magThreshold > 90.0:
            magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        else:
            magMin, magMax = self.magMin, self.magMax

        axScatter.set_xlim(magMin, magMax)
        yDelta = 0.01*(self.qMax - (self.qMin - deltaMin))
        axScatter.set_ylim((self.qMin - deltaMin) + yDelta, self.qMax - yDelta)

        # Get current y limits for scatter plot to base histogram bin sizing on
        axScatterY1 = axScatter.get_ylim()[0]
        axScatterY2 = axScatter.get_ylim()[1]

        nxDecimal = int(-1.0*np.around(np.log10(0.05*abs(magMax - magMin)) - 0.5))
        xBinwidth = min(0.1, np.around(0.05*abs(magMax - magMin), nxDecimal))
        xBins = np.arange(magMin + 0.5*xBinwidth, magMax + 0.5*xBinwidth, xBinwidth)
        nyDecimal = int(-1.0*np.around(np.log10(0.05*abs(axScatterY2 - (axScatterY1 - deltaMin))) - 0.5))
        yBinwidth = max(0.5/10**nyDecimal, np.around(0.02*abs(axScatterY2 - (axScatterY1 - deltaMin)),
                                                     nyDecimal))
        yBins = np.arange((axScatterY1 - deltaMin) - 0.5*yBinwidth, axScatterY2 + 0.55*yBinwidth, yBinwidth)
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
        axHistx.set_yscale("log", nonpositive="clip")
        axHisty.set_xscale("log", nonpositive="clip")
        nTotal = 0
        fullSampleMag = []
        fullSampleQuantity = []
        for name, data in self.data.items():
            if data.plot:
                nTotal += len(data.mag)
                fullSampleMag.extend(data.mag)
                fullSampleQuantity.extend(data.quantity)
        axScatterYlim = np.around(nTotal, -1*int(np.floor(np.log10(nTotal))))
        axHistx.set_ylim(0.8, axScatterYlim)
        axHisty.set_xlim(0.8, axScatterYlim)

        # Plot full sample histograms
        axHistx.hist(fullSampleMag, bins=xBins, color="black", alpha=0.4, label="All")
        axHisty.hist(fullSampleQuantity, bins=yBins, color="black", orientation="horizontal",
                     alpha=0.4, label="All")

        nxSyDecimal = int(-1.0*np.around(np.log10(0.05*abs(self.magThreshold - magMin)) - 0.5))
        xSyBinwidth = min(0.1, np.around(0.05*abs(self.magThreshold - magMin), nxSyDecimal))
        xSyBins = np.arange(magMin + 0.5*xSyBinwidth, self.magThreshold + 0.5*xSyBinwidth, xSyBinwidth)

        royalBlue = "#4169E1"
        cornflowerBlue = "#6495ED"

        # A call to axScatter.scatter() will have it's label included in the
        # legend if its return is appended to dataPoints.  Use as sparingly
        # as possible as long legend lists hide data.
        dataPoints = []
        runStats = []
        ptSize = None
        for name, data in self.data.items():
            if not data.plot:
                log.info("plotAgainstMagAndHist: Not plotting data for dataset: {0:s} (N = {1:d})".
                         format(name, len(data.mag)))
                continue
            if not data.mag.any():
                log.info("plotAgainstMagAndHist: No data for dataset: {:s}".format(name))
                continue
            schema = getSchema(data.catalog)
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            alpha = min(0.75, max(0.25, 1.0 - 0.2*np.log10(len(data.mag))))
            # draw mean and stdev at intervals (defined by xBins)
            histColor = "red"
            if name == "split" or name == "notStar":
                histColor = "green"
            if name == "unknown":
                histColor = "orange"
            if name == "star" or name == "all":
                histColor = royalBlue
                # shade the portion of the plot fainter that self.magThreshold
                axScatter.axvspan(self.magThreshold, axScatter.get_xlim()[1], facecolor="k",
                                  edgecolor="none", alpha=0.10)
                axScatter.axvspan(self.magThresholdHigh, axScatter.get_xlim()[1], facecolor="k",
                                  edgecolor="none", alpha=0.10)
                # compute running stats (just for plotting)
                if plotRunStats:
                    belowThresh = data.mag < magMax  # set lower if you want to truncate plotted running stats
                    numHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins))
                    # Only plot running stats if there are a significant number
                    # of data points per bin (as otherwise it looks too messy).
                    # This is computed as the mean number in the brightest
                    # 10-30% of the bins which we require to be greater than 12
                    # (these magic numbers were selected based on trial and
                    # error to be the best compromise between information and
                    # messy clutter).
                    if numHist[max(1, int(0.10*len(xSyBins))):max(2, int(0.3*len(xSyBins)))].mean() > 12:
                        syHist, dataHist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                        weights=data.quantity[belowThresh])
                        syHist2, datahist = np.histogram(data.mag[belowThresh], bins=len(xSyBins),
                                                         weights=data.quantity[belowThresh]**2)
                        meanHist = syHist/numHist
                        stdHist = np.sqrt(syHist2/numHist - meanHist*meanHist)
                        runStats.append(axScatter.errorbar((dataHist[1:] + dataHist[:-1])/2, meanHist,
                                        yerr=stdHist, fmt="o", mfc=cornflowerBlue, mec="k",
                                        ms=2, ecolor="k", elinewidth=0.7,
                                        label="Running\nstats (all\nstars)"))

            if highlightList is not None:
                # Make highlight as a background ring of larger size than the
                # data point size.
                sizeFactor = 1.3
                for flag, threshValue, color in highlightList:
                    if flag in schema:
                        highlightSelection = data.catalog[flag] > threshValue
                        if sum(highlightSelection) > 0:
                            label = flag
                            for k, v in strMappingList:
                                label = label.replace(k, v)
                            if not any(label == lab.get_label() for lab in dataPoints):
                                dataPoints.append(
                                    axScatter.scatter(data.mag[highlightSelection],
                                                      data.quantity[highlightSelection], s=sizeFactor*ptSize,
                                                      marker="o", facecolors="none", edgecolors=color,
                                                      linewidth=1.0/sizeFactor, label=label))
                            else:
                                axScatter.scatter(data.mag[highlightSelection],
                                                  data.quantity[highlightSelection], s=sizeFactor*ptSize,
                                                  marker="o", facecolors="none", edgecolors=color,
                                                  linewidth=1.0/sizeFactor)
                            sizeFactor *= 1.3
            # Plot data.  Append the axScatter.scatter() calls to dataPoints if
            # it's label is to be included in the legend.
            axScatter.scatter(data.mag, data.quantity, s=ptSize, marker="o",
                              facecolors=data.color, edgecolors="face",
                              label=name, alpha=alpha, linewidth=0.5)

            if stats is not None and (name == "star" or name == "all") and "foot" not in description:
                labelStr = self.signalToNoiseStr if self.signalToNoiseStr else "stats"
                axScatter.scatter(data.mag[stats[name].dataUsed],
                                  data.quantity[stats[name].dataUsed], s=ptSize,
                                  marker="o", facecolors="none", edgecolors=data.color,
                                  label=labelStr, alpha=1, linewidth=0.5)

            if self.statsHigh is not None and (name == "star" or name == "all") and "foot" not in description:
                axScatter.scatter(data.mag[self.statsHigh[name].dataUsed],
                                  data.quantity[self.statsHigh[name].dataUsed], s=ptSize,
                                  marker="o", facecolors=data.color, edgecolors="face",
                                  label=self.signalToNoiseHighStr, alpha=1, linewidth=0.5)

            axHistx.hist(data.mag, bins=xBins, color=histColor, alpha=0.6, label=name)
            axHisty.hist(data.quantity, bins=yBins, color=histColor, alpha=0.6, orientation="horizontal",
                         label=name)
        # Make sure stars used histogram is plotted last
        for name, data in self.data.items():
            if (name == "star" or name == "all") and "foot" not in description:
                if stats is not None:
                    labelStr = self.signalToNoiseStr if self.signalToNoiseStr else "stats"
                    axHisty.hist(data.quantity[stats[name].dataUsed], bins=yBins, facecolor="none",
                                 edgecolor=data.color, linewidth=0.5, orientation="horizontal",
                                 label=labelStr)
                    axHistx.hist(data.mag[stats[name].dataUsed], bins=xBins, facecolor="none",
                                 edgecolor=data.color, linewidth=0.5, label=labelStr)
                if self.statsHigh is not None and (name == "star" or name == "all"):
                    axHisty.hist(data.quantity[self.statsHigh[name].dataUsed], bins=yBins,
                                 color=data.color, orientation="horizontal", label=self.signalToNoiseHighStr)
                    axHistx.hist(data.mag[self.statsHigh[name].dataUsed], bins=xBins,
                                 color=data.color, linewidth=0.5, label=self.signalToNoiseHighStr)
        axHistx.tick_params(axis="x", which="major", direction="in", length=5)
        axHistx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axHisty.tick_params(axis="y", which="major", direction="in", length=5)
        axHisty.yaxis.set_minor_locator(AutoMinorLocator(2))

        axScatter.tick_params(which="major", direction="in", length=5)
        axScatter.xaxis.set_minor_locator(AutoMinorLocator(2))
        axScatter.yaxis.set_minor_locator(AutoMinorLocator(2))

        yLabel = r"%s %s" % (self.quantityName, filterLabelStr)
        fontSize = min(11, max(6, 11 - int(np.log(max(1, len(yLabel) - 45)))))

        axScatter.set_xlabel("%s mag %s" % (fluxToPlotString(self.fluxColumn), filterLabelStr), fontsize=11)
        axScatter.set_ylabel(yLabel, fontsize=fontSize)

        if stats is not None and "foot" not in description:
            l1, l2 = annotateAxes(description, axScatter, stats, dataType, self.magThreshold,
                                  signalToNoiseStrConf=self.signalToNoiseStr, statsHigh=self.statsHigh,
                                  magThresholdHigh=self.magThresholdHigh,
                                  signalToNoiseHighStr=self.signalToNoiseHighStr,
                                  matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                                  unitScale=self.unitScale, doPrintMedian=doPrintMedian)
            dataPoints = dataPoints + runStats + [l1, l2]
        legendFontSize = 7 if len(dataPoints) < 4 else 6
        axScatter.legend(handles=dataPoints, loc=1, fontsize=legendFontSize, labelspacing=0.3)
        axHistx.legend(fontsize=7, loc=2, edgecolor="w", labelspacing=0.2)
        axHisty.legend(fontsize=7, labelspacing=0.2)
        # Add axis with units of FWHM = 2*sqrt(2*ln(2))*Trace for Trace plots
        if ("race" in self.shortName and "iff" not in self.shortName
                and "ompare" not in plotInfoDict["plotType"]):
            axHisty2 = axHisty.twinx()  # instantiate a second axes that shares the same x-axis
            sigmaToFwhm = 2.0*np.sqrt(2.0*np.log(2.0))
            axHisty2.set_ylim(axScatterY1*sigmaToFwhm, axScatterY2*sigmaToFwhm)
            axHisty2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axHisty2.tick_params(axis="y", which="both", direction="in", labelsize=7)
            axHisty2.set_ylabel(r"FWHM: $2\sqrt{2\,ln\,2}*$Trace (pixels)", rotation=270, labelpad=12,
                                fontsize=fontSize)

        # Label total number of objects of each data type
        xLoc, yLoc = 0.09, 1.355
        lenNameMax = 0
        for name, data in self.data.items():
            if data.mag.any():
                lenNameMax = len(name) if len(name) > lenNameMax else lenNameMax
        xLoc += 0.02*lenNameMax

        plt.text(xLoc, yLoc, "N$_{all}$  = " + str(len(fullSampleMag)), ha="left", va="center",
                 fontsize=8, transform=axScatter.transAxes, color="black", alpha=0.6)
        for name, data in self.data.items():
            if not (data.mag.any() and data.plot):
                continue
            yLoc -= 0.045
            plt.text(xLoc, yLoc, "N$_{" + name[:4] + "}$ = " + str(len(data.mag)), ha="left", va="center",
                     fontsize=7, transform=axScatter.transAxes, color=data.color)

        labelVisit(plotInfoDict, plt.gcf(), axScatter, 1.18, -0.11, color="green")
        if self.config.doLabelRerun and not any(ss in self.shortName for ss in ["race-", "race_"]):
            rerunKwargs = dict(fontSize=6, color="purple", rotation=-90)
            if "rerun2" in plotInfoDict:
                plotText("rerun: " + plotInfoDict["rerun"], plt, axHisty, 1.18, 0.7, **rerunKwargs)
                plotText("rerun2: " + plotInfoDict["rerun2"], plt, axHisty, 1.08, 0.7, **rerunKwargs)
            else:
                rerunKwargs.update(fontSize=7)
                plotText("rerun: " + plotInfoDict["rerun"], plt, axHisty, 1.12, 0.7, **rerunKwargs)

        if zpLabel is not None:
            # The following sets yOff to accommodate the longer labels for the
            # compare scripts and/or for the presence of the extra uberCalLabel
            # (for coadds).  Font size is adjusted with uberFontSize below.
            yOff = 0.02 if uberCalLabel is not None else 0
            yOff = -0.02 if "_2" in zpLabel else yOff
            prefix = "" if "GalExt" in zpLabel else "zp: "
            plotText(zpLabel, plt.gcf(), axScatter, 0.11, -0.1 + yOff, prefix=prefix, fontSize=7,
                     color="green")
        if uberCalLabel:
            uberFontSize = 5 if "_2" in uberCalLabel else 7
            plotText(uberCalLabel, plt.gcf(), axScatter, 0.11, -0.13, fontSize=uberFontSize, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt.gcf(), axScatter, 0.86, -0.10, prefix="cat: ", fontSize=7, color="green")
        if extraLabels is not None:
            for i, extraLabel in enumerate(extraLabels):
                plotText(extraLabel, plt.gcf(), axScatter, 0.3, 0.21 + i*0.05, fontSize=7, color="tab:orange")
        yield Struct(fig=plt.gcf(), description=description, stats=self.stats, statsHigh=self.statsHigh,
                     dpi=120, style="psfMagHist")

    def plotHistogram(self, description, plotInfoDict, numBins=51, stats=None, zpLabel=None,
                      forcedStr=None, filterStr=None, magThreshold=None, matchRadius=None,
                      matchRadiusUnitStr=None, uberCalLabel=None, doPrintMedian=False, vertLineList=None,
                      logPlot=True, density=False, cumulative=False, addDataList=None, addDataLabelList=None):
        """Plot histogram of quantity.
        """
        fig, axes = plt.subplots(1, 1)
        axes.axvline(0, linestyle="--", color="0.6")
        if vertLineList:
            for xLine in vertLineList:
                axes.axvline(xLine, linestyle="--", color="0.6")
        numMin = 0 if density else 0.9
        numMax = 1
        alpha = 0.4
        ic = 1
        axes2 = None
        numGood = None
        for name, data in self.data.items():
            if not data.mag.any():
                continue
            color = "tab:" + data.color
            ic += 1
            good = np.isfinite(data.quantity)
            if magThreshold and stats is not None:
                good &= data.mag < magThreshold
            nValid = np.abs(data.quantity[good]) <= self.qMax  # need to have datapoints lying within range
            if good.sum() == 0 or nValid.sum() == 0:
                continue
            numGood, fluxBins, _ = axes.hist(data.quantity[good], bins=numBins, range=(self.qMin, self.qMax),
                                             density=density, log=logPlot, color=color, alpha=alpha,
                                             label=name, histtype="stepfilled")
            numBins = fluxBins if type(numBins) is str else numBins
            if cumulative:
                axes2 = axes.twinx()  # instantiate a second axes that shares the same x-axis
                axes2.hist(data.quantity[good], bins=numBins, density=True, log=False, color=data.color,
                           label=name + "_cum", histtype="step", cumulative=cumulative)
            else:
                axes2 = None
            # yaxis limit for non-normalized histograms
            numMax = max(numMax, numGood.max()*1.1) if not density else numMax
        if cumulative and axes2 is not None:
            axes2.set_ylim(0, 1.05)
            axes2.tick_params(axis="y", which="both", direction="in")
            axes2.set_ylabel("Cumulative Fraction", rotation=270, labelpad=12, color=color, fontsize=9)
            axes2.legend(loc="right", fontsize=8)
            axes2.grid(True, "both", color="black", alpha=0.3)
        if addDataList is not None and numGood is not None:
            hatches = ["\\\\", "//", "*", "+"]
            cmap = plt.cm.Spectral
            addColors = [cmap(i) for i in np.linspace(0, 1, len(addDataList))]
            if addDataLabelList is None:
                addDataLabelList = ["" for i in len(addDataList)]
            for i, extraData in enumerate(addDataList):
                axes.hist(extraData, bins=numBins, range=(self.qMin, self.qMax), density=density,
                          log=logPlot, color=addColors[i], alpha=0.65, label=addDataLabelList[i],
                          histtype="step", hatch=hatches[i%4])

        axes.tick_params(axis="both", which="both", direction="in", labelsize=8)
        axes.set_xlim(self.qMin, self.qMax)
        axes.set_ylim(numMin, numMax)
        if filterStr is None:
            filterStr = ""
        axes.set_xlabel("{0:s} [{1:s}]".format(self.quantityName, filterStr), fontsize=9)
        axes.set_ylabel("Number", fontsize=9)
        axes.set_yscale("log", nonpositive="clip")
        x0, y0 = 0.03, 0.97
        if self.qMin == 0.0:
            x0, y0 = 0.68, 0.81
        if stats is not None:
            annotateAxes(description, plt, axes, stats, "star", self.magThreshold,
                         signalToNoiseStr=self.signalToNoiseStr, x0=x0, y0=y0,
                         isHist=True, matchRadius=matchRadius,
                         matchRadiusUnitStr=matchRadiusUnitStr, unitScale=self.unitScale,
                         doPrintMedian=doPrintMedian)
        axes.legend(loc="upper right", fontsize=8)
        xOff = 0.0
        if plotInfoDict["cameraName"] is not None:
            xOff = max(0.12, 0.04*len(plotInfoDict["cameraName"]))
            labelCamera(plotInfoDict, plt.gcf(), axes, 0.5 - xOff, 1.04, fontSize=9)
        labelVisit(plotInfoDict, plt.gcf(), axes, 0.5 + xOff, 1.04, fontSize=9)
        if self.config.doLabelRerun:
            plotText("rerun: " + plotInfoDict["rerun"], plt, axes, 0.5, 1.1, fontSize=7, color="purple")
        if zpLabel is not None:
            prefix = "" if "GalExt" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes, 0.14, -0.10, prefix=prefix, fontSize=7, color="green")
        if uberCalLabel:
            plotText(uberCalLabel, plt, axes, 0.14, -0.13, fontSize=7, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.88, -0.10, prefix="cat: ", fontSize=7, color="green")
        if plotInfoDict["camera"] is not None and plotInfoDict["plotType"] == "plotVisit":
            axTopMiddle = plt.axes([0.42, 0.68, 0.2, 0.2])
            axTopMiddle.set_aspect("equal")
            plotCameraOutline(axTopMiddle, plotInfoDict["camera"], plotInfoDict["ccdList"])
        if self.config.doPlotTractOutline and "Coadd" in plotInfoDict["plotType"]:
            axTopMiddle = plt.axes([0.42, 0.68, 0.2, 0.2])
            axTopMiddle.set_aspect("equal")
            plotTractOutline(axTopMiddle, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"])

        yield Struct(fig=fig, description=description, stats=self.stats, statsHigh=self.statsHigh, dpi=120,
                     style="hist")

    def plotSkyPosition(self, description, plotInfoDict, areaDict, cmap=plt.cm.Spectral, stats=None,
                        matchRadius=None, matchRadiusUnitStr=None, zpLabel=None, highlightList=None,
                        forcedStr=None, dataName="star", uberCalLabel=None, doPrintMedian=False,
                        style="sky-all"):
        """Plot quantity as a function of position.
        """
        pad = 0.02  # Number of degrees to pad the axis ranges
        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        raMin, raMax = np.round(ra.min() - pad, 2), np.round(ra.max() + pad, 2)
        decMin, decMax = np.round(dec.min() - pad, 2), np.round(dec.max() + pad, 2)

        deltaRa = raMax - raMin
        deltaDec = decMax - decMin
        deltaDeg = max((deltaRa, deltaDec))
        raMin = (raMin + deltaRa/2.0) - deltaDeg/2.0
        raMax = raMin + deltaDeg
        decMin = (decMin + deltaDec/2.0) - deltaDeg/2.0
        decMax = decMin + deltaDeg

        magThreshold = self.magThreshold
        if dataName == "galaxy" and magThreshold < 99.0:
            magThreshold += 1.0  # plot to fainter mags for galaxies
        if dataName == "star" and "matches" in description and magThreshold < 99.0:
            magThreshold += 1.0  # plot to fainter mags for matching against ref cat
        good = (self.mag < magThreshold if (magThreshold > 0 and magThreshold < 90.0)
                else np.ones(len(self.mag), dtype=bool))
        if ((dataName == "star" or "matches" in description or "Compare" in plotInfoDict["plotType"])
                and ("pStar" not in description and "race" not in description and "resolution"
                     not in description) or ("compareUnforced" in description)):

            vMin, vMax = 0.4*self.qMin, 0.4*self.qMax
            if "mag_" in description or any(ss in description for ss in
                                            ["compareUnforced", "overlap", "matches"]):
                vMin, vMax = 0.6*vMin, 0.6*vMax
            if "matches" in description and "_mag" in description:
                vMax = -vMin
        elif "CModel" in description and "overlap" not in description:
            vMin, vMax = 1.5*self.qMin, 0.5*self.qMax
        elif "raceDiff" in description or "Resids" in description:
            vMin, vMax = 0.5*self.qMin, 0.5*self.qMax
        elif "race" in description:
            yDelta = 0.05*(self.qMax - self.qMin)
            vMin, vMax = self.qMin + yDelta, self.qMax - yDelta
        elif "pStar" in description:
            vMin, vMax = 0.0, 1.0
        elif "resolution" in description and "Compare" not in plotInfoDict["plotType"]:
            vMin, vMax = 0.0, 0.2
        elif "galacticExtinction" in description and "compare" not in description:
            vMin, vMax = min(1.05*self.qMin, 0.98*self.qMax), max(1.02*self.qMin, 0.95*self.qMax)
        else:
            vMin, vMax = self.qMin, self.qMax
        if "Compare" not in plotInfoDict["plotType"]:
            if dataName == "star" and "deconvMom" in description:
                vMin, vMax = -0.1, 0.1
            if dataName == "galaxy" and "deconvMom" in description:
                vMin, vMax = -0.1, 3.0*self.qMax
            if dataName == "galaxy" and "resolution" in description:
                vMin, vMax = 0.0, 1.0
        if (dataName == "galaxy" and "mag_" in description
                and ("matches" not in description or plotInfoDict["plotType"] == "plotColor")):
            vMin = 3.0*self.qMin
            if "GaussianFlux" in description:
                vMin, vMax = 5.0*self.qMin, 0.0
        if (dataName == "galaxy" and ("CircularApertureFlux" in description or "KronFlux" in description)
                and "Compare" not in plotInfoDict["plotType"] and "overlap" not in description
                and "compareUnforced" not in description):
            vMin, vMax = 4.0*self.qMin, 1.0*self.qMax

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.35"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        ptSize = None

        if "Visit" in plotInfoDict["plotType"]:
            ccdList = plotInfoDict["ccdList"]
            if any(ss in description for ss in ["commonZp", "_raw"]):
                plotCcdOutline(axes, areaDict, ccdList,
                               raMin=raMin, raMax=raMax, decMin=decMin, decMax=decMax)
            else:
                plotCcdOutline(axes, areaDict, ccdList, tractInfo=None,
                               raMin=raMin, raMax=raMax, decMin=decMin, decMax=decMax)
            if plotInfoDict["tractInfo"] is not None:
                tractBBox = plotInfoDict["tractInfo"].getBBox()
                tractWcs = plotInfoDict["tractInfo"].getWcs()
                tractRa, tractDec = bboxToXyCoordLists(tractBBox, wcs=tractWcs)
                tractRas = tractRa + (tractRa[0], )
                tractDecs = tractDec + (tractDec[0], )
                axes.plot(tractRas, tractDecs, "w--", linewidth=1, alpha=0.7, label=plotInfoDict["tract"])

        tractLevelPlot = "Coadd" in plotInfoDict["plotType"] or "Color" in plotInfoDict["plotType"]
        if plotInfoDict["tractInfo"] is not None and tractLevelPlot:
            patchBoundary = getRaDecMinMaxPatchList(plotInfoDict["patchIdList"], plotInfoDict["tractInfo"],
                                                    pad=pad, nDecimals=2, raMin=raMin, raMax=raMax,
                                                    decMin=decMin, decMax=decMax)
            raMin = patchBoundary.raMin
            raMax = patchBoundary.raMax
            decMin = patchBoundary.decMin
            decMax = patchBoundary.decMax
            plotPatchOutline(axes, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"])

        stats0 = None
        lightShades = ["white", "lavenderblush", "floralwhite", "paleturquoise", ]
        for name, data in self.data.items():
            if name is not dataName:
                continue
            if not data.mag.any():
                continue
            schema = getSchema(data.catalog)
            if ptSize is None:
                ptSize = 0.7*setPtSize(len(data.mag))
            stats0 = self.calculateStats(data.quantity, good[data.selection])
            selection = data.selection & good
            if highlightList is not None:
                # Make highlight as a background ring of larger size than the
                # data point size.
                i = -1
                sizeFactor = 1.4
                for flag, threshValue, color in highlightList:
                    if flag in schema:
                        # Only a white "halo" really shows up, so ignore color
                        highlightSelection = (self.catalog[flag] > threshValue) & selection
                        if sum(highlightSelection) > 0:
                            i += 1
                            label = flag
                            for k, v in strMappingList:
                                label = label.replace(k, v)
                            axes.scatter(ra[highlightSelection], dec[highlightSelection],
                                         s=sizeFactor*ptSize, marker="o", facecolors="none",
                                         edgecolors=lightShades[i%len(lightShades)],
                                         linewidth=1.0/sizeFactor, label=label)
                            sizeFactor *= 1.4

            axes.scatter(ra[selection], dec[selection], s=ptSize, marker="o", lw=0, label=name,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=vMin, vmax=vMax)

        if stats0 is None:  # No data to plot
            plt.close(fig)
            return
        filterStr = plotInfoDict["filter"]
        filterLabelStr = "[" + filterStr + "]" if ("color" not in plotInfoDict["plotType"]
                                                   and "galacticExtinction" not in description) else ""
        if "lsst" in plotInfoDict["cameraName"]:
            filterLabelStr = "[" + plotInfoDict["cameraName"] + "-" + plotInfoDict["filter"] + "]"
        axes.set_xlabel("RA (deg) {0:s}".format(filterLabelStr))
        axes.set_ylabel("Dec (deg) {0:s}".format(filterLabelStr))

        axes.set_xlim(raMax, raMin)
        axes.set_ylim(decMin, decMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []  # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        colorbarLabel = self.quantityName + " " + filterLabelStr
        fontSize = min(10, max(6, 10 - int(np.log(max(1, len(colorbarLabel) - 50)))))
        cb.ax.tick_params(labelsize=max(6, fontSize - 2))
        cb.ax.yaxis.offsetText.set_fontsize(6)
        cb.set_label(colorbarLabel, fontsize=fontSize, rotation=270, labelpad=15)
        if plotInfoDict["hscRun"] is not None:
            axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")
        if self.config.doLabelRerun:
            rerunKwargs = dict(fontSize=5, color="purple")
            if "rerun2" in plotInfoDict:
                plotText("rerun: " + plotInfoDict["rerun"], plt, axes, 0.5, 1.13, **rerunKwargs)
                plotText("rerun2: " + plotInfoDict["rerun2"], plt, axes, 0.5, 1.105, **rerunKwargs)
            else:
                rerunKwargs.update(fontSize=6)
                plotText("rerun: " + plotInfoDict["rerun"], plt, axes, 0.5, 1.11, **rerunKwargs)
        labelCamera(plotInfoDict, fig, axes, 0.5, 1.07)
        labelVisit(plotInfoDict, fig, axes, 0.5, 1.03)
        if zpLabel is not None:
            prefix = "" if "GalExt" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes, 0.14, -0.07, prefix=prefix, color="green")
        if uberCalLabel:
            plotText(uberCalLabel, plt, axes, 0.14, -0.11, fontSize=7, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes, 0.85, -0.08, prefix="cat: ", color="green")
        strKwargs = dict(loc="upper left", fancybox=True, markerscale=1.2, scatterpoints=3, framealpha=0.35,
                         facecolor="k")
        if highlightList is not None:
            axes.legend(bbox_to_anchor=(-0.05, 1.13), fontsize=6, **strKwargs)
        else:
            axes.legend(bbox_to_anchor=(-0.01, 1.10), fontsize=6, **strKwargs)

        meanStr = "{0.mean:.4f}".format(stats0)
        medianStr = "{0.median:.4f}".format(stats0)
        stdevStr = "{0.stdev:.4f}".format(stats0)
        statsUnitStr = None
        if "(milli)" in self.quantityName:
            statsUnitStr = " (milli)"
        if "(mmag)" in self.quantityName:
            statsUnitStr = " (mmag)"
        if "(mas)" in self.quantityName:
            statsUnitStr = " (mas)"
        if statsUnitStr is not None:
            meanStr = "{0.mean:.2f}".format(stats0)
            medianStr = "{0.median:.2f}".format(stats0)
            stdevStr = "{0.stdev:.2f}".format(stats0)

        x0 = 0.86
        deltaX = 0.004
        lenStr = 0.016*(max(len(meanStr), len(stdevStr)))
        strKwargs = dict(xycoords="axes fraction", va="center", fontsize=7)
        axes.annotate("mean = ", xy=(x0, 1.07), ha="right", **strKwargs)
        axes.annotate(meanStr, xy=(x0 + lenStr, 1.07), ha="right", **strKwargs)
        if doPrintMedian:
            deltaX += (0.155 + lenStr)
            axes.annotate("median = ", xy=(x0 + deltaX, 1.07), ha="right", **strKwargs)
            axes.annotate(medianStr, xy=(x0 + lenStr + deltaX, 1.07), ha="right", **strKwargs)
            deltaX += 0.004
        if statsUnitStr is not None:
            axes.annotate(statsUnitStr, xy=(x0 + lenStr + deltaX, 1.07), ha="left", **strKwargs)
        axes.annotate("stdev = ", xy=(x0, 1.035), ha="right", **strKwargs)
        axes.annotate(stdevStr, xy=(x0 + lenStr, 1.035), ha="right", **strKwargs)
        axes.annotate(r"N = {0} [mag<{1:.1f}]".format(stats0.num, magThreshold),
                      xy=(x0 + lenStr + 0.012, 1.035), ha="left", **strKwargs)

        yield Struct(fig=fig, description=description, stats=self.stats, statsHigh=self.statsHigh, dpi=150,
                     style=style)

    def plotRaDec(self, description, plotInfoDict, stats=None, matchRadius=None, matchRadiusUnitStr=None,
                  zpLabel=None, forcedStr=None, uberCalLabel=None, doPrintMedian=False, style="radec"):
        """Plot quantity as a function of RA, Dec.
        """
        ra = np.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = np.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.magThreshold if self.magThreshold is not None else
                np.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        axes[0].axhline(0, linestyle="--", color="0.6")
        axes[1].axhline(0, linestyle="--", color="0.6")
        ptSize = None
        for name, data in self.data.items():
            if not data.mag.any():
                continue
            if ptSize is None:
                ptSize = setPtSize(len(data.mag))
            selection = data.selection & good
            kwargs = {"s": ptSize, "marker": "o", "lw": 0, "c": data.color, "alpha": 0.5}
            axes[0].scatter(ra[selection], data.quantity[good[data.selection]], label=name, **kwargs)
            axes[1].scatter(dec[selection], data.quantity[good[data.selection]], **kwargs)

        axes[0].set_xlabel("RA (deg)", labelpad=-1)
        axes[1].set_xlabel("Dec (deg)")
        fig.text(0.02, 0.5, self.quantityName, ha="center", va="center", rotation="vertical")

        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        axes[0].legend()
        if stats is not None:
            annotateAxes(description, axes[0], stats, "star", self.magThreshold,
                         signalToNoiseStr=self.signalToNoiseStr, x0=0.03, yOff=0.07,
                         hscRun=plotInfoDict["hscRun"], matchRadius=matchRadius,
                         matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian)
            annotateAxes(description, axes[1], stats, "star", self.magThreshold,
                         signalToNoiseStr=self.signalToNoiseStr, x0=0.03, yOff=0.07,
                         hscRun=["hscRun"], matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                         unitScale=self.unitScale, doPrintMedian=doPrintMedian)
        labelVisit(plotInfoDict, fig, axes[0], 0.5, 1.1)
        if zpLabel is not None:
            prefix = "" if "GalExt" in zpLabel else "zp: "
            plotText(zpLabel, plt, axes[0], 0.13, -0.09, prefix=prefix, fontsize=8, color="green")
        if uberCalLabel:
            plotText(uberCalLabel, plt, axes[0], 0.13, -0.14, fontSize=8, color="green")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes[0], 0.85, -0.09, prefix="cat: ", fontsize=8, color="green")
        yield Struct(fig=fig, description=description, stats=self.stats, statsHigh=self.statsHigh, dpi=120,
                     style=style)

    def plotQuiver(self, catalog, description, plotInfoDict, areaDict, log, cmap=plt.cm.Spectral, stats=None,
                   matchRadius=None, zpLabel=None, forcedStr=None, dataName="star", uberCalLabel=None,
                   scale=1):
        """Plot ellipticity residuals quiver plot.
        """
        schema = getSchema(catalog)
        # Use HSM algorithm results if present, if not, use SDSS Shape
        if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
            compareCol = "ext_shapeHSM_HsmSourceMoments"
            psfCompareCol = "ext_shapeHSM_HsmPsfMoments"
            shapeAlgorithm = "HSM"
            flags = ["ext_shapeHSM_HsmSourceMoments_flag", "ext_shapeHSM_HsmPsfMoments_flag"]
        else:
            compareCol = "base_SdssShape"
            psfCompareCol = "base_SdssShape_psf"
            shapeAlgorithm = "SDSS"
            flags = ["base_SdssShape_flag", "base_SdssShape_flag_psf"]

        # Cull the catalog of flagged sources
        bad = np.zeros(len(catalog), dtype=bool)
        bad |= catalog["deblend_nChild"] > 0
        for flag in flags:
            bad |= catalog[flag]
        # Cull the catalog down to calibration candidates (or stars if
        # calibration flags not available).
        if "calib_psf_used" in schema:
            bad |= ~catalog["calib_psf_used"]
            catStr = "psf_used"
            thresholdType = "calib_psf_used"
            thresholdValue = None
        elif "base_ClassificationExtendedness_value" in schema:
            bad |= catalog["base_ClassificationExtendedness_value"] > 0.5
            bad |= -2.5*np.log10(catalog[self.fluxColumn]) > self.magThreshold
            catStr = "ClassExtendedness"
            thresholdType = "mag"
            thresholdValue = self.magThreshold
        else:
            raise RuntimeError("Neither calib_psf_used nor base_ClassificationExtendedness_value in schema. "
                               "Skip quiver plot.")
        catalog = catalog[~bad].copy(deep=True)

        pad = 0.02  # Number of degrees to pad the axis ranges
        ra = np.rad2deg(catalog["coord_ra"])
        dec = np.rad2deg(catalog["coord_dec"])
        raMin, raMax = np.round(ra.min() - pad, 2), np.round(ra.max() + pad, 2)
        decMin, decMax = np.round(dec.min() - pad, 2), np.round(dec.max() + pad, 2)

        deltaRa = raMax - raMin
        deltaDec = decMax - decMin
        deltaDeg = max((deltaRa, deltaDec))
        raMin = (raMin + deltaRa/2.0) - deltaDeg/2.0
        raMax = raMin + deltaDeg
        decMin = (decMin + deltaDec/2.0) - deltaDeg/2.0
        decMax = decMin + deltaDeg

        fig, axes = plt.subplots(1, 1, subplot_kw=dict(facecolor="0.7"))
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)

        if plotInfoDict["plotType"] == "plotVisit":
            plotCcdOutline(axes, areaDict, plotInfoDict["ccdList"], tractInfo=plotInfoDict["tractInfo"],
                           raMin=raMin, raMax=raMax, decMin=raMin, decMax=decMax)

        if plotInfoDict["tractInfo"] is not None and "Coadd" in plotInfoDict["plotType"]:
            for ip, patch in enumerate(plotInfoDict["tractInfo"]):
                if str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1]) in plotInfoDict["patchIdList"]:
                    wcs = plotInfoDict["tractInfo"].getWcs()
                    raPatch, decPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=wcs)
                    raMin = min(np.round(min(raPatch) - pad, 2), raMin)
                    raMax = max(np.round(max(raPatch) + pad, 2), raMax)
                    decMin = min(np.round(min(decPatch) - pad, 2), decMin)
                    decMax = max(np.round(max(decPatch) + pad, 2), decMax)
            plotPatchOutline(axes, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"])

        e1 = E1Resids(compareCol, psfCompareCol)
        e1 = e1(catalog)
        e2 = E2Resids(compareCol, psfCompareCol)
        e2 = e2(catalog)
        e = np.sqrt(e1**2 + e2**2)

        nz = matplotlib.colors.Normalize()
        nz.autoscale(e)
        cax, _ = matplotlib.colorbar.make_axes(plt.gca())
        cax.tick_params(labelsize=7)
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.jet, norm=nz)
        cb.set_label(
            r"ellipticity residual: $\delta_e$ = $\sqrt{(e1_{src}-e1_{psf})^2 + (e2_{src}-e2_{psf})^2}$",
            rotation=-90, labelpad=16, fontsize=9)

        getQuiver(ra, dec, e1, e2, axes, color=plt.cm.jet(nz(e)), scale=scale, width=0.002, label=catStr)

        filterLabelStr = "[" + plotInfoDict["filter"] + "]"
        if "lsst" in plotInfoDict["cameraName"]:
            filterLabelStr = "[" + plotInfoDict["cameraName"] + "-" + plotInfoDict["filter"] + "]"
        axes.set_xlabel("RA (deg) {0:s}".format(filterLabelStr))
        axes.set_ylabel("Dec (deg) {0:s}".format(filterLabelStr))

        axes.set_xlim(raMax, raMin)
        axes.set_ylim(decMin, decMax)

        good = np.ones(len(e), dtype=bool)
        stats0 = self.calculateStats(e, good, thresholdType=thresholdType, thresholdValue=thresholdValue)
        log.info("plotQuiver: Statistics from %s of %s: %s" % (plotInfoDict["dataId"], self.quantityName,
                                                               stats0))
        meanStr = "{0.mean:.4f}".format(stats0)
        stdevStr = "{0.stdev:.4f}".format(stats0)

        x0 = 0.86
        lenStr = 0.1 + 0.022*(max(max(len(meanStr), len(stdevStr)) - 6, 0))
        axes.annotate("mean = ", xy=(x0, 1.07), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(meanStr, xy=(x0 + lenStr, 1.07), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate("stdev = ", xy=(x0, 1.03), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(stdevStr, xy=(x0 + lenStr, 1.03), xycoords="axes fraction",
                      ha="right", va="center", fontsize=8)
        axes.annotate(r"N = {0}".format(stats0.num), xy=(x0 + lenStr + 0.02, 1.03), xycoords="axes fraction",
                      ha="left", va="center", fontsize=8)

        if plotInfoDict["hscRun"] is not None:
            axes.set_title("HSC stack run: " + plotInfoDict["hscRun"], color="#800080")
        labelCamera(plotInfoDict, fig, axes, 0.5, 1.07, fontSize=8)
        labelVisit(plotInfoDict, fig, axes, 0.5, 1.03, fontSize=8)
        if self.config.doLabelRerun:
            plotText("rerun: " + plotInfoDict["rerun"], plt, axes, 0.5, 1.11, fontSize=7, color="purple")

        if zpLabel is not None:
            plotText(zpLabel, fig, axes, 0.14, -0.08, prefix="zp: ", color="green")
        if uberCalLabel:
            plotText(uberCalLabel, fig, axes, 0.14, -0.12, fontSize=7, color="green")
        plotText(shapeAlgorithm, fig, axes, 0.85, -0.06, prefix="Shape Alg: ", fontSize=7, color="green")
        if forcedStr is not None:
            plotText(forcedStr, fig, axes, 0.85, -0.105, prefix="cat: ", fontSize=7, color="green")
        axes.legend(loc="upper left", bbox_to_anchor=(0.0, 1.08), fancybox=True, shadow=True, fontsize=7)

        yield Struct(fig=fig, description=description, stats=stats, statsHigh=None, dpi=150, style="quiver")

    def plotRhoStatistics(self, description, plotInfoDict, log, treecorrParams, stats,
                          zpLabel=None, forcedStr=None, postFix="", flagsCat=None, uberCalLabel=None,
                          verifyJob=None):
        """Plot Rho Statistics.
        """
        schema = getSchema(self.catalog)
        figAxes = [plt.subplots(), plt.subplots(), plt.subplots()]
        # first plot for Rho 1, 3, 4, second for Rho 2, 5 and third for Rho 0
        figs, axes = list(zip(*figAxes))

        compareCol = "base_SdssShape"
        psfCompareCol = "base_SdssShape_psf"
        shapeAlgorithm = "SDSS"
        flags = list(self.config.flags) + ["base_SdssShape_flag", "base_SdssShape_flag_psf"]

        # Cull the catalog of flagged sources
        bad = np.zeros(len(self.catalog), dtype=bool)
        bad |= self.catalog["deblend_nChild"] > 0
        for flag in flags:
            bad |= self.catalog[flag]
        good_catalog = self.catalog[~bad].copy(deep=True)

        rhoStatsFunc = RhoStatistics(compareCol, psfCompareCol, **treecorrParams)
        rhoStats = rhoStatsFunc(good_catalog)
        plotRhoStats(axes, rhoStats)
        log.debug("Tract id in Rho Stats: {0}".format(plotInfoDict["tract"]))

        xOff = 0.18 if uberCalLabel else 0.10
        yOff = -0.09 if uberCalLabel else -0.10
        for figId, figax in enumerate(figAxes):
            fig, ax = figax
            figDescription = description + str(figId + 1)
            labelCamera(plotInfoDict, fig, ax, 0.5, 1.09)
            labelVisit(plotInfoDict, fig, ax, 0.5, 1.04)
            if self.config.doLabelRerun:
                plotText("rerun: " + plotInfoDict["rerun"], fig, ax, 1.03, 0.5, fontSize=7, color="purple",
                         rotation=-90)
            if zpLabel is not None:
                plotText(zpLabel, fig, ax, xOff, yOff, prefix="zp: ", color="green")
            if uberCalLabel:
                plotText(uberCalLabel, fig, ax, xOff, yOff - 0.04, prefix="uberCal: ", fontSize=7,
                         color="green")
            plotText(shapeAlgorithm, fig, ax, 0.85, yOff, prefix="Shape Alg: ", fontSize=8, color="green")
            if forcedStr is not None:
                xxOff = 0.85 if uberCalLabel else 0.29
                yyOff = -0.13 if uberCalLabel else yOff
                plotText(forcedStr, fig, ax, xxOff, yyOff, prefix="cat: ", fontSize=8, color="green")

            yield Struct(fig=fig, description=figDescription, stats=stats, statsHigh=None, dpi=120,
                         style="RhoStats")

        if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
            figAxes = [plt.subplots(), plt.subplots(), plt.subplots()]
            # first plot for Rho 1, 3, 4, second for Rho 2, 5
            # and third for Rho 0.
            figs, axes = list(zip(*figAxes))

            description = description.replace("Rho", "hsmRho")
            compareCol = "ext_shapeHSM_HsmSourceMoments"
            psfCompareCol = "ext_shapeHSM_HsmPsfMoments"
            shapeAlgorithm = "HSM"
            flags = list(self.config.flags) + ["ext_shapeHSM_HsmSourceMoments_flag",
                                               "ext_shapeHSM_HsmPsfMoments_flag"]

            # Cull the catalog of flagged sources
            bad = np.zeros(len(self.catalog), dtype=bool)
            bad |= self.catalog["deblend_nChild"] > 0
            for flag in flags:
                bad |= self.catalog[flag]
            good_catalog = self.catalog[~bad].copy(deep=True)

            rhoStatsFunc = RhoStatistics(compareCol, psfCompareCol, **treecorrParams)
            rhoStats = rhoStatsFunc(good_catalog)
            plotRhoStats(axes, rhoStats)

            if verifyJob is not None:
                metaDict = {"shapeAlgorithm": "HSM"}
                verifyJob = updateVerifyJob(verifyJob, metaDict=metaDict)
                measExtrasDictList = [{"name": kk, "value": vv, "label": None,
                                       "description": "keyword arguments relevant to treecorr"}
                                      for kk, vv in treecorrParams.items()]
                measExtrasDictList.append(None)
                catalogName = "calibPsfUsed" if self.calibUsedOnly else "allStars"

                for rhoId in range(6):
                    xi = rhoStats[rhoId].xip if rhoId > 0 else rhoStats[rhoId].xi
                    measExtrasDictList[-1] = {"name": f"rhoStatistics{rhoId}Function", "value": xi,
                                              "label": f"Rho{rhoId}",
                                              "description": f"Measured Rho Statistics {rhoId} using "
                                              f"{shapeAlgorithm} method"}

                    verifyMetricName = (f"pipe_analysis.rhoStatistics{rhoId}_{shapeAlgorithm}_smallScale"
                                        f"_{catalogName}")
                    verifyJob = addMetricMeasurement(verifyJob, verifyMetricName,
                                                     measureRhoMetrics(rhoStats[rhoId], 1, "<="),
                                                     measExtrasDictList=measExtrasDictList)

                    verifyMetricName = (f"pipe_analysis.rhoStatistics{rhoId}_{shapeAlgorithm}_largeScale"
                                        f"_{catalogName}")
                    verifyJob = addMetricMeasurement(verifyJob, verifyMetricName,
                                                     measureRhoMetrics(rhoStats[rhoId], 1, ">"),
                                                     measExtrasDictList=measExtrasDictList)

            for figId, figax in enumerate(figAxes):
                fig, ax = figax
                figDescription = description + str(figId + 1)
                labelCamera(plotInfoDict, fig, ax, 0.5, 1.09)
                labelVisit(plotInfoDict, fig, ax, 0.5, 1.04)
                if self.config.doLabelRerun:
                    plotText("rerun: " + plotInfoDict["rerun"], fig, ax, 1.03, 0.5, fontSize=7,
                             color="purple", rotation=-90)
                if zpLabel is not None:
                    plotText(zpLabel, fig, ax, xOff, yOff, prefix="zp: ", fontSize=7, color="green")
                if uberCalLabel:
                    plotText(uberCalLabel, fig, ax, xOff, yOff - 0.04, prefix="uberCal: ", fontSize=7,
                             color="green")
                plotText(shapeAlgorithm, fig, ax, 0.85, yOff, prefix="Shape Alg: ", fontSize=7,
                         color="green")
                if forcedStr is not None:
                    xxOff = 0.85 if uberCalLabel else 0.29
                    yyOff = -0.13 if uberCalLabel else yOff
                    plotText(forcedStr, fig, ax, xxOff, yyOff, prefix="cat: ", fontSize=7, color="green")

                yield Struct(fig=fig, description=figDescription, stats=stats, statsHigh=None, dpi=120,
                             style="RhoStats")

    def plotInputCounts(self, catalog, description, plotInfoDict, log, forcedStr=None, uberCalLabel=None,
                        cmap=plt.cm.viridis, alpha=0.5, doPlotPatchOutline=True, sizeFactor=5.0,
                        maxDiamPix=1000, columnName="base_InputCount_value", fluxScale=1e12):
        """Plot visit input counts of tract.

        Can optionally plot with a background that is a grayscale of the image.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            The source catalog whose objects will be plotted as ellipses
            (scaled by a factor of ``sizeFactor`` but truncated to a maximum
            diameter of ``maxDiamPix``) and color-mapped by their
            base_InputCount_value.
        description : `str`
            The type of plot being made, used by the butler to save the figure.
        plotInfoDict : `dict`
            A dictionary of useful plot information.
        log : `lsst.log.Log`
            Logger object for logging messages.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            The camera associated with the dataset (used to label the plot with
            the camera's name).
        forcedStr : `str`, optional
            String to label the catalog type (forced vs. unforced) on the plot.
        cmap : `matplotlib.colors.ListedColormap`, optional
            The matplotlib colormap to use.  It will be given transparency
            level set by ``alpha``.
        alpha : `float`, optional
            The matplotlib blending value, between 0 (transparent) and 1
            (opaque).
        doPlotPatchOutline : `bool`, optional
            A boolean indicating whether to overplot the patch outlines and
            index labels.
        sizeFactor : `float`, optional
            Factor by which to multiply the source ellipse sizes for plotting
            (the nominal size is quite small).
        maxDiamPix : `int`, optional
            A maximum diameter to be plotted for any source's ellipse (such
            that a single ellipse cannot overwhelm the plot and noting that
            this will not be indicative of the true input counts for the outer
            pixels as that number strictly applies to the objects centroid
            pixel).  If a given object gets truncated to this size, an opaque
            blue outline will be plotted around its ellipse.
        """
        tractBbox = plotInfoDict["tractInfo"].getBBox()
        tractWcs = plotInfoDict["tractInfo"].getWcs()

        fig, axes = plt.subplots(1, 1)
        axes.tick_params(which="both", direction="in", top=True, right=True, labelsize=7)

        centStr = "slot_Centroid"
        shapeStr = "slot_Shape"

        diamAs = []  # matplotlib.patches.Ellipse wants diameters for width and height
        diamBs = []
        thetas = []
        edgeColors = []  # to outline any ellipses truncated at maxDiamPix
        if "CircularApertureFlux" in columnName:
            m = re.search(r"CircularApertureFlux_(\d+)_(\d+)_", columnName)
            apertureRadius = float(m.groups()[0]) + float(m.groups()[1])/10.0
            diamAs = [min(sizeFactor*2.0*apertureRadius, maxDiamPix)]*len(catalog)
            diamBs = diamAs
            thetas = [0.0]*len(catalog)
            edgeColors = ["None"]*len(catalog)
        else:
            for _, src in catalog.iterrows():
                edgeColor = "None"
                srcQuad = afwGeom.Quadrupole(src[shapeStr + "_xx"], src[shapeStr + "_yy"],
                                             src[shapeStr + "_xy"])
                srcEllip = afwGeom.ellipses.Axes(srcQuad)
                diamA = srcEllip.getA()*2.0*sizeFactor
                diamB = srcEllip.getB()*2.0*sizeFactor
                # Truncate ellipse size to a maximum width or height of
                # maxDiamPix.
                if diamA > maxDiamPix or diamB > maxDiamPix:
                    edgeColor = "blue"
                    if diamA >= diamB:
                        diamB = diamB*(maxDiamPix/diamA)
                        diamA = maxDiamPix
                    else:
                        diamA = diamA*(maxDiamPix/diamB)
                        diamB = maxDiamPix
                diamAs.append(diamA)
                diamBs.append(diamB)
                thetas.append(np.degrees(srcEllip.getTheta()))
                edgeColors.append(edgeColor)

        xyOffsets = np.stack((catalog[centStr + "_x"], catalog[centStr + "_y"]), axis=-1)
        inputCounts = catalog[columnName]
        if "instFlux" in columnName:
            inputCounts *= fluxScale
            finiteFlux = np.isfinite(inputCounts)
            clippedStats = calcQuartileClippedStats(inputCounts[finiteFlux], nSigmaToClip=5.0)
            boundMin = clippedStats.mean - 7.0*clippedStats.stdDev
            boundMax = clippedStats.mean + 7.0*clippedStats.stdDev
            boundStep = np.around((boundMax - boundMin)/20, 3)
            cbarExtend = "both"
        else:
            boundMin = 1
            boundMax = int(np.nanmax(inputCounts)) + 1
            boundStep = 1
            cbarExtend = "min"
        bounds = np.arange(boundMin, boundMax, boundStep)
        alphaCmap = makeAlphaCmap(cmap=cmap, alpha=alpha)
        norm = matplotlib.colors.BoundaryNorm(bounds, alphaCmap.N)
        if cbarExtend == "min":
            alphaCmap.set_under("r")
        elif cbarExtend == "max":
            alphaCmap.set_over("r")
        elif cbarExtend == "both":
            alphaCmap.set_under("r")
            alphaCmap.set_over("r")
        else:
            log.warn("plotInputCounts: Unknown extend string for matplotlib colorbar: {:}.  Setting to "
                     "\"neither\"".format(cbarExtend))
            cbarExtend = "neither"

        ellipsePatchList = [matplotlib.patches.Ellipse(xy=xy, width=diamA, height=diamB, angle=theta)
                            for xy, diamA, diamB, theta in zip(xyOffsets, diamAs, diamBs, thetas)]
        ec = matplotlib.collections.PatchCollection(ellipsePatchList, cmap=alphaCmap, norm=norm,
                                                    edgecolors=edgeColors, linewidth=0.1)

        ec.set_array(inputCounts)
        axes.add_collection(ec)
        cbar = plt.colorbar(ec, extend=cbarExtend, fraction=0.04, pad=0.03)
        columnStr = fluxToPlotString(columnName)
        fluxScaleStr = "{:.0e}".format(fluxScale)
        columnStr = columnStr + "*" + fluxScaleStr if "instFlux" in columnName else columnStr
        cbar.set_label(columnStr + ": ellipse size * {:} [maxDiam = {:}] (pixels)".
                       format(sizeFactor, maxDiamPix), fontsize=7, rotation=-90, labelpad=10)
        cbar.ax.tick_params(direction="in", labelsize=6)

        axes.set_xlim(tractBbox.getMinX(), tractBbox.getMaxX())
        axes.set_ylim(tractBbox.getMinY(), tractBbox.getMaxY())

        filterStr = plotInfoDict["filter"]
        filterLabelStr = "[" + filterStr + "]"
        if "lsst" in plotInfoDict["cameraName"]:
            filterLabelStr = "[" + plotInfoDict["cameraName"] + "-" + plotInfoDict["filter"] + "]"
        axes.set_xlabel("xTract (pixels) {0:s}".format(filterLabelStr), size=9)
        axes.set_ylabel("yTract (pixels) {0:s}".format(filterLabelStr), size=9)

        # Get RA and Dec tract limits to add to plot axis labels
        tract00 = tractWcs.pixelToSky(tractBbox.getMinX(),
                                      tractBbox.getMinY()).getPosition(units=geom.degrees)
        tract0N = tractWcs.pixelToSky(tractBbox.getMinX(),
                                      tractBbox.getMaxY()).getPosition(units=geom.degrees)
        tractN0 = tractWcs.pixelToSky(tractBbox.getMaxX(),
                                      tractBbox.getMinY()).getPosition(units=geom.degrees)

        textKwargs = dict(ha="left", va="center", transform=axes.transAxes, fontsize=7, color="blue")
        plt.text(-0.05, -0.07, str("{:.2f}".format(tract00.getX())), **textKwargs)
        plt.text(-0.15, 0.00, str("{:.2f}".format(tract00.getY())), **textKwargs)
        plt.text(0.96, -0.07, str("{:.2f}".format(tractN0.getX())), **textKwargs)
        plt.text(-0.15, 0.97, str("{:.2f}".format(tract0N.getY())), **textKwargs)
        textKwargs["fontsize"] = 8
        plt.text(0.45, -0.11, "RA (deg)", **textKwargs)
        plt.text(-0.15, 0.5, "Dec (deg)", rotation=90, **textKwargs)

        if doPlotPatchOutline:
            plotPatchOutline(axes, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"], plotUnits="pixel",
                             idFontSize=5)
        xOff = 0.0
        if plotInfoDict["cameraName"] is not None:
            xOff = max(0.09, 0.03*len(plotInfoDict["cameraName"]))
            labelCamera(plotInfoDict, fig, axes, 0.5 - xOff, 1.04)
        labelVisit(plotInfoDict, fig, axes, 0.5 + xOff, 1.04)
        if self.config.doLabelRerun:
            plotText("rerun: " + plotInfoDict["rerun"], plt, axes, 0.5, 1.09, fontSize=7, color="purple")
        if forcedStr is not None:
            plotText(forcedStr, fig, axes, 0.80, -0.11, prefix="cat: ", fontSize=7, color="green")
        if uberCalLabel:
            plotText(uberCalLabel, fig, axes, 0.16, -0.11, fontSize=7, color="green")

        yield Struct(fig=fig, description=description, stats=None, statsHigh=None, dpi=1200, style="tract")

    def plotSkyObjects(self, skyObjCat, description, plotInfoDict, log, zpLabel=None, forcedStr=None,
                       verifyJob=None):
        """Plot histograms of sky object/source measurements.

        Also plots the focal plane or tract outline to indicate which sub-units
        (i.e. ccd or patch) contributed data (i.e. for which a source catalog
        existed) and are color-coded by the measured metric per sub-unit, where
        here this metric is the mean flux measured in the the sky sources
        in a circular aperture of 9 pixels.

        Parameters
        ----------
        skyObjCat : `lsst.afw.table.SourceCatalog`
            The sky objects source catalog.
        description : `str`
            The description of the plot, used to save the figure.
        plotInfoDict : `dict`
            A dictionary of useful plot information.
        log : `lsst.log.Log`
            Logger object for logging messages.
        zpLabel : `str`, optional
            A label indicating the external calibration applied (currently
            either "jointcal", "fgcm", "fgcm_tract", or "meas_mosaic", but the
            latter is effectively retired).
        forcedStr : `str`, optional
            String to label the catalog type (forced vs. unforced) on the plot.
        verifyJob : `lsst.verify.Job` or None, optional
            The verify Job to add any metric measurements to.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
        fig.subplots_adjust(wspace=0.17, bottom=0.18, left=0.08, right=0.76, top=0.9)
        topRight = [0.80, 0.68, 0.2, 0.2]
        axes[0].tick_params(axis="x", which="both", direction="in", labelsize=7)
        axes[1].tick_params(axis="x", which="both", direction="in", labelsize=7)
        axes[0].tick_params(axis="y", which="both", direction="in", labelsize=6)
        axes[1].tick_params(axis="y", which="both", direction="in", labelsize=6)
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        if "lsst" in plotInfoDict["cameraName"]:
            filterStr = plotInfoDict["cameraName"] + "-" + plotInfoDict["filter"]
        else:
            filterStr = plotInfoDict["filter"]
        fluxStrList = ["base_PsfFlux_instFlux", "base_CircularApertureFlux_9_0_instFlux",
                       "base_CircularApertureFlux_12_0_instFlux", "base_CircularApertureFlux_25_0_instFlux"]
        fluxScale = 1e12
        countMax = 0
        countChiMax = 0
        xOff = 0.96
        yOff = 1.065
        legendLoc = "upper left"
        alpha = 0.35
        nBins = None
        nBinsChi = None
        for i, fluxStr in enumerate(fluxStrList):
            yOff -= 0.1
            flux = skyObjCat[fluxStr]*fluxScale
            fluxErr = skyObjCat[fluxStr + "Err"]*fluxScale
            finiteFlux = np.isfinite(flux)
            skyFluxArr = flux[finiteFlux]
            skyFluxErrArr = fluxErr[finiteFlux]
            nSigmaToClip = 5.0
            clippedStats = calcQuartileClippedStats(skyFluxArr, nSigmaToClip=nSigmaToClip)
            # add sigma-clipped mean to verify metrics tracker
            if verifyJob is not None:
                if fluxStr == "base_CircularApertureFlux_9_0_instFlux":
                    NANOJANSKYS_PER_AB_FLUX = (0*u.ABmag).to_value(u.nJy)
                    skyObjectMean = (clippedStats.mean/fluxScale)*NANOJANSKYS_PER_AB_FLUX*u.nanojansky
                    skyObjectStd = (clippedStats.stdDev/fluxScale)*NANOJANSKYS_PER_AB_FLUX*u.nanojansky
                    measExtrasDictList = [{"name": "nTotal", "value": len(flux),
                                           "label": "nTotal", "description": "Total number of sky objects."},
                                          {"name": "nUsed", "value": len(skyFluxArr),
                                           "label": "nUsed", "description": "Number of sky objects used."},
                                          {"name": "nSigmaToClip", "value": nSigmaToClip,
                                           "label": "nSigma",
                                           "description": "Number of sigma's used for clipped statistics."}]
                    verifyJob = addMetricMeasurement(verifyJob, "pipe_analysis.skyObjectMean_CircApRad9pix",
                                                     skyObjectMean, measExtrasDictList=measExtrasDictList)
                    verifyJob = addMetricMeasurement(verifyJob, "pipe_analysis.skyObjectStd_CircApRad9pix",
                                                     skyObjectStd, measExtrasDictList=measExtrasDictList)
            meanStr = "mean = {0:5.2f}".format(clippedStats.mean)
            stdStr = "  std = {0:5.2f}".format(clippedStats.stdDev)
            numStr = "    N = {}".format(sum(clippedStats.goodArray))
            nBins = "auto" if i == 0 else nBins
            count, binsFlux, ignored = axes[0].hist(skyFluxArr[clippedStats.goodArray], bins=nBins,
                                                    label=fluxToPlotString(fluxStr),
                                                    range=(-10.0*clippedStats.stdDev,
                                                           10.0*clippedStats.stdDev),
                                                    density=True, color=colors[i], edgecolor=colors[i],
                                                    alpha=alpha)
            nBins = binsFlux if i == 0 else nBins
            countMax = count.max() if count.max() and count.max() > countMax else countMax
            if i == 0:
                xLim = 5.0*clippedStats.stdDev
                # Put labels on left if typically over subtracted (so the mean
                # lines don't cover the text).
                if clippedStats.mean > 0:
                    xOff -= 0.62
                    legendLoc = "upper right"
            axes[0].plot(binsFlux,
                         (1/((clippedStats.stdDev*np.sqrt(2*np.pi))*(
                             np.exp((binsFlux - clippedStats.mean)**2/(2*clippedStats.stdDev**2))))),
                         color=colors[i])
            axes[0].axvline(x=clippedStats.mean, color=colors[i], linestyle=":")
            kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=6, color=colors[i])
            axes[0].annotate(meanStr, xy=(xOff, yOff), **kwargs)
            axes[0].annotate(stdStr, xy=(xOff, yOff - 0.035), **kwargs)
            axes[0].annotate(numStr, xy=(xOff, yOff - 0.07), **kwargs)

            chiArr = skyFluxArr/skyFluxErrArr
            meanChi = chiArr[clippedStats.goodArray].mean()
            stdDevChi = chiArr[clippedStats.goodArray].std()
            meanChiStr = "mean = {0:5.2f}".format(meanChi)
            stdChiStr = "  std = {0:5.2f}".format(stdDevChi)
            numChiStr = "    N = {}".format(len(chiArr[clippedStats.goodArray]))

            nBinsChi = "auto" if i == 0 else nBinsChi
            countChi, binsChi, ignoredChi = axes[1].hist(chiArr[clippedStats.goodArray], bins=nBinsChi,
                                                         label=fluxToPlotString(fluxStr),
                                                         range=(-10.0*stdDevChi, 10.0*stdDevChi),
                                                         density=True, color=colors[i], edgecolor=colors[i],
                                                         alpha=alpha)
            nBinsChi = binsChi if i == 0 else nBinsChi
            countChiMax = countChi.max() if countChi.max() and countChi.max() > countChiMax else countChiMax
            if i == 0:
                xLimChi = 5.0*stdDevChi
            axes[1].plot(binsChi,
                         1/(stdDevChi*np.sqrt(2*np.pi))*np.exp(-(binsChi - meanChi)**2/(2*stdDevChi**2)),
                         color=colors[i])
            axes[1].axvline(x=meanChi, color=colors[i], linestyle=":")
            axes[1].annotate(meanChiStr, xy=(xOff, yOff), **kwargs)
            axes[1].annotate(stdChiStr, xy=(xOff, yOff - 0.035), **kwargs)
            axes[1].annotate(numChiStr, xy=(xOff, yOff - 0.07), **kwargs)

        yLim = max(0.0015, np.round(1.25*countMax, 3))
        yChiLim = np.round(1.25*countChiMax, 2)

        axes[0].set_xlabel("%s %.0e [%s]" % ("flux *", fluxScale, filterStr), fontsize=8)
        axes[0].set_ylabel("Normalized Counts", fontsize=8)
        axes[0].set_xlim(-xLim, xLim)
        axes[0].set_ylim(bottom=0.0, top=yLim)
        axes[0].axvline(x=0.0, color="black", linestyle="--")

        axes[1].set_xlabel("chi = %s [%s]" % ("flux/fluxErr", filterStr), fontsize=8)
        axes[1].set_xlim(-xLimChi, xLimChi)
        axes[1].set_ylim(bottom=0.0, top=yChiLim)
        axes[1].axvline(x=0.0, color="black", linestyle="--")
        axes[1].legend(loc=legendLoc, fontsize=6)

        if plotInfoDict["cameraName"] is not None:
            labelCamera(plotInfoDict, fig, axes[0], 0.5, 1.04)
        labelVisit(plotInfoDict, fig, axes[1], 0.5, 1.04)
        if self.config.doLabelRerun:
            plotText("rerun: " + plotInfoDict["rerun"], plt, axes[0], 1.02, 1.09, fontSize=7, color="purple")
        if forcedStr is not None:
            plotText(forcedStr, plt, axes[0], 0.99, -0.15, prefix="cat: ", fontSize=8, color="green")

        # Set color-coding for camera/tract outline to be based on mean metric
        # per unit (ccd/patch).
        metricFluxStr = "base_CircularApertureFlux_9_0_instFlux"
        metricStr = "mean(" + fluxToPlotString(metricFluxStr) + ")"
        metricPerUnitDict = {}
        unitList = None
        if plotInfoDict["ccdList"] is not None:
            unitList = plotInfoDict["ccdList"]
            unitStr = "ccdId"
        elif plotInfoDict["patchList"] is not None:
            unitList = plotInfoDict["patchList"]
            unitStr = "patchId"
        for iUnit in unitList:
            good = skyObjCat[unitStr] == iUnit
            perUnitSkyObjCat = skyObjCat[good].copy(deep=True)
            if len(perUnitSkyObjCat) > 0:
                perUnitSkyFlux = perUnitSkyObjCat[metricFluxStr]*fluxScale
                finiteFlux = np.isfinite(perUnitSkyFlux)
                perUnitSkyFluxArr = perUnitSkyFlux[finiteFlux]
                perUnitClippedStats = calcQuartileClippedStats(perUnitSkyFluxArr, nSigmaToClip=5.0)
                metricPerUnitDict[str(iUnit)] = perUnitClippedStats.mean
            else:
                metricPerUnitDict[str(iUnit)] = np.nan
                log.info("No good sky objects for %s %s" % (unitStr, iUnit))

        if plotInfoDict["camera"] is not None and plotInfoDict["ccdList"] is not None:
            axTopRight = plt.axes(topRight)
            axTopRight.set_aspect("equal")
            plotCameraOutline(axTopRight, plotInfoDict["camera"], plotInfoDict["ccdList"],
                              metricPerCcdDict=metricPerUnitDict, metricStr=metricStr, fig=fig)

        tractLevelPlot = "Coadd" in plotInfoDict["plotType"] or "Color" in plotInfoDict["plotType"]
        if self.config.doPlotTractOutline and tractLevelPlot:
            if "patchIdList" in plotInfoDict and plotInfoDict["patchIdList"] != plotInfoDict["patchList"]:
                # Hack for gen3 patch ids
                for patchId, patch in zip(plotInfoDict["patchIdList"], plotInfoDict["patchList"]):
                    metricPerUnitDict[patchId] = metricPerUnitDict.pop(str(patch))
            if plotInfoDict["tractInfo"] is not None and len(plotInfoDict["patchIdList"]) > 0:
                axTopRight = plt.axes(topRight)
                axTopRight.set_aspect("equal")
                plotTractOutline(axTopRight, plotInfoDict["tractInfo"], plotInfoDict["patchIdList"],
                                 metricPerPatchDict=metricPerUnitDict, metricStr=metricStr, fig=fig)

        yield Struct(fig=fig, description=description, stats=None, statsHigh=None, dpi=120, style="hist")

    def plotAll(self, description, plotInfoDict, areaDict, log, enforcer=None, matchRadius=None,
                matchRadiusUnitStr=None, zpLabel=None, forcedStr=None, postFix="", plotRunStats=True,
                highlightList=None, extraLabels=None, uberCalLabel=None, doPrintMedian=False):
        """Make all plots.
        """
        stats = self.stats
        # Make sure you have some good data to plot
        if all(stats[stat].num == 0 for stat in stats):
            log.warn("plotAll: No good data points to plot for: {:}.  Skipping plots...".
                     format(self.shortName))
            return
        # Dict of all parameters common to plot* functions
        plotKwargs = dict(stats=stats, matchRadius=matchRadius, matchRadiusUnitStr=matchRadiusUnitStr,
                          zpLabel=zpLabel, forcedStr=forcedStr, uberCalLabel=uberCalLabel,
                          doPrintMedian=doPrintMedian)
        if "galacticExtinction" not in self.shortName:
            yield from self.plotAgainstMagAndHist(log, description, plotInfoDict, plotRunStats=plotRunStats,
                                                  highlightList=highlightList, extraLabels=extraLabels,
                                                  **plotKwargs)

        if self.config.doPlotOldMagsHist and "galacticExtinction" not in self.shortName:
            yield from self.plotAgainstMag(self.shortName, **plotKwargs)
            yield from self.plotHistogram(self.shortName, plotInfoDict, **plotKwargs)

        skyPositionKwargs = dict(highlightList=highlightList)
        skyPositionKwargs.update(plotKwargs)
        if "all" in self.data:
            styleStr = "sky-all"
            dataName = "all"
            if self.checkGoodDataExists(dataName, stats, log, styleStr):
                yield from self.plotSkyPosition(self.shortName, plotInfoDict, areaDict,
                                                style=styleStr + postFix, dataName=dataName,
                                                **skyPositionKwargs)
        if "star" in self.data:
            styleStr = "sky-stars"
            dataName = "star"
            if self.checkGoodDataExists(dataName, stats, log, styleStr):
                yield from self.plotSkyPosition(self.shortName, plotInfoDict, areaDict,
                                                style=styleStr + postFix, dataName=dataName,
                                                **skyPositionKwargs)

        if "galaxy" in self.data and (not any(ss in self.shortName for ss in
                                              ["pStar", "race", "Xx", "Yy", "Resids", "gri", "riz", "izy",
                                               "z9y", "color_"])):
            styleStr = "sky-gals"
            dataName = "galaxy"
            if self.checkGoodDataExists(dataName, stats, log, styleStr):
                yield from self.plotSkyPosition(self.shortName, plotInfoDict, areaDict,
                                                style=styleStr + postFix, dataName=dataName,
                                                **skyPositionKwargs)

        if ("unknown" in self.data
            and (not any(ss in self.shortName for ss in ["pStar", "race", "Xx", "Yy", "Resids", "gri",
                                                         "riz", "izy", "z9y", "color_"]))):
            styleStr = "sky-unkn"
            dataName = "unknown"
            if self.checkGoodDataExists(dataName, stats, log, styleStr):
                yield from self.plotSkyPosition(self.shortName, plotInfoDict, areaDict,
                                                style=styleStr + postFix, dataName=dataName,
                                                **skyPositionKwargs)
        if "diff_" in self.shortName:
            styleStr = "sky-split"
            dataName = "split"
            if self.checkGoodDataExists(dataName, stats, log, styleStr):
                yield from self.plotSkyPosition(self.shortName, plotInfoDict, areaDict,
                                                style=styleStr + postFix, dataName=dataName,
                                                **skyPositionKwargs)

        if self.config.doPlotRaDec:
            yield from self.plotRaDec(self.shortName, style="radec" + postFix, **plotKwargs)

        log.info("Statistics from %s of %s: %s" % (plotInfoDict["dataId"], self.quantityName, stats))
        if enforcer:
            enforcer(stats, plotInfoDict["dataId"], log, self.quantityName)
        # TODO: DM-24795, Should this be returned? Is it vanishing into the
        # ether?
        return stats

    def statistics(self, magThreshold=None, signalToNoiseThreshold=None, forcedMean=None):
        """Calculate statistics on quantity.

        Parameters
        ----------
        magThreshold : `float` or `None`
            Subsample for computing stats only includes objects brighter than
            ``magThreshold``.
        signalToNoiseThreshold : `float` or `None`
            Subsample for computing stats only includes objects with S/N
            greater than ``signalToNoiseThreshold``.

        Raises
        ------
        RuntimeError
            If both ``magThreshold`` and ``signalToNoiseThreshold`` are -- or
            are not -- `None`.
        """
        thresholdList = [magThreshold, signalToNoiseThreshold]
        if (all(threshold is not None for threshold in thresholdList)
                or all(threshold is None for threshold in thresholdList)):
            raise RuntimeError("Must specify one AND ONLY one of magThreshold and signalToNoiseThreshold. "
                               "They are currently set to {0:} and {1:}, respectively".
                               format(magThreshold, signalToNoiseThreshold))
        thresholdType = "S/N" if signalToNoiseThreshold else "mag"
        thresholdValue = signalToNoiseThreshold if signalToNoiseThreshold else magThreshold
        stats = {}
        for name, data in self.data.items():
            good = np.ones(len(data.mag), dtype=bool)
            if signalToNoiseThreshold:
                good &= data.signalToNoise >= signalToNoiseThreshold
            if magThreshold:
                good &= data.mag <= magThreshold
            stats[name] = self.calculateStats(data.quantity, good, forcedMean=forcedMean,
                                              thresholdType=thresholdType, thresholdValue=thresholdValue)
            if self.quantityError is not None:
                stats[name].sysErr = self.calculateSysError(data.quantity, data.error,
                                                            good, forcedMean=forcedMean)
            if not stats:
                stats = None
        return stats

    def calculateStats(self, quantity, selection, forcedMean=None, thresholdType="", thresholdValue=None):
        """Calculate basic statistics for a (sub-selection of a) quanatity.

        Parameters
        ----------
        quantity : `numpy.ndarray` of `float`
            Array containing the values of the quantity on which the statistics
            are to be computed.
        selection : `numpy.ndarray` of `bool`
            Boolean array indicating the sub-selection of data points in
            ``quantity`` to be considered for the statistics computation.
        forcedMean : `float`, `int`, or `None`, optional
            If provided, the value at which to force the mean (i.e. the other
            stats will be calculated based on an assumed mean of this value).
        thresholdType : `str`, optional
            String representing the type of threshold to be used in culling to
            the subset of ``quantity`` to be used in the statistics
            computation: "S/N" and "mag" indicate a threshold based on
            signal-to-noise or magnitude, respectively.  A flag name, e.g.
            "calib_psf_used", indicates that the sample was culled based on the
            value of this flag.  Provided here simply for inclusion in the
            returned ``Stats`` object.
        thresholdValue : `float`, `int`, or `None`, optional
            The threshold value used in culling ``quantity`` to the subset to
            be included in the statistics computation.  Provided here simply
            for inclusion in the returned ``Stats`` object.

        Returns
        -------
        Stats : `lsst.pipe.analysis.utils.Stats`
            Instance of the `lsst.pipe.analysis.utils.Stats` class (a
            sub-class of `lsst.pipe.base.Struct`) containing the results of
            the statistics calculation.  Attributes are:

            ``dataUsed``
                Boolean array indicating the subset of ``quantity`` that was
                used in the statistics computation (`numpy.ndarray` of `bool`).
            ``num``
                Number of data points used in calculation after culling based
                on ``selection`` and sigma clipping during the computation
                (`int`).
            ``total``
                Number of data points considered for use in calculation after
                cut based on ``selection`` (`int`).
            ``mean``
                Mean of the data points used in the calculation (`float`).
            ``stddev``
                Standard deviation of the data points used in the calculation
                (`float`).
            ``forcedMean``
                Value provided in ``forcedMean`` indicating (if not `None`) the
                value the mean was forced to be for computation of the other
                statistics.  A value of `None` indicates the mean was computed
                from the data themselves (`float` or `None`).
            ``median``
                Median of the data points used in the calculation (`float`).
            ``clip``
                Value used for clipping outliers from the data points used in
                statistics calculation (`float`).
                - i.e. clip x if abs(x - ``mean``) > ``clip``
                - this parameter is controlled by the config parameter
                  ``analysis.config.clip`` which is in units of number of
                  standard deviations, defined here as
                  0.74*interQuartileDistance (`float`).
            ``thresholdType``
                String provided in input variable ``thresholdType``
                representing the type of threshold used for culling the data
                (`str`).
            ``thresholdValue``
                Value provided in input variable ``thresholdValue``
                representing the value used for the threshold culling of the
                data (`float`).
        """
        total = selection.sum()  # Total number we're considering
        if total == 0:
            return Stats(dataUsed=0, num=0, total=0, mean=np.nan, stdev=np.nan, forcedMean=np.nan,
                         median=np.nan, clip=np.nan, thresholdType=thresholdType,
                         thresholdValue=thresholdValue)
        clippedStats = calcQuartileClippedStats(quantity[selection], nSigmaToClip=self.config.clip)
        mean = clippedStats.mean if forcedMean is None else forcedMean
        good = selection & np.logical_not(np.abs(quantity - clippedStats.median) > clippedStats.clipValue)
        return Stats(dataUsed=good, num=good.sum(), total=total, mean=mean, stdev=clippedStats.stdDev,
                     forcedMean=forcedMean, median=clippedStats.median, clip=clippedStats.clipValue,
                     thresholdType=thresholdType, thresholdValue=thresholdValue)

    def calculateSysError(self, quantity, error, selection, forcedMean=None, tol=1.0e-3):
        """Calculate the systematic error of a (sub-selection of a) quantity.

        Parameters
        ----------
        quantity : `numpy.ndarray` of `float`
            Array containing the values of the quantity on which the statistics
            are to be computed.
        error : `numpy.ndarray` of `float`
            Array containing the errors on the data in ``quantity``.
        selection : `numpy.ndarray` of `bool`
            Boolean array indicating the sub-selection of data points in
            ``quantity`` to be considered for the statistics computation.
        forcedMean : `float`, `int`, or `None`, optional
            If provided, the value at which to forced the mean (i.e. the other
            stats will be calculated based on an assumed mean of this value).
            Otherwise, a value of `None` indicates the mean is to be computed
            from the data themselves.
        tol : `float`, optional
           Stopping tolerance for the `scipy.optimize.root` routine.

        Returns
        -------
        answer : `str`
           String providing the mean and spread of the systematic error.
        """
        import scipy.optimize

        def function(sysErr2):
            sigNoise = quantity/np.sqrt(error**2 + sysErr2)
            stats = self.calculateStats(sigNoise, selection, forcedMean=forcedMean)
            return stats.stdev - 1.0

        if True:
            result = scipy.optimize.root(function, 0.0, tol=tol)
            if not result.success:
                print("Warning: sysErr calculation failed: {:s}".format(result.message))
                answer = np.nan
            else:
                answer = np.sqrt(result.x[0])
        else:
            answer = np.sqrt(scipy.optimize.newton(function, 0.0, tol=tol))
        print("calculateSysError: {0:.4f}, {1:.4f}, {2:.4f}".format(function(answer**2),
                                                                    function((answer+0.001)**2),
                                                                    function((answer-0.001)**2)))
        return answer

    def checkGoodDataExists(self, dataName, stats, log, styleStr):
        """Check if good data points exist in stats object for given data type.

        Parameters
        ----------
        dataName : `str`
            The name and key of the dataset for consideration in the ``stats``
            object.
        stats : `dict` of `lsst.pipe.analysis.utils.Stats`
            A dictionary containing the statistics info per ``dataName``.
        log : `lsst.log.Log`
            Logger object for logging messages.

        Returns
        -------
        answer : `bool`
            Returns `True` if data points were included in ``stats`` for
            ``dataName``, else `False`.
        """
        if stats[dataName].num == 0:
            log.warn("No good data points to plot for: {:} {:}.  Skipping {:} plot.".
                     format(self.shortName, dataName, styleStr))
            answer = False
        else:
            answer = True
        return answer
