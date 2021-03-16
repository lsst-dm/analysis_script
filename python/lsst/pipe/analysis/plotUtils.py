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

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import numpy as np

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
from lsst.pipe.base import Struct

from .utils import calcQuartileClippedStats

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsExposure
except ImportError:
    applyMosaicResultsExposure = None

__all__ = ["AllLabeller", "StarGalaxyLabeller", "OverlapsStarGalaxyLabeller", "MatchesStarGalaxyLabeller",
           "CosmosLabeller", "plotText", "annotateAxes", "labelVisit", "labelCamera",
           "filterStrFromFilename", "plotCameraOutline", "plotTractOutline", "plotPatchOutline",
           "plotCcdOutline", "bboxToXyCoordLists", "getRaDecMinMaxPatchList", "percent", "setPtSize",
           "getQuiver", "makeAlphaCmap", "buildTractImage", "determineExternalCalLabel", "getPlotInfo"]


class AllLabeller(object):
    labels = {"all": 0}
    plot = ["all"]

    def __call__(self, catalog):
        return np.zeros(len(catalog))


class StarGalaxyLabeller(object):
    # The "unknown" category is to accommodate failed star/gal classification
    labels = {"star": 0, "galaxy": 1, "unknown": 9}
    plot = ["star", "galaxy", "unknown"]
    _column = "base_ClassificationExtendedness_value"

    def __call__(self, catalog):
        starGal = catalog[self._column].copy(deep=True)
        starGal[np.isnan(starGal)] = 9
        starGal[(starGal > 0.5) & (starGal < 1.5)] = 1
        starGal[starGal <= 0.5] = 0
        return starGal  # np.where(catalog[self._column] < 0.5, 0, 1)


class OverlapsStarGalaxyLabeller(StarGalaxyLabeller):
    # The "unknown" category is to accommodate failed star/gal classification
    labels = {"star": 0, "galaxy": 1, "split": 2, "unknown": 9}
    plot = ["star", "galaxy", "split", "unknown"]

    def __init__(self, first="first_", second="second_"):
        self._first = first
        self._second = second

    def __call__(self, catalog1, catalog2=None):
        catalog2 = catalog2 if catalog2 is not None else catalog1
        starGal1 = catalog1[self._first + self._column].copy(deep=True)
        starGal1[np.isnan(starGal1)] = 9
        starGal1[(starGal1 > 0.5) & (starGal1 < 1.5)] = 1
        starGal1[starGal1 <= 0.5] = 0
        starGal2 = catalog2[self._second + self._column].copy(deep=True)
        starGal2[np.isnan(starGal2)] = 9
        starGal2[(starGal2 > 0.5) & (starGal2 < 1.5)] = 1
        starGal2[starGal2 <= 0.5] = 0
        return np.where(starGal1 == starGal2, starGal1, 2)


class MatchesStarGalaxyLabeller(StarGalaxyLabeller):
    _column = "src_base_ClassificationExtendedness_value"


class CosmosLabeller(StarGalaxyLabeller):
    """Do star/galaxy classification using Alexie Leauthaud's Cosmos catalog.
    """
    def __init__(self, filename, radius):
        original = afwTable.BaseCatalog.readFits(filename)
        good = (original["CLEAN"] == 1) & (original["MU.CLASS"] == 2)
        num = good.sum()
        cosmos = afwTable.SimpleCatalog(afwTable.SimpleTable.makeMinimalSchema())
        cosmos.reserve(num)
        for ii in range(num):
            cosmos.addNew()
        cosmos["id"][:] = original["NUMBER"][good]
        cosmos["coord_ra"][:] = original["ALPHA.J2000"][good]*(1.0*geom.degrees).asRadians()
        cosmos["coord_dec"][:] = original["DELTA.J2000"][good]*(1.0*geom.degrees).asRadians()
        self.cosmos = cosmos
        self.radius = radius

    def __call__(self, catalog):
        # A kdTree would be better, but this is all we have right now
        matches = afwTable.matchRaDec(self.cosmos, catalog, self.radius)
        good = set(mm.second.getId() for mm in matches)
        return np.array([0 if ii in good else 1 for ii in catalog["id"]])


def plotText(textStr, fig, axis, xLoc, yLoc, prefix="", fontSize=None, color="k", coordSys="axes", **kwargs):
    """Label the plot with the string provided at a given location.

    Parameters
    ----------
    textStr : `str`
        String of text to plot.
    fig : `matplotlib.figure.Figure`
        The figure to plot ``textStr`` on.
    axis : `matplotlib.axes._axes.Axes`
        Particular matplotlib axes of ``plt`` on which to plot ``testStr``.
    xLoc, yLoc : `float`
        x and y coordinates, in corrdinate system set by ``coordSys``, at which
        to plot the ``textStr``.  The string will be centered both horizontally
        and vertically at this position.
    prefix : `str`, optional
        Optional prefix to add to ``textStr``.
    fontSize : `int` or `str` or `None`, optional
        Size of font for plotting of ``textStr``.  May be either an absolute
        font size in points, or a size string, relative to the default font
        size, or `None`.  If `None`, in which case an automatic scaling based
        on the length of ``textStr`` will be used.
    color : `str`, optional
        Color to plot ``textStr``.  Can be any matplotlib color `str`.
    coordSys : `str`, optional
        Coordinate system for ``xLoc``, ``yLoc``.  Choices and matplotlib
        mappings are:
            "axes" => axis.transAxes (`matplotlib.transforms.BboxTransformTo`)
            "data" => axis.transData (`matplotlib.transforms.BboxTransformTo`)
            "figure" => axis.transFigure
                        (`matplotlib.transforms.BboxTransformTo`)
    **kwargs
        Arbitrary keyword arguments.  These can include any of those accecpted
        by matplotlib's matplotlib.pyplot.text function (i.e. are properties of
        the matplotlib.text.Text class).  Of particular interest here include:

        - ``rotation`` : Angle in degrees to rotate ``textStr`` for plotting
                         or one of strings "vertical" or "horizontal".  The
                         matplotlib default is 0 degrees (`int` or `str`).
        - ``alpha`` : The matplotlib blending value, between 0 (transparent)
                      and 1 (opaque).  The matplotlib default is 1 (`float`).

    Raises
    ------
    ValueError
        If unrecognized ``coordSys`` is requested (i.e. something other than
        axes, data, or figure).
    """
    if coordSys == "axes":
        transform = axis.transAxes
    elif coordSys == "data":
        transform = axis.transData
    elif coordSys == "figure":
        transform = axis.transFigure
    else:
        raise ValueError("Unrecognized coordSys: {}.  Must be one of axes, data, figure".format(coordSys))
    if not fontSize:
        fontSize = int(9 - min(3, len(textStr)/10))
    fig.text(xLoc, yLoc, prefix + textStr, ha="center", va="center", fontsize=fontSize, transform=transform,
             color=color, **kwargs)


def annotateAxes(description, axes, statsConf, dataSet, magThresholdConf, signalToNoiseStrConf=None,
                 statsHigh=None, magThresholdHigh=None, signalToNoiseHighStr=None,
                 x0=0.03, y0=0.96, yOff=0.05, fontSize=8, ha="left", va="top", color="blue",
                 isHist=False, hscRun=None, matchRadius=None, matchRadiusUnitStr="\"",
                 unitScale=1.0, doPrintMedian=False):
    """Label the plot with the statistical computation results.

    Parameters
    ----------
    description : `str`
        String representing the type of plot. Used here to select for/against
        certain annotations for certain styles of plots.
    axes : `matplotlib.axes._axes.Axes`
        Particular matplotlib axes of ``plt`` on which to plot the annotations.
    statsConf : `lsst.pipe.analysis.utils.Stats`
        `lsst.pipe.analysis.utils.Stats` object that contains the results from
        the "configured" statistical computation results for the threshold type
        and values set in ``analysis.config.suseSignalToNoiseThreshold`` and:
        - ``analysis.config.signalToNoiseThreshold`` if the former is `True` or
        - ``analysis.config.magThreshold`` if it is `False`.
    dataset : `str`
        Name of the catalog dataset to for which annotations are being added.
        Valid strings are "star", "galaxy", "all", and "split".
    magThresholdConf : `float`
        The "configured" value for the magnitude threshold (i.e. the value set
        in ``analysis.config.magThreshold`` if the threshold was set based on
        magnitude or the "effective" magnitude threshold if the cut was based
        on S/N).
    signalToNoiseStrConf : `str` or `None`, optional
        A string representing the type of threshold used in culling the data to
        the subset of the quantity that was used in the statistics computation
        of ``statsConf``: "S/N" and "mag" indicate a threshold based on
        signal-to-noise or magnitude, respectively.
    statsHigh : `lsst.pipe.analysis.utils.Stats`, optional
        `lsst.pipe.analysis.utils.Stats` object that contains the results from
        the "high" statistical computation results whose value is set in
        ``analysis.config.signalToNoiseHighThreshold``.
    magThresholdHigh : `float`, optional
        The "effective" magnitude threshold based on the "high" S/N cut.
    signalToNoiseHighStr : `str`
        A string representing the threshold used in culling of the dataset to
        the subset of the quantity that was used in the statistics computation
        of ``statsHigh``.
    x0, y0 : `float`, optional
        Axis coordinates controlling placement of annotations on the plot.
    yOff : `float`, optional
        Offset by which to separate annotations along the y-axis.
    fontSize : `int`, optional
        Font size for plot labels.
    ha, va : `str`, optional
        Horizontal and vertical allignments for text labels.  Can be any valid
        matplotlib allignment string.
    color : `str`, optional
        Color for annotations.  Can be any matplotlib color `str`.
    isHist : `bool`, optional
        Boolean indicating if this is a histogram style plot (for slightly
        different annotation settings).
    hscRun : `str` or `None`, optional
        String representing "HSCPIPE_VERSION" fits header if the data were
        processed with the (now obsolete, but old reruns still exist)
        "HSC stack".
    matchRadius : `float` or `None`, optional
        Maximum search radius for source matching between catalogs.
    matchRadiusUnitStr : `str`, optional
        String representing the units of the match radius (e.g. "arcsec",
        "pixel").
    unitScale : `float`, optional
        Number indicating any scaling of the units (e.g 1000.0 means units
        are in "milli" of the base unit).
    doPrintMedian : `bool`, optional
        Boolean to indicate if the median (in addition to the mean) should
        be printed on the plot.

    Returns
    -------
    l1, l2 : `matplotlib.lines.Line2D`
        Output of the axes.axvline commands for the median and clipped
        values (used for plot legends).
    """
    xThresh = axes.get_xlim()[0] + 0.58*(axes.get_xlim()[1] - axes.get_xlim()[0])
    plotInfo = [[statsConf, magThresholdConf, signalToNoiseStrConf, y0],
                [statsHigh, magThresholdHigh, signalToNoiseHighStr, 0.18]]
    for stats, magThreshold, signalToNoiseStr, y00 in plotInfo:
        axes.annotate(dataSet + r" N = {0.num:d} (of {0.total:d})".format(stats[dataSet]),
                      xy=(x0, y00), xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color=color)
        if signalToNoiseStr:
            axes.annotate(signalToNoiseStr, xy=(xThresh, y00), xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
            axes.annotate(r" [mag$\lesssim${0:.1f}]".format(magThreshold), xy=(xThresh, y00 - yOff),
                          xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
        else:
            axes.annotate(r" [mag$\leqslant${0:.1f}]".format(magThreshold), xy=(xThresh, y00),
                          xycoords=("data", "axes fraction"),
                          ha="right", va=va, fontsize=fontSize, color="k", alpha=0.8)
        meanStr = "{0.mean:.4f}".format(stats[dataSet])
        medianStr = "{0.median:.4f}".format(stats[dataSet])
        stdevStr = "{0.stdev:.4f}".format(stats[dataSet])
        statsUnitStr = None
        if unitScale == 1000.0:
            meanStr = "{0.mean:.2f}".format(stats[dataSet])
            medianStr = "{0.median:.2f}".format(stats[dataSet])
            stdevStr = "{0.stdev:.2f}".format(stats[dataSet])
            statsUnitStr = " (milli)"
            if any(ss in description for ss in ["_ra", "_dec", "distance"]):
                statsUnitStr = " (mas)"
            if any(ss in description for ss in ["Flux", "_photometry", "_mag"]):
                statsUnitStr = " (mmag)"
        lenStr = 0.12 + 0.017*(max(len(meanStr), len(stdevStr)))
        strKwargs = dict(xycoords="axes fraction", va=va, fontsize=fontSize, color="k")
        yOffMult = 1
        axes.annotate("mean = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
        axes.annotate(meanStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)
        if statsUnitStr is not None:
            axes.annotate(statsUnitStr, xy=(x0 + lenStr + 0.006, y00 - yOffMult*yOff), ha="left", **strKwargs)
        yOffMult += 1
        axes.annotate("stdev = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
        axes.annotate(stdevStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)
        if doPrintMedian:
            yOffMult += 1
            axes.annotate("med = ", xy=(x0 + 0.12, y00 - yOffMult*yOff), ha="right", **strKwargs)
            axes.annotate(medianStr, xy=(x0 + lenStr, y00 - yOffMult*yOff), ha="right", **strKwargs)

    if matchRadius is not None:
        yOffMult += 1
        axes.annotate("Match radius = {0:.2f}{1:s}".format(matchRadius, matchRadiusUnitStr),
                      xy=(x0, y0 - yOffMult*yOff), ha=ha, **strKwargs)
    if hscRun is not None:
        yOffMult += 1
        axes.annotate("HSC stack run: {0:s}".format(hscRun), xy=(x0, y0 - yOffMult*yOff),
                      xycoords="axes fraction", ha=ha, va=va, fontsize=fontSize, color="#800080")
    if isHist:
        l1 = axes.axvline(stats[dataSet].median, linestyle="dotted", color="0.7")
        l2 = axes.axvline(stats[dataSet].median + stats[dataSet].clip, linestyle="dashdot", color="0.7")
        axes.axvline(stats[dataSet].median - stats[dataSet].clip, linestyle="dashdot", color="0.7")
    else:
        l1 = axes.axhline(stats[dataSet].median, linestyle="dotted", color="0.7", label="median")
        l2 = axes.axhline(stats[dataSet].median + stats[dataSet].clip, linestyle="dashdot", color="0.7",
                          label="clip")
        axes.axhline(stats[dataSet].median - stats[dataSet].clip, linestyle="dashdot", color="0.7")
    return l1, l2


def labelVisit(plotInfoDict, fig, axis, xLoc, yLoc, color="k", fontSize=8):
    """Add Visit information to the plot.

    Parameters
    ----------
    plotInfoDict : `dict`
        A dict containing useful information to add to the plot.
    fig : `matplotlib.figure.Figure`
        The plot to add the information to.
    axis : `matplotlib.axes._subplots.AxesSubplot`
        The specfic axis to add the information to.
    xLoc : `float`
        The x location to put the label at.
    yLoc : `float`
        The y location to put the label at.
    color : `str`
        The color to make the label.
    fontSize : ``
        The fontsize to use for the label.
    """
    if plotInfoDict["visit"] is None:
        labelStr = "Tract: {}".format(plotInfoDict["tract"])
    else:
        labelStr = "Tract: {} Visit: {}".format(plotInfoDict["tract"], plotInfoDict["visit"])
    fig.text(xLoc, yLoc, labelStr, ha="center", va="center", fontsize=fontSize, transform=axis.transAxes,
             color=color)


def labelCamera(plotInfoDict, fig, axis, xLoc, yLoc, color="k", fontSize=9):
    labelStr = "camera: " + plotInfoDict["cameraName"]
    fig.text(xLoc, yLoc, labelStr, ha="center", va="center", fontsize=fontSize, transform=axis.transAxes,
             color=color)


def filterStrFromFilename(filename):
    """!Determine filter string from filename
    """
    filterStr = None
    f1 = filename.find("plots/") + len("plots/")
    f2 = filename.find("/", f1)
    filterStr = filename[f1:f2]

    return filterStr


def plotCameraOutline(axes, camera, ccdList, color="k", fontSize=6, metricPerCcdDict=None,
                      metricStr="", fig=None, metricSigmaRange=4.0):
    """Plot the outline of the camera ccds highlighting those with data.

    Parameters
    ----------
    axes : `matplotlib.axes._axes.Axes`
        Particular matplotlib axes on which to plot the tract outline.
    camera : `lsst.afw.cameraGeom.Camera`, optional
        The ``camera`` associated with the dataset (used to label the plot with
        the camera's name).
    ccdList : `list` of `int`, optional
        List of ccd IDs with data to be plotted.
    fontSize : `int`, optional
        Font size for plot labels.
    metricPerCcdDict : `dict` of `float`, optional
        Dictionary of per patch metric averages; {ccdId: metricValue}.  If
        provided, these values will be used to color-code the camera outline
        plot.
    metricStr : `str`, optional
        String representing the computed metric values provided in
        ``metricPerCcdDict``.
    fig : `matplotlib.figure.Figure`, optional
        The figure on which to add the per-ccd metric info (required to add
        the colorbar).
    metricSigmaRange : `float`, optional
        Number of sigma to make the +/- range for the metric colorbar.
    """
    if metricPerCcdDict:
        if fig is None:
            raise RuntimeError("Must supply the matplotlib fig if color-coding by metric-per-ccd")
    axes.tick_params(which="both", direction="in", labelleft=False, labelbottom=False)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)
    camRadius = max(camera.getFpBBox().getWidth(), camera.getFpBBox().getHeight())/2
    camRadius = np.round(camRadius, -1)
    camLimits = np.round(1.25*camRadius, -1)
    intCcdList = [int(ccd) for ccd in ccdList]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors.pop(colors.index("#7f7f7f"))  # get rid of the gray one as is doesn't contrast well with white
    colors.append("gold")
    hasRotatedCcds = False
    for ccd in camera:
        if ccd.getOrientation().getNQuarter() != 0:
            hasRotatedCcds = True
            break
    if metricPerCcdDict:  # color-code the ccds by the per-ccd metric measurement
        cmap = plt.cm.viridis
        metricPerCcdArray = np.fromiter(metricPerCcdDict.values(), dtype="float32")
        finiteMetrics = np.isfinite(metricPerCcdArray)
        clippedStats = calcQuartileClippedStats(metricPerCcdArray[finiteMetrics], nSigmaToClip=5.0)
        vMin = clippedStats.mean - metricSigmaRange*clippedStats.stdDev
        vMax = clippedStats.mean + metricSigmaRange*clippedStats.stdDev
        vMax = max(abs(vMin), vMax) if vMax > 0 else vMax  # Make range symmetric about 0 if it crosses 0
        vMin = -vMax if vMax > 0 else vMin
        cmapBins = np.linspace(vMin, vMax, cmap.N - 1)
    for ic, ccd in enumerate(camera):
        ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
        if ccd.getType() == cameraGeom.DetectorType.SCIENCE:
            axes.add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                           facecolor="none", edgecolor="k", ls="solid", lw=0.5, alpha=0.5))
        if ccd.getId() in intCcdList:
            if metricPerCcdDict is None:
                if hasRotatedCcds:
                    nQuarter = ccd.getOrientation().getNQuarter()
                    fillColor = colors[nQuarter%len(colors)]
                elif ccd.getName()[0] == "R":
                    try:
                        fillColor = colors[(int(ccd.getName()[1]) + int(ccd.getName()[2]))%len(colors)]
                    except Exception:
                        fillColor = colors[ic%len(colors)]
                else:
                    fillColor = colors[ic%len(colors)]
            else:
                cmapBinIndex = np.digitize(metricPerCcdDict[str(ccd.getId())], cmapBins)
                fillColor = cmap.colors[cmapBinIndex]
            ccdCorners = ccd.getCorners(cameraGeom.FOCAL_PLANE)
            axes.add_patch(patches.Rectangle(ccdCorners[0], *list(ccdCorners[2] - ccdCorners[0]),
                                             fill=True, facecolor=fillColor, edgecolor="k",
                                             ls="solid", lw=1.0, alpha=0.7))
    axes.set_xlim(-camLimits, camLimits)
    axes.set_ylim(-camLimits, camLimits)
    if camera.getName() == "HSC":
        for x, y, t in ([-1, 0, "N"], [0, 1, "W"], [1, 0, "S"], [0, -1, "E"]):
            axes.text(1.085*camRadius*x, 1.085*camRadius*y, t, ha="center", va="center",
                      fontsize=fontSize - 1)
            axes.text(-0.82*camRadius, 1.04*camRadius, "%s" % camera.getName(), ha="center",
                      fontsize=fontSize, color=color)
    else:
        axes.text(0.0, 1.04*camRadius, "%s" % camera.getName(), ha="center", fontsize=fontSize,
                  color=color)
    if metricPerCcdDict:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []  # fake up the array of the scalar mappable. Urgh...
        axesBbox = axes.get_position()
        caxDim = [axesBbox.xmin, 0.95*axesBbox.ymin, axesBbox.width, 0.07*axesBbox.height]
        cax = fig.add_axes(caxDim)
        cax.tick_params(labelsize=fontSize - 1)
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label(label=metricStr, size=fontSize)


def plotTractOutline(axes, tractInfo, patchList, fontSize=5, maxDegBeyondPatch=1.5, metricPerPatchDict=None,
                     metricStr="", fig=None, metricSigmaRange=7.0):
    """Plot the outline of the tract and patches highlighting those with data.

    As some skyMap settings can define tracts with a large number of patches,
    this can become very crowded.  So, if only a subset of patches are
    included, find the outer boudary of all patches in ``patchList`` and only
    plot to ``maxDegBeyondPatch`` degrees beyond those boundaries (in all four
    directions).

    Parameters
    ----------
    axes : `matplotlib.axes._axes.Axes`
        Particular matplotlib axes on which to plot the tract outline.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
        Tract information object for extracting tract RA and Dec limits.
    patchList : `list` of `str`
        List of patch IDs with data to be plotted.  These will be color shaded
        in the outline plot.
    fontSize : `int`, optional
        Font size for plot labels.
    maxDegBeyondPatch : `float`, optional
        Maximum number of degrees to plot beyond the border defined by all
        patches with data to be plotted.
    metricPerPatchDict : `dict` of `float`, optional
        Dictionary of per patch metric averages; {patchId: metricValue}.  If
        provided, these values will be used to color-code the tract outline
        plot.
    metricStr : `str`, optional
        String representing the computed metric values provided in
        ``metricPerPatchDict``.
    fig : `matplotlib.figure.Figure`, optional
        The figure on which to add the per-patch metric info (required to add
        the colorbar).
    metricSigmaRange : `float`, optional
        Number of sigma to make the +/- range for the metric colorbar.
    """
    if metricPerPatchDict:
        if fig is None:
            raise RuntimeError("Must supply the matplotlib fig if color-coding by metric-per-patch")
    buff = 0.02
    axes.tick_params(which="both", direction="in", labelsize=fontSize)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)

    tractRa, tractDec = bboxToXyCoordLists(tractInfo.getBBox(), wcs=tractInfo.getWcs())
    patchBoundary = getRaDecMinMaxPatchList(patchList, tractInfo, pad=maxDegBeyondPatch)

    xMin = min(max(tractRa), patchBoundary.raMax) + buff
    xMax = max(min(tractRa), patchBoundary.raMin) - buff
    yMin = max(min(tractDec), patchBoundary.decMin) - buff
    yMax = min(max(tractDec), patchBoundary.decMax) + buff
    xlim = xMin, xMax
    ylim = yMin, yMax
    axes.fill(tractRa, tractDec, fill=False, color="k", edgecolor="k", lw=0.5, linestyle="solid", alpha=0.3)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors.pop(colors.index("#7f7f7f"))  # get rid of the gray one as that's our no-data colour
    colors.append("gold")
    if metricPerPatchDict:  # color-code the ccds by the per-patch metric measurement
        cmap = plt.cm.viridis
        metricPerPatchArray = np.fromiter(metricPerPatchDict.values(), dtype="float32")
        finiteMetrics = np.isfinite(metricPerPatchArray)
        clippedStats = calcQuartileClippedStats(metricPerPatchArray[finiteMetrics], nSigmaToClip=5.0)
        vMin = clippedStats.mean - metricSigmaRange*clippedStats.stdDev
        vMax = clippedStats.mean + metricSigmaRange*clippedStats.stdDev
        vMax = max(abs(vMin), vMax) if vMax > 0 else vMax  # Make range symmetric about 0 if it crosses 0
        vMin = -vMax if vMax > 0 else vMin
        cmapBins = np.linspace(vMin, vMax, cmap.N - 1)

    for ip, patch in enumerate(tractInfo):
        patchIndexStr = str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1])
        color = "k"
        alpha = 0.05
        if patchIndexStr in patchList:
            if metricPerPatchDict is None:
                (color, alpha) = (colors[ip%len(colors)], 0.5)
            else:
                if np.isfinite(metricPerPatchDict[patchIndexStr]):
                    alpha = 1.0
                    cmapBinIndex = np.digitize(metricPerPatchDict[patchIndexStr], cmapBins)
                    color = cmap.colors[cmapBinIndex]
                else:
                    color, alpha = "red", 0.9
        ra, dec = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
        deltaRa = abs(max(ra) - min(ra))
        deltaDec = abs(max(dec) - min(dec))
        pBuff = 0.5*max(deltaRa, deltaDec)
        centerRa = min(ra) + 0.5*deltaRa
        centerDec = min(dec) + 0.5*deltaDec
        if (centerRa < xMin + pBuff and centerRa > xMax - pBuff
                and centerDec > yMin - pBuff and centerDec < yMax + pBuff):
            axes.fill(ra, dec, fill=True, color=color, lw=0.5, linestyle="solid", alpha=alpha)
            if patchIndexStr in patchList or (centerRa < xMin - 0.2*pBuff
                                              and centerRa > xMax + 0.2*pBuff
                                              and centerDec > yMin + 0.2*pBuff
                                              and centerDec < yMax - 0.2*pBuff):
                axes.text(percent(ra), percent(dec, 0.5), str(patchIndexStr),
                          fontsize=fontSize - 1, horizontalalignment="center", verticalalignment="center")
    axes.text(percent((xMin, xMax), 1.065), percent((yMin, yMax), -0.08), "RA",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center", color="green")
    axes.text(percent((xMin, xMax), 1.14), percent((yMin, yMax), -0.02), "Dec",
              fontsize=fontSize, horizontalalignment="center", verticalalignment="center",
              rotation="vertical", color="green")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if metricPerPatchDict:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vMin, vmax=vMax))
        mappable._A = []  # fake up the array of the scalar mappable. Urgh...
        axesBbox = axes.get_position()
        caxDim = [axesBbox.xmin, 0.93*axesBbox.ymin, axesBbox.width, 0.07*axesBbox.height]
        cax = fig.add_axes(caxDim)
        cax.tick_params(labelsize=fontSize)
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.set_label(label=metricStr, size=fontSize + 1)


def plotCcdOutline(axes, areaDict, ccdList, tractInfo=None, fontSize=8, lineStyle="-", color="k",
                   labelStr=None, doPlotCcdId=True, raMin=-1e12, raMax=1e12, decMin=-1e12, decMax=1e12):
    """Plot the outlines of the ccds in ccdList on a given axis.

    Parameters
    ----------
    axes : `matplotlib.axes._subplots.AxesSubplot`
        The axes to draw the ccds on.
    areaDict : `dict`
        A dictionary containing information about the ccds and their corners
        using the wcs of the visit.
    ccdList : `list`
        A list of the ccds used in this analysis.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`, optional
        The information about the tract.
    fontSize : `int`, optional
        The fontsize to use for the ccd labels.
    lineStyle : `str`, optional
        The matplotlib-style line style string.
    color : `str`, optional
        The matplotlib-style color in which to plot the outlines.
    labelStr : `str` or `None`, optional
        An optional string to use for the legend entry of the CCD outlines.
    doPlotCcdId : `bool`, optional
        Whetther to plot the CCD Id label in the middle of each outline.
    """
    for ccd in ccdList:
        # Use the precomputed corners to make lists of RA and Dec to plot
        ccdCorners = areaDict["corners_{}".format(ccd)]
        ra0 = ccdCorners[0].getRa().asDegrees()
        ra1 = ccdCorners[2].getRa().asDegrees()
        dec0 = ccdCorners[0].getDec().asDegrees()
        dec1 = ccdCorners[2].getDec().asDegrees()

        ras = [cx.asDegrees() for ic, (cx, cy) in enumerate(ccdCorners)]
        decs = [cy.asDegrees() for ic, (cx, cy) in enumerate(ccdCorners)]
        ras.append(ra0)
        decs.append(dec0)

        cenX = ra0 + (ra1 - ra0)/2
        cenY = dec0 + (dec1 - dec0)/2

        # Only plot the ccds with any corner in the tract given in tractInfo,
        # plot all if no tractInfo.
        inTract = False
        if tractInfo is not None:
            for coord in ccdCorners:
                if tractInfo.contains(coord):
                    inTract = True
                    break

        if not tractInfo or inTract:
            if cenX > raMin and cenX < raMax and cenY > decMin and cenY < decMax:
                axes.plot(ras, decs, linestyle=lineStyle, color=color, linewidth=1, label=labelStr)
                if doPlotCcdId:
                    axes.text(cenX, cenY, "{}".format(ccd), ha="center", va="center", fontsize=fontSize,
                              color=color)


def plotPatchOutline(axes, tractInfo, patchList, plotUnits="deg", idFontSize=None):
    """Plot outlines of patches in patchList.
    """
    validWcsUnits = ["deg", "rad"]
    idFontSize = max(5, 9 - int(0.4*len(patchList))) if not idFontSize else idFontSize
    for ip, patch in enumerate(tractInfo):
        if str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1]) in patchList:
            if len(patchList) < 9:
                if plotUnits in validWcsUnits:
                    xCoord, yCoord = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs(),
                                                        wcsUnits=plotUnits)
                else:
                    xCoord, yCoord = bboxToXyCoordLists(patch.getOuterBBox(), wcs=None)
                xCoords = xCoord + (xCoord[0], )
                yCoords = yCoord + (yCoord[0], )
                axes.plot(xCoords, yCoords, color="black", lw=0.5, linestyle="solid")
            if plotUnits in validWcsUnits:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), tractInfo.getWcs(),
                                                    wcsUnits=plotUnits)
            else:
                xCoord, yCoord = bboxToXyCoordLists(patch.getInnerBBox(), wcs=None)
            xCoords = xCoord + (xCoord[0], )
            yCoords = yCoord + (yCoord[0], )
            axes.plot(xCoords, yCoords, color="black", lw=0.8, linestyle="dashed")
            axes.text(percent(xCoords), percent(yCoords, 0.5), str(patch.getIndex()),
                      fontsize=idFontSize, horizontalalignment="center", verticalalignment="center")


def bboxToXyCoordLists(bbox, wcs=None, wcsUnits="deg"):
    """Get the corners of a BBox and convert them to x and y coord lists.

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        The bounding box under consideration.
    wcs : `lsst.afw.geom.SkyWcs`, optional
        If provided, the coordinate lists returned will be RA and Dec in
        ``wcsUnits``.  Ignored if ``wcs`` is `None`.
    wcsUnits : `str`, optional
        Coordinate units to be returned if a ``wcs`` is provided (ignored
        otherwise).  Can be either "deg" or "rad".

    Raises
    ------
    RuntimeError
        If ``wcsUnits`` is neither "deg" nor "rad".

    Returns
    -------
    xCoords, yCoords : `list` of `float`
        The lists associated with the x and y coordinates in appropriate uints.
    """
    validWcsUnits = ["deg", "rad"]
    corners = []
    for corner in bbox.getCorners():
        p = geom.Point2D(corner.getX(), corner.getY())
        if wcs:
            if wcsUnits not in validWcsUnits:
                raise RuntimeError("wcsUnits must be one of {:}".format(validWcsUnits))
            coord = wcs.pixelToSky(p)
            if wcsUnits == "deg":
                corners.append([coord.getRa().asDegrees(), coord.getDec().asDegrees()])
            elif wcsUnits == "rad":
                corners.append([coord.getRa().asRadians(), coord.getDec().asRadians()])
        else:
            coord = p
            corners.append([coord.getX(), coord.getY()])
    xCoords, yCorrds = zip(*corners)
    return xCoords, yCorrds


def getRaDecMinMaxPatchList(patchList, tractInfo, pad=0.0, nDecimals=4, raMin=360.0, raMax=0.0,
                            decMin=90.0, decMax=-90.0):
    """Find the max and min RA and Dec (deg) boundaries encompased in the
    patchList.

    Parameters
    ----------
    patchList : `list` of `str`
        List of patch IDs.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
        Tract information associated with the patches in ``patchList``.
    pad : `float`, optional
        Pad the boundary by ``pad`` degrees.
    nDecimals : `int`, optional
        Round coordinates to this number of decimal places.
    raMin, raMax : `float`, optional
        Initiate minimum[maximum] RA determination at raMin[raMax] (deg).
    decMin, decMax : `float`, optional
        Initiate minimum[maximum] Dec determination at decMin[decMax] (deg).

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains the RA and Dec min and max values for the patchList provided.
    """
    for ip, patch in enumerate(tractInfo):
        if str(patch.getIndex()[0]) + "," + str(patch.getIndex()[1]) in patchList:
            raPatch, decPatch = bboxToXyCoordLists(patch.getOuterBBox(), wcs=tractInfo.getWcs())
            raMin = min(np.round(min(raPatch) - pad, nDecimals), raMin)
            raMax = max(np.round(max(raPatch) + pad, nDecimals), raMax)
            decMin = min(np.round(min(decPatch) - pad, nDecimals), decMin)
            decMax = max(np.round(max(decPatch) + pad, nDecimals), decMax)
    return Struct(
        raMin=raMin,
        raMax=raMax,
        decMin=decMin,
        decMax=decMax,
    )


def percent(values, p=0.5):
    """Return a value a faction of the way between the min and max values in a
    list.
    """
    m = min(values)
    interval = max(values) - m
    return m + p*interval


def setPtSize(num, ptSize=12):
    """Set the point size according to the size of the catalog.
    """
    if num > 10:
        ptSize = min(12, max(3, int(20/np.log10(num))))
    return ptSize


def getQuiver(x, y, e1, e2, ax, color=None, scale=3, width=0.005, label=""):
    """Return the quiver object for the given input parameters.
    """
    theta = [np.math.atan2(a, b)/2.0 for a, b in zip(e1, e2)]
    e = np.sqrt(e1**2 + e2**2)
    c1 = e*np.cos(theta)
    c2 = e*np.sin(theta)
    if color is None:
        color = e
    q = ax.quiver(x, y, c1, c2, color=color, angles="uv", scale=scale, units="width", pivot="middle",
                  width=width, headwidth=0.0, headlength=0.0, headaxislength=0.0, label=label)
    return q


def makeAlphaCmap(cmap=plt.cm.viridis, alpha=1.0):
    """Given a matplotlib colormap, return it but with given alpha
    transparency.

    Parameters
    ----------
    cmap : `matplotlib.colors.ListedColormap`, optional
        The matplotlib colormap to make transparent with level ``alpha``.
    alpha : `float`, optional
        The matplotlib blending value, between 0 (transparent) and 1 (opaque).

    Returns
    -------
    alphaCmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap ``cmap`` but with transparency level ``alpha``.
    """
    alphaCmap = cmap(np.arange(cmap.N))
    alphaCmap[:, -1] = alpha
    alphaCmap = ListedColormap(alphaCmap)
    return alphaCmap


def buildTractImage(butler, dataId, tractInfo, patchList=None, coaddName="deep"):
    """Build up an image of an entire tract or list of patches.

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
        The butler associated with the data of interest.
    dataId : `lsst.daf.persistence.DataId`
        An instance of `lsst.daf.persistence.DataId` from which to extract the
        filter name.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
        Tract information object.
    patchList : `list` of `str`, optional
        A list of the patches to include.  If `None`, the full list of patches
        in ``tractInfo`` will be included.
    coaddName : `str`, optional
        The base name of the coadd (e.g. "deep" or "goodSeeing").

    Raises
    ------
    RuntimeError
        If ``nPatches`` is zero, i.e. no data was found.

    Returns
    -------
    image : `lsst.afw.image.ImageF`
        The full tract or patches in ``patchList`` image.
    """
    tractBbox = tractInfo.getBBox()
    nPatches = 0
    if not patchList:
        patchList = []
        nPatchX, nPatchY = tractInfo.getNumPatches()
        for iPatchX in range(nPatchX):
            for iPatchY in range(nPatchY):
                patchList.append("%d,%d" % (iPatchX, iPatchY))
    tractArray = np.full((tractBbox.getMaxY() + 1, tractBbox.getMaxX() + 1), np.nan, dtype="float32")
    for patch in patchList:
        expDataId = {"filter": dataId["filter"], "tract": tractInfo.getId(), "patch": patch}
        try:
            exp = butler.get(coaddName + "Coadd_calexp", expDataId)
            bbox = exp.getBBox()
            tractArray[bbox.getMinY():bbox.getMaxY() + 1,
                       bbox.getMinX():bbox.getMaxX() + 1] = exp.maskedImage.image.array
            nPatches += 1
        except Exception:
            continue
    if nPatches == 0:
        raise RuntimeError("No data found for tract {:}".format(tractInfo.getId()))
    tractArray = np.flipud(tractArray)
    image = afwImage.ImageF(geom.ExtentI(tractBbox.getMaxX() + 1, tractBbox.getMaxY() + 1))
    image.array[:] = tractArray
    return image


def determineExternalCalLabel(repoInfo, patch, coaddName="deep"):
    """Determine external calibration applied to make coadd.

    Make a string representing the external calibrations used in assembling the
    coadd to serve as a label for the plots.  Note that this is really a "best
    guess" based on the datasets that exist in the repo.  The prioritized
    assignment is based on the assumption that a particular calibration type
    will have been applied in coaddition if it does indeed exist in the repo
    (most notably, it is assumed that any fgcm photoCalib would have been used
    for photometric calibration even if the jointcal equivalent exists).

    External photometric calibration:
    fcr_md                dataset exists: label as MEAS_MOSAIC
    fgcm_tract_photoCalib dataset exists: label as FGCMTRACT
    fgcm_photoCalib       dataset exists: label as FGCM
    jointcal_photoCalib   dataset exists: label as JOINTCAL

    External astrometric calibration:
    fcr_md                dataset exists: label as MEAS_MOSAIC
    jointcal_wcs          dataset exists: label as JOINTCAL

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.struct.Struct`
        A struct containing elements with repo information needed to create
        appropriate dataIds to look for the external calibration datasets.
    patch : `str`
        An existing patch to use in the coaddDataId.
    coaddName : `str`, optional
        The base name of the coadd (e.g. "deep" or "goodSeeing").

    Returns
    -------
    uberCalLabel : `str`
        The label to be used for the external calibration used.
    """
    # Find a visit/ccd input so that you can check for meas_mosaic input (i.e.
    # to set uberCalLabel).
    if repoInfo.isGen3:
        return "Unknown_Gen3"

    coaddDataId = {"tract": repoInfo.tractInfo.getId(), "patch": patch, "filter": repoInfo.filterName}
    fname = repoInfo.butler.getUri(coaddName + "Coadd_calexp", coaddDataId)
    coaddInputs = afwImage.ExposureFitsReader(fname).readExposureInfo().getCoaddInputs()
    try:
        visitDataId = {"visit": coaddInputs.ccds[0]["visit"], "ccd": coaddInputs.ccds[0]["ccd"],
                       "filter": repoInfo.filterName, "tract": repoInfo.tractInfo.getId()}
        repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)
    except Exception:  # The above will throw if ccd is not a valid dataId key, try detector instead
        visitDataId = {"visit": coaddInputs.ccds[0]["visit"], "detector": coaddInputs.ccds[0]["ccd"],
                       "filter": repoInfo.filterName, "tract": repoInfo.tractInfo.getId()}
        repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)

    if repoInfo.butler.datasetExists("fcr_md", dataId=visitDataId):
        uberCalLabel = "externCal: MEAS_MOSAIC"
    elif not repoInfo.butler.datasetExists("fcr_md", dataId=visitDataId):
        uberCalLabel = "photoCal: "
        if repoInfo.camera.getName() == "HSC":  # fgcm only in play for HSC data to date
            if repoInfo.butler.datasetExists("fgcm_tract_photoCalib", dataId=visitDataId):
                uberCalLabel += "FGCMTRACT"
            elif repoInfo.butler.datasetExists("fgcm_photoCalib", dataId=visitDataId):
                uberCalLabel += "FGCM"
        if ("FGCM" not in uberCalLabel
                and repoInfo.butler.datasetExists("jointcal_photoCalib", dataId=visitDataId)):
            uberCalLabel += "JOINTCAL"
        elif uberCalLabel == "photoCal: ":
            uberCalLabel += "SFM"
        uberCalLabel += "  wcs: "
        if repoInfo.butler.datasetExists("jointcal_wcs", dataId=visitDataId):
            uberCalLabel += "JOINTCAL"
        else:
            uberCalLabel += "SFM"

    return uberCalLabel


def plotRhoStats(axes, rhoStats):
    """Plot all five Rho Statistics.

    Parameters
    ----------
    axes : `list`
        A list containing three `matplotlib.figure.ax` handles.
    rhoStats : `dict`
        A Python dictionary object with keys 1..5, each containing a
        `treecorr.GGCorrelation` object and key 0 containing a
        `treecorr.KKCorrelation` object.
    """
    for rhoIndex in range(6):
        rho = rhoStats[rhoIndex]

        # The mapping creates plots as in DES papers
        ax = axes[1 if rhoIndex in (2, 5) else 2 if rhoIndex == 0 else 0]
        colorStr = "C{}".format(rhoIndex)
        xi = rho.xip if rhoIndex > 0 else rho.xi
        isPositive = xi > 0
        #  KK/GGCorrelation objects have different entries for variance in x
        if hasattr(rho, "varxi"):
            yErrPositive = np.sqrt(rho.varxi)[isPositive]
            yErrNotPositive = np.sqrt(rho.varxi)[~isPositive]
        elif hasattr(rho, "varxip"):
            yErrPositive = np.sqrt(rho.varxip)[isPositive]
            yErrNotPositive = np.sqrt(rho.varxip)[~isPositive]
        else:
            yErrPositive = None
            yErrNotPositive = None
        ax.errorbar(rho.meanr[isPositive], xi[isPositive], yerr=yErrPositive,
                    color=colorStr, fmt="o", label=r"$\rho_{0}(\theta)$".format(rhoIndex))
        ax.errorbar(rho.meanr[~isPositive], -xi[~isPositive], yerr=yErrNotPositive,
                    color=colorStr, fillstyle="none", fmt="o", label=None)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(r"$\rho(\theta)$")
        ax.set_xlabel(r"$\theta$ (arcmin)")
        ax.legend(loc="lower left")


def getPlotInfo(repoInfo):
    """Parse the repoInfo into a dict of useful info for plots.

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.Struct`
        The struct has the following components:
            ``butler``
                The butler (`lsst.daf.persistence.butler`) associated with the
                ``dataset`` to be plotted.
            ``cameraName``
                The name of the camera used to take the data (`str`).
            ``dataId``
                The dataId associated with the dataset
                (`lsst.daf.persistence.DataId`).
            ``filterName``
                The name of the filter associated with the data (`str`).
            ``genericBandName``
                The generic band name associated with ``filterName`` (`str`).
            ``metadata``
                The metadata associated with the data
                (`lsst.daf.base.propertyContainer.PropertyList`).
            ``hscRun``
                A string representing "HSCPIPE_VERSION" fits header if the data
                were processed with the (now obsolete, but old reruns still
                exist) "HSC stack", `None` otherwise (`str` or `None`).
            ``dataset``
                The dataset name, ("src" if ``dataRef`` is visit level,
                coaddName + coaddDataset if ``dataRef`` is a coadd (`str`).
            ``skyMap``
                The sky map associated with the data
                (`lsst.skymap.SkyMap` or `None`).
            ``wcs``
                The wcs of the coadd image (`lsst.afw.geom.SkyWcs` or `None`).
            ``tractInfo``
                The tract information associated with the dataset
                (`lsst.skymap.tractInfo.ExplicitTractInfo` or `None`).

    Returns
    -------
    plotInfoDict : `dict`
        A dictionary of information about the data being plotted.
            ``"camera"``
                The camera used to take the data
                (`lsst.afw.cameraGeom.Camera`).
            ``"cameraName"``
                The name of the camera used to take the data (`str`).
            ``"ccdKey"``
                The ccd/dectector key associated with this camera (`str`).
            ``"filter"``
                The filter used for this data (`str`).
            ``"tract"``
                The tract that associated with the data (`str`).
            ``"visit"``
                The visit of the data; only included if the data is from a
                single epoch dataset (`str`).
            ``"patch"``
                The patch that the data is from; only included if the data are
                from a coadd dataset (`str`).
            ``"photoCalibDataset"``
                The dataset used for the calibration, e.g: jointcal or fgcm
                (`str`).
            ``"skyWcsDataset"``
                The sky Wcs dataset used (`str`).
            ``"rerun"``
                The rerun the data is stored in (`str`).
    """
    camera = repoInfo.camera
    cameraName = camera.getName()
    dataId = repoInfo.dataId
    if repoInfo.isGen3 and "band" in dataId:
        filterName = dataId["band"]
    else:
        filterName = dataId["filter"]
    ccdKey = repoInfo.ccdKey
    # Try to get the visit and patch id.  Set to None if not available.
    try:
        visit = str(dataId["visit"])
    except KeyError:
        visit = None
    try:
        patch = dataId["patch"]
    except KeyError:
        patch = None

    tract = str(dataId["tract"])
    photoCalibDataset = repoInfo.photoCalibDataset
    skyWcsDataset = repoInfo.skyWcsDataset
    try:
        rerun = list(repoInfo.butler.storage.repositoryCfgs)[0]
    except AttributeError:
        rootDir = str(repoInfo.butler.datastore.root)
        rootDir = rootDir.replace("file://", "")
        rerun = rootDir + repoInfo.butler.collections[0]

    plotInfoDict = dict(camera=camera, cameraName=cameraName, filter=filterName, ccdKey=ccdKey, tract=tract,
                        visit=visit, patch=patch, photoCalibDataset=photoCalibDataset,
                        skyWcsDataset=skyWcsDataset, rerun=rerun)
    return plotInfoDict
