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

import re
import operator

import astropy.coordinates as coord
import astropy.time
import astropy.units as units
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import scipy.odr as scipyOdr
import scipy.optimize as scipyOptimize
import scipy.stats as scipyStats

from contextlib import contextmanager

from lsst.pipe.base import Struct, TaskError
from lsst.pipe.tasks.parquetTable import ParquetTable, MultilevelParquetTable

import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.sphgeom as sphgeom
import lsst.verify as verify
import treecorr

from deprecated.sphinx import deprecated

try:
    from lsst.meas.mosaic.updateExposure import applyMosaicResultsCatalog
except ImportError:
    applyMosaicResultsCatalog = None

__all__ = ["Data", "Stats", "Enforcer", "MagDiff", "MagDiffMatches", "MagDiffCompare",
           "AstrometryDiff", "AngularDistance", "TraceSize", "PsfTraceSizeDiff", "TraceSizeCompare",
           "PercentDiff", "E1", "E2", "E1Resids", "E2Resids", "E1ResidsHsmRegauss", "E2ResidsHsmRegauss",
           "FootAreaDiffCompare", "MagDiffErr", "MagDiffCompareErr", "ApCorrDiffErr",
           "CentroidDiff", "CentroidDiffErr", "deconvMom", "deconvMomStarGal", "concatenateCatalogs",
           "joinMatches", "matchAndJoinCatalogs", "checkIdLists", "checkPatchOverlap", "joinCatalogs",
           "getFluxKeys", "addColumnsToSchema", "addApertureFluxesHSC", "addFpPoint", "addFootprintArea",
           "addRotPoint", "makeBadArray", "addFlag", "addElementIdColumn", "addIntFloatOrStrColumn",
           "calibrateSourceCatalogMosaic", "calibrateSourceCatalogPhotoCalib", "calibrateSourceCatalog",
           "backoutApCorr", "matchNanojanskyToAB", "checkHscStack", "fluxToPlotString", "andCatalog",
           "writeParquet", "getRepoInfo", "findCcdKey", "getCcdNameRefList", "getDataExistsRefList",
           "orthogonalRegression", "distanceSquaredToPoly", "p1CoeffsFromP2x0y0", "p2p1CoeffsFromLinearFit",
           "lineFromP2Coeffs", "linesFromP2P1Coeffs", "makeEqnStr", "catColors", "setAliasMaps",
           "addPreComputedColumns", "addMetricMeasurement", "updateVerifyJob", "computeMeanOfFrac",
           "calcQuartileClippedStats", "savePlots", "getSchema", "loadRefCat",
           "loadDenormalizeAndUnpackMatches", "loadReferencesAndMatchToCatalog",
           "computePhotoCalibScaleArray", "computeAreaDict", "determineIfSrcOnElement",
           "getParquetColumnsList"]


NANOJANSKYS_PER_AB_FLUX = (0*units.ABmag).to_value(units.nJy)
log = logging.getLogger(__name__)


def savePlots(plotList, plotType, dataId, butler, subdir=""):
    """Persist plots and parse stats yielded by the supplied generator.

    Parameters
    ----------
    plotList : `list`
        List of generators that will yield plots.
    plotType : `str`
        Tells the butler what type of plot is being saved.
    dataId : `dict`
        The dataId that will be used in persisting plots.
    butler : `lsst.daf.persistence.Butler`
        The butler that might be used in persisting the plots.
    subdir : `str`, optional
        If desired, an additional subdirectory to the plot output filenames
        (useful for testing data subsets).
    """
    allStats = {}
    allStatsHigh = {}
    for plots in plotList:
        # In this context next is called to "prime" the generator
        # Each plotting function yields right after it is called so that
        # each python execution frame is created, but not run. Calling next
        # on the reference to the plot function begins the execution of the
        # body, which will then yield each plot the function creates.
        next(plots)
        for plot in plots:
            if hasattr(plot, "dpi"):
                dpi = plot.dpi
                plot.fig.set_dpi(dpi)
            if hasattr(plot, "description"):
                dataId["description"] = plot.description
                dataId["subdir"] = "/" + subdir
                dataId["style"] = plot.style
                key = plot.description
                butler.put(plot.fig, plotType, dataId)
            else:
                raise AttributeError("Yielded struct is missing description")
            plt.close(plot.fig)
            if hasattr(plot, "stats"):
                allStats[key] = plot.stats
            if hasattr(plot, "statsHigh"):
                allStatsHigh[key] = plot.statsHigh
    return allStats, allStatsHigh


def writeParquet(dataRef, table, badArray=None, prefix=""):
    """Write an afwTable to a desired ParquetTable butler dataset.

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        Reference to butler dataset.
    table : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        Table to be written to parquet.
    badArray : `numpy.ndarray` or `None`, optional
        Boolean array with same length as catalog whose values indicate whether
        the source was deemed inappropriate for qa analyses.
    prefix : `str`, optional
        A string to be prepended to the column id name.

    Notes
    -----
    If ``table`` is an instance of `lsst.afw.table.SourceCatalog`, this
    function first converts the afwTable to an astropy table, then to a pandas
    DataFrame, which is then written to parquet format using the butler.
    """
    schema = getSchema(table)
    if badArray is not None and "qaBad_flag" not in schema:
        # Add flag indicating source "badness" for qa analyses for the benefit
        # of the Parquet files being written to disk for subsequent interactive
        # QA analysis.
        table = addFlag(table, badArray, "qaBad_flag", "Set to True for any source deemed bad for qa")
    if isinstance(table, pd.DataFrame):
        df = table
    else:
        df = table.asAstropy().to_pandas() if not isinstance(table, pd.DataFrame) else table
        df = df.set_index(prefix + "id", drop=False)

    dataRef.put(ParquetTable(dataFrame=df))


class Data(Struct):
    def __init__(self, catalog, quantity, mag, signalToNoise, selection, color, error=None, plot=True):
        Struct.__init__(self, catalog=catalog[selection].copy(deep=True), quantity=quantity[selection],
                        mag=mag[selection], signalToNoise=signalToNoise[selection], selection=selection,
                        color=color, plot=plot, error=error[selection] if error is not None else None)


class Stats(Struct):
    def __init__(self, dataUsed, num, total, mean, stdev, forcedMean, median, clip, thresholdType,
                 thresholdValue):
        Struct.__init__(self, dataUsed=dataUsed, num=num, total=total, mean=mean, stdev=stdev,
                        forcedMean=forcedMean, median=median, clip=clip, thresholdType=thresholdType,
                        thresholdValue=thresholdValue)

    def __repr__(self):
        return ("Stats(mean={0.mean:.4f}; stdev={0.stdev:.4f}; num={0.num:d}; total={0.total:d}; "
                "median={0.median:.4f}; clip={0.clip:.4f}; forcedMean={0.forcedMean:}; "
                "thresholdType={0.thresholdType:s}; thresholdValue={0.thresholdValue:})".format(self))


class Enforcer(object):
    """Functor for enforcing limits on statistics.
    """
    def __init__(self, requireGreater={}, requireLess={}, doRaise=False):
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.doRaise = doRaise

    def __call__(self, stats, dataId, log, description):
        for label in self.requireGreater:
            for ss in self.requireGreater[label]:
                value = getattr(stats[label], ss)
                if value <= self.requireGreater[label][ss]:
                    text = ("%s %s = %.2f exceeds minimum limit of %.2f: %s" %
                            (description, ss, value, self.requireGreater[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)
        for label in self.requireLess:
            for ss in self.requireLess[label]:
                value = getattr(stats[label], ss)
                if value >= self.requireLess[label][ss]:
                    text = ("%s %s = %.2f exceeds maximum limit of %.2f: %s" %
                            (description, ss, value, self.requireLess[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)


class MagDiff(object):
    """Functor to calculate magnitude difference.
    """
    def __init__(self, col1, col2, unitScale=1.0):
        self.col1 = col1
        self.col2 = col2
        self.unitScale = unitScale

    def __call__(self, catalog1, catalog2=None):
        catalog2 = catalog2 if catalog2 is not None else catalog1
        return -2.5*np.log10(catalog1[self.col1]/catalog2[self.col2])*self.unitScale


class MagDiffErr(object):
    """Functor to calculate magnitude difference error.
    """
    def __init__(self, col1, col2, unitScale=1.0):
        self.col1 = col1
        self.col2 = col2
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = 2.5*np.log10(np.e)*(catalog[self.col1 + "Err"]/catalog[self.col1])
        err2 = 2.5*np.log10(np.e)*(catalog[self.col2 + "Err"]/catalog[self.col2])
        return np.sqrt(err1**2 + err2**2)*self.unitScale


class MagDiffMatches(object):
    """Functor to calculate magnitude difference for match catalog.
    """
    def __init__(self, column, colorterm, zp=27.0, unitScale=1.0):
        self.column = column
        self.colorterm = colorterm
        self.zp = zp
        self.unitScale = unitScale

    def __call__(self, catalog):
        ref1 = -2.5*np.log10(catalog["ref_" + self.colorterm.primary + "_flux"])
        ref2 = -2.5*np.log10(catalog["ref_" + self.colorterm.secondary + "_flux"])
        ref = self.colorterm.transformMags(ref1, ref2)
        src = self.zp - 2.5*np.log10(catalog["src_" + self.column])
        return (src - ref)*self.unitScale


class MagDiffCompare(object):
    """Functor to calculate magnitude difference between two entries in
    comparison catalogs.

    Note that the column entries are in flux units and converted to mags here.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        src1 = -2.5*np.log10(catalog["first_" + self.column])
        src2 = -2.5*np.log10(catalog["second_" + self.column])
        return (src1 - src2)*self.unitScale


class AstrometryDiff(object):
    """Functor to calculate difference between astrometry.
    """
    def __init__(self, first, second, declination1=None, declination2=None, unitScale=1.0):
        self.first = first
        self.second = second
        self.declination1 = declination1
        self.declination2 = declination2
        self.unitScale = unitScale

    def __call__(self, catalog):
        first = catalog[self.first]
        second = catalog[self.second]
        cosDec1 = np.cos(catalog[self.declination1]) if self.declination1 is not None else 1.0
        cosDec2 = np.cos(catalog[self.declination2]) if self.declination2 is not None else 1.0
        return (first*cosDec1 - second*cosDec2)*(1.0*geom.radians).asArcseconds()*self.unitScale


class AngularDistance(object):
    """Functor to calculate the Haversine angular distance between two points.

    The Haversine formula, which determines the great-circle distance between
    two points on a sphere given their longitudes (ra) and latitudes (dec), is
    given by:

    distance =
    2*arcsin(
       sqrt(sin**2((dec2-dec1)/2) + cos(del1)cos(del2)sin**2((ra1-ra2)/2)))

    Parameters
    ----------
    raStr1 : `str`
       The name of the column for the RA (in radians) of the first point.
    decStr1 : `str`
       The name of the column for the Dec (in radians) of the first point.
    raStr2 : `str`
       The name of the column for the RA (in radians) of the second point.
    decStr1 : `str`
       The name of the column for the Dec (in radians) of the second point.
    catalog : `lsst.afw.table.SourceCatalog`
       The source catalog under consideration containing columns representing
       the (RA, Dec) coordinates for each object with names given by
       ``raStr1``, ``decStr1``, ``raStr2``, and ``decStr2``.

    Returns
    -------
    angularDistance : `numpy.ndarray`
       An array containing the Haversine angular distance (in radians) between
       the points:
       (``catalog``[``ra1Str``], ``catalog``[``dec1Str``]) and
       (``catalog``[``ra2Str``], ``catalog``[``dec2Str``]).
    """
    def __init__(self, raStr1, raStr2, decStr1, decStr2):
        self.raStr1 = raStr1
        self.raStr2 = raStr2
        self.decStr1 = decStr1
        self.decStr2 = decStr2

    def __call__(self, catalog):
        ra1 = catalog[self.raStr1]
        ra2 = catalog[self.raStr2]
        deltaRa = ra1 - ra2
        dec1 = catalog[self.decStr1]
        dec2 = catalog[self.decStr2]
        deltaDec = dec1 - dec2
        haverDeltaRa = np.sin(deltaRa/2.00)
        haverDeltaDec = np.sin(deltaDec/2.00)
        haverAlpha = np.sqrt(np.square(haverDeltaDec) + np.cos(dec1)*np.cos(dec2)*np.square(haverDeltaRa))
        angularDistance = 2.0*np.arcsin(haverAlpha)
        return angularDistance


class TraceSize(object):
    """Functor to calculate trace radius size for sources.
    """
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        return np.array(srcSize)


class PsfTraceSizeDiff(object):
    """Functor to calculate trace radius size difference (%) between object and
    PSF model.
    """
    def __init__(self, column, psfColumn):
        self.column = column
        self.psfColumn = psfColumn

    def __call__(self, catalog):
        srcSize = np.sqrt(0.5*(catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        psfSize = np.sqrt(0.5*(catalog[self.psfColumn + "_xx"] + catalog[self.psfColumn + "_yy"]))
        sizeDiff = 100*(srcSize - psfSize)/(0.5*(srcSize + psfSize))
        return np.array(sizeDiff)


class TraceSizeCompare(object):
    """Functor to calculate trace radius size difference (%) between objects in
    matched catalog.
    """
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        srcSize1 = np.sqrt(0.5*(catalog["first_" + self.column + "_xx"]
                                + catalog["first_" + self.column + "_yy"]))
        srcSize2 = np.sqrt(0.5*(catalog["second_" + self.column + "_xx"]
                                + catalog["second_" + self.column + "_yy"]))
        sizeDiff = 100.0*(srcSize1 - srcSize2)/(0.5*(srcSize1 + srcSize2))
        return np.array(sizeDiff)


class PercentDiff(object):
    """Functor to calculate the percent difference between a given column entry
    in matched catalog.
    """
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        value1 = catalog["first_" + self.column]
        value2 = catalog["second_" + self.column]
        percentDiff = 100.0*(value1 - value2)/(0.5*(value1 + value2))
        return np.array(percentDiff)


class E1(object):
    """Function to calculate e1 ellipticities from a given catalog.

    Parameters
    ----------
    column : `str`
        The name of the shape measurement algorithm. It should be one of
        ("base_SdssShape", "ext_shapeHSM_HsmSourceMoments") or
        ("base_SdssShape_psf", "ext_shapeHSM_HsmPsfMoments") for corresponding
        PSF ellipticities.
    unitScale : `float`, optional
        A numerical scaling factor to multiply the ellipticity.

    Returns
    -------
    e1 : `numpy.array`
        A numpy array of e1 ellipticity values.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        e1 = ((catalog[self.column + "_xx"]
               - catalog[self.column + "_yy"])/(catalog[self.column + "_xx"]
                                                + catalog[self.column + "_yy"]))
        return np.array(e1)*self.unitScale


class E2(object):
    """Function to calculate e2 ellipticities from a given catalog.

    Parameters
    ----------
    column : `str`
        The name of the shape measurement algorithm. It should be one of
        ("base_SdssShape", "ext_shapeHSM_HsmSourceMoments") or
        ("base_SdssShape_psf", "ext_shapeHSM_HsmPsfMoments") for corresponding
        PSF ellipticities.
    unitScale : `float`, optional
        A numerical scaling factor to multiply the ellipticity.

    Returns
    -------
    e2 : `numpy.array`
        A numpy array of e2 ellipticity values.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        e2 = (2.0*catalog[self.column + "_xy"]/(catalog[self.column + "_xx"] + catalog[self.column + "_yy"]))
        return np.array(e2)*self.unitScale


class E1Resids(object):
    """Functor to calculate e1 ellipticity residuals from an object catalog
    and PSF model.

    Parameters
    ----------
    column : `str`
        The name of the shape measurement algorithm. It should be one of
        ("base_SdssShape", "ext_shapeHSM_HsmSourceMoments").
    psfColumn : `str`
        The name used for PSF shape measurements from the same algorithm.
        It must be one of ("base_SdssShape_psf", "ext_shapeHSM_HsmPsfMoments")
        and correspond to the algorithm name specified for ``column``.
    unitScale : `float`, optional
        A numerical scaling factor to multiply both the object and PSF
        ellipticities.

    Returns
    -------
    e1Resids : `numpy.array`
        A numpy array of e1 residual ellipticity values.
    """
    def __init__(self, column, psfColumn, unitScale=1.0):
        self.column = column
        self.psfColumn = psfColumn
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE1func = E1(self.column, self.unitScale)
        psfE1func = E1(self.psfColumn, self.unitScale)

        srcE1 = srcE1func(catalog)
        psfE1 = psfE1func(catalog)

        e1Resids = srcE1 - psfE1
        return e1Resids


class E2Resids(object):
    """Functor to calculate e2 ellipticity residuals from an object catalog
    and PSF model.

    Parameters
    ----------
    column : `str`
        The name of the shape measurement algorithm. It should be one of
        ("base_SdssShape", "ext_shapeHSM_HsmSourceMoments").
    psfColumn : `str`
        The name used for PSF shape measurements from the same algorithm.
        It must be one of ("base_SdssShape_psf", "ext_shapeHSM_HsmPsfMoments")
        and correspond to the algorithm name specified for ``column``.
    unitScale : `float`, optional
        A numerical scaling factor to multiply both the object and PSF
        ellipticities.

    Returns
    -------
    e2Resids : `numpy.array`
        A numpy array of e2 residual ellipticity values.
    """
    def __init__(self, column, psfColumn, unitScale=1.0):
        self.column = column
        self.psfColumn = psfColumn
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE2func = E2(self.column, self.unitScale)
        psfE2func = E2(self.psfColumn, self.unitScale)

        srcE2 = srcE2func(catalog)
        psfE2 = psfE2func(catalog)

        e2Resids = srcE2 - psfE2
        return e2Resids


@deprecated(reason="This operation is ill-defined and must not be used. This functor will be removed "
            "without a replacement when ported to Gen3. Use `E1Resids()` for HSM shapes.",
            version="v22.0", category=FutureWarning)
class E1ResidsHsmRegauss(object):
    """Functor to calculate HSM e1 ellipticity residuals from a given star
    catalog and PSF model.
    """
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE1 = catalog["ext_shapeHSM_HsmShapeRegauss_e1"]
        psfE1 = ((catalog["ext_shapeHSM_HsmPsfMoments_xx"]
                  - catalog["ext_shapeHSM_HsmPsfMoments_yy"])/(catalog["ext_shapeHSM_HsmPsfMoments_xx"]
                                                               + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e1Resids = srcE1 - psfE1
        return np.array(e1Resids)*self.unitScale


@deprecated(reason="This operation is ill-defined and must not be used. This functor will be removed  "
            "without a replacement when ported to Gen3. Use `E2Resids()` for HSM shapes.",
            version="v22.0", category=FutureWarning)
class E2ResidsHsmRegauss(object):
    """Functor to calculate HSM e1 ellipticity residuals from a given star
    catalog and PSF model.
    """
    def __init__(self, unitScale=1.0):
        self.unitScale = unitScale

    def __call__(self, catalog):
        srcE2 = catalog["ext_shapeHSM_HsmShapeRegauss_e2"]
        psfE2 = (2.0*catalog["ext_shapeHSM_HsmPsfMoments_xy"]/(catalog["ext_shapeHSM_HsmPsfMoments_xx"]
                                                               + catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        e2Resids = srcE2 - psfE2
        return np.array(e2Resids)*self.unitScale


class RhoStatistics(object):
    """Functor to compute Rho statistics given star catalog and PSF model.

    For detailed description of Rho statistics, refer to
    Rowe (2010) and Jarvis et al., (2016).

    Parameters
    ----------
    column : `str`
        The name of the shape measurement algorithm. It should be one of
        ("base_SdssShape", "ext_shapeHSM_HsmSourceMoments").
    psfColumn : `str`
        The name used for PSF shape measurements from the same algorithm.
        It must be one of ("base_SdssShape_psf", "ext_shapeHSM_HsmPsfMoments")
        and correspond to the algorithm name specified for ``column``.
    **kwargs
        Additional keyword arguments passed to treecorr. See
        https://rmjarvis.github.io/TreeCorr/_build/html/gg.html for details.

    Returns
    -------
    rhoStats : `dict` [`int`, `treecorr.KKCorrelation` or
                              `treecorr.GGCorrelation`]
        A dictionary with keys 0..5, containing one `treecorr.KKCorrelation`
        object (key 0) and five `treecorr.GGCorrelation` objects corresponding
        to Rho statistic indices. rho0 corresponds to autocorrelation function
        of PSF size residuals.
    """
    def __init__(self, column, psfColumn, **kwargs):
        self.column = column
        self.psfColumn = psfColumn
        self.e1Func = E1(self.psfColumn)
        self.e2Func = E2(self.psfColumn)
        self.e1ResidsFunc = E1Resids(self.column, self.psfColumn)
        self.e2ResidsFunc = E2Resids(self.column, self.psfColumn)
        self.traceSizeFunc = TraceSize(self.column)
        self.psfTraceSizeFunc = TraceSize(self.psfColumn)
        self.kwargs = kwargs

    def __call__(self, catalog):
        e1 = self.e1Func(catalog)
        e2 = self.e2Func(catalog)
        e1Res = self.e1ResidsFunc(catalog)
        e2Res = self.e2ResidsFunc(catalog)
        traceSize2 = self.traceSizeFunc(catalog)**2
        psfTraceSize2 = self.psfTraceSizeFunc(catalog)**2
        SizeRes = (traceSize2 - psfTraceSize2)/(0.5*(traceSize2 + psfTraceSize2))

        isFinite = np.isfinite(e1Res) & np.isfinite(e2Res) & np.isfinite(SizeRes)
        e1 = e1[isFinite]
        e2 = e2[isFinite]
        e1Res = e1Res[isFinite]
        e2Res = e2Res[isFinite]
        SizeRes = SizeRes[isFinite]

        # Scale the SizeRes by ellipticities
        e1SizeRes = e1*SizeRes
        e2SizeRes = e2*SizeRes

        # Package the arguments to capture auto-/cross-correlations for the
        # Rho statistics.
        args = {0: (SizeRes, None),
                1: (e1Res, e2Res, None, None),
                2: (e1, e2, e1Res, e2Res),
                3: (e1SizeRes, e2SizeRes, None, None),
                4: (e1Res, e2Res, e1SizeRes, e2SizeRes),
                5: (e1, e2, e1SizeRes, e2SizeRes)}

        ra = np.rad2deg(catalog["coord_ra"][isFinite])*60.  # arcmin
        dec = np.rad2deg(catalog["coord_dec"][isFinite])*60.  # arcmin

        # Pass the appropriate arguments to the correlator and build a dict
        rhoStats = {rhoIndex: corrSpin2(ra, dec, *(args[rhoIndex]), raUnits="arcmin", decUnits="arcmin",
                                        **self.kwargs) for rhoIndex in range(1, 6)}
        rhoStats[0] = corrSpin0(ra, dec, *(args[0]), raUnits="arcmin", decUnits="arcmin", **self.kwargs)

        return rhoStats


class FootAreaDiffCompare(object):
    """Functor to calculate footprint area difference between two entries in
    comparison catalogs.
    """
    def __init__(self, column):
        self.column = column

    def __call__(self, catalog):
        footprintArea1 = catalog["first_" + self.column]
        footprintArea2 = catalog["second_" + self.column]
        return footprintArea1 - footprintArea2


class MagDiffCompareErr(object):
    """Functor to calculate magnitude difference error.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = 2.5*np.log10(np.e)*(catalog["first_" + self.column + "Err"]/catalog["first_" + self.column])
        err2 = 2.5*np.log10(np.e)*(catalog["second_" + self.column + "Err"]/catalog["second_" + self.column])
        return np.sqrt(err1**2 + err2**2)*self.unitScale


class ApCorrDiffErr(object):
    """Functor to calculate magnitude difference error.
    """
    def __init__(self, column, unitScale=1.0):
        self.column = column
        self.unitScale = unitScale

    def __call__(self, catalog):
        err1 = catalog["first_" + self.column + "Err"]
        err2 = catalog["second_" + self.column + "Err"]
        return np.sqrt(err1**2 + err2**2)*self.unitScale


class CentroidDiff(object):
    """Functor to calculate difference in astrometry.
    """
    def __init__(self, component, first="first_", second="second_", centroid1="base_SdssCentroid",
                 centroid2="base_SdssCentroid", unitScale=1.0):
        self.component = component
        self.first = first
        self.second = second
        self.centroid1 = centroid1
        self.centroid2 = centroid2
        self.unitScale = unitScale

    def __call__(self, catalog):
        first = self.first + self.centroid1 + "_" + self.component
        second = self.second + self.centroid2 + "_" + self.component
        return (catalog[first] - catalog[second])*self.unitScale


class CentroidDiffErr(CentroidDiff):
    """Functor to calculate difference error for astrometry.
    """
    def __call__(self, catalog):
        first = self.first + self.centroid + "_" + self.component + "Err"
        second = self.second + self.centroid + "_" + self.component + "Err"

        return np.hypot(catalog[first], catalog[second])*self.unitScale


def deconvMom(catalog):
    """Calculate deconvolved moments.
    """
    schema = getSchema(catalog)
    if "ext_shapeHSM_HsmSourceMoments_xx" in schema:
        hsm = catalog["ext_shapeHSM_HsmSourceMoments_xx"] + catalog["ext_shapeHSM_HsmSourceMoments_yy"]
    else:
        hsm = np.ones(len(catalog))*np.nan
    sdss = catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]
    if "ext_shapeHSM_HsmPsfMoments_xx" in schema:
        psfXxName = "ext_shapeHSM_HsmPsfMoments_xx"
        psfYyName = "ext_shapeHSM_HsmPsfMoments_yy"
    elif "base_SdssShape_psf_xx" in schema:
        psfXxName = "base_SdssShape_psf_xx"
        psfYyName = "base_SdssShape_psf_yy"
    else:
        raise RuntimeError("No psf shape parameter found in catalog")
    psf = catalog[psfXxName] + catalog[psfYyName]
    return np.where(np.isfinite(hsm), hsm, sdss) - psf


def deconvMomStarGal(catalog):
    """Calculate P(star) from deconvolved moments.
    """
    rTrace = deconvMom(catalog)
    snr = catalog["base_PsfFlux_instFlux"]/catalog["base_PsfFlux_instFluxErr"]
    poly = (-4.2759879274 + 0.0713088756641*snr + 0.16352932561*rTrace - 4.54656639596e-05*snr*snr
            - 0.0482134274008*snr*rTrace + 4.41366874902e-13*rTrace*rTrace + 7.58973714641e-09*snr*snr*snr
            + 1.51008430135e-05*snr*snr*rTrace + 4.38493363998e-14*snr*rTrace*rTrace
            + 1.83899834142e-20*rTrace*rTrace*rTrace)
    return 1.0/(1.0 + np.exp(-poly))


def concatenateCatalogs(catalogList):
    """Concatenate a list of catalogs.

    Parameters
    ----------
    catalogList : `list` of `lsst.afw.table.SourceCatalog`
       The `list` of catalogs to concatenate.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog` or `None`
       The concatenated catalog or `None` if ``catalogList`` is empty.
    """
    if len(catalogList) == 0:  # "No catalogs to concatenate"
        return None
    template = catalogList[0]
    schema = getSchema(template)
    catalog = type(template)(schema)
    catalog.reserve(sum(len(cat) for cat in catalogList))
    for cat in catalogList:
        catalog.extend(cat, True)
    return catalog


def joinMatches(matches, first="first_", second="second_"):
    """Join a match catalog into a base source catalog.

    Parameters
    ----------
    matches : `lsst.afw.table.match.SimpleMatch`
        The catalog of unpacked matches to join, i.e. a list of Match objects
        whose schema has "first" and "second" attributes which, resepectively,
        contain the reference and source catalog entries, and a "distance"
        field (the measured distance between the reference and source objects).

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        The joined matched catalog with prefixes ``first`` and ``second``
        for reference and source entries, respectively.
    """
    if not matches:
        return []

    mapperList = afwTable.SchemaMapper.join([matches[0].first.schema, matches[0].second.schema],
                                            [first, second])
    firstAliases = matches[0].first.schema.getAliasMap()
    secondAliases = matches[0].second.schema.getAliasMap()
    schema = mapperList[0].getOutputSchema()
    distanceKey = schema.addField("distance", type="Angle",
                                  doc="Distance between {0:s} and {1:s}".format(first, second))
    catalog = afwTable.BaseCatalog(schema)
    aliases = schema.getAliasMap()
    catalog.reserve(len(matches))
    for mm in matches:
        row = catalog.addNew()
        row.assign(mm.first, mapperList[0])
        row.assign(mm.second, mapperList[1])
        row.set(distanceKey, mm.distance*geom.radians)
    # make sure aliases get persisted to match catalog
    for k, v in firstAliases.items():
        aliases.set(first + k, first + v)
    for k, v in secondAliases.items():
        aliases.set(second + k, second + v)
    return catalog


def matchAndJoinCatalogs(catalog1, catalog2, matchRadius, raColStr="coord_ra", decColStr="coord_dec",
                         unit=units.rad, prefix1="first_", prefix2="second_", nthNeighbor=1, log=None,
                         matchXy=False, camera1=None, camera2=None):
    """Match two catalogs by RA/Dec or x/y using astropy and join the results.

    Parameters
    ----------
    catalog1, catalog2 : `pandas.core.frame.DataFrame`
        The two catalogs on which to do the matching.
    matchRadius : `float`
        The match radius within which to consider two objects a match in units
        of arcsec if ``matchXy`` is `False` else in pixels (which will get
        converted to arcsec prior to matching).
    raColStr, decColStr : `str`, optional
        The string names for the RA and Dec columns in the catalogs.
    unit : `astropy.units.core.IrreducibleUnit`, optional
        The astropy compliant unit to use in the matching.
    prefix1, prefix2 : `str`, optional
        The prefix strings to prepend to the two catalogs upon joining them.
    nthNeighbor : `int`, optional
        Which closest neighbor to search for in astropy's match_coordinates_sky
        function.  As per the astropy documentation, this is typically 1 as is
        appropriate for matching one set of coordinates to another. Another use
        case is 2, for matching a coordinate catalog against itself (i.e. when
        ``catalog1`` and ``catalog2`` are actually the same catalog, as in our
        overlaps identification case).  A value of 1 would inappropriate in
        that case because each point will find itself as the closest match).
        However, this is not robust agaist other zero distance matches (i.e.
        if any object other than the source itself has identical coordinates,
        the returned index may still be that of "self" as it will just choose
        between the two, and the choice may be that of "self").
    log : `lsst.log.Log` or `None`, optional
        Logger object for logging messages.
    camera1, camera2 : `lsst.afw.cameraGeom.Camera` or `None`, optional
        The cameras associated with ``catalog1`` and ``catalog2``.
    matchXy : `bool`, optional
        Whether to perform the matching in "x/y" pixel coordinates (these are
        converted to pseudo-arcsec coordinates to make use of astropy's
        match_coordinates_sky function).

    Raises
    ------
    RuntimeError
        If ``matchXy`` is `True` but either ``camera1`` or ``camera2`` was not
        provided.

    Returns
    -------
    matches : `pandas.core.frame.DataFrame`
        The matched and joined catalog.  The parameters associated with
        ``catalog1`` and ``catalog2`` are prefixed with ``prefix1`` and
        ``prefix2``, respectively.
    """
    if matchXy:
        if camera1 is None or camera2 is None:
            raise RuntimeError(f"matchXy is True, but at least one of the two cameras was not provided: "
                               f"camera1 = {camera1}, camera2 = {camera2}")
        # The astropy matching requires "sky" coordinates, so convert to rough
        # "arcsec" units.
        pixelSize1 = camera1[0].getPixelSize()[0]  # rough arcsec/pixel (assumes square pixels)
        pixelSize2 = camera2[0].getPixelSize()[0]  # rough arcsec/pixel (assumes square pixels)
        matchRadius *= pixelSize1  # convert from pixel to arcsec
        skyCoords1 = coord.SkyCoord(catalog1["slot_Centroid_x"]*pixelSize1,
                                    catalog1["slot_Centroid_y"]*pixelSize1,
                                    unit=units.arcsec)
        skyCoords2 = coord.SkyCoord(catalog2["slot_Centroid_x"]*pixelSize2,
                                    catalog2["slot_Centroid_y"]*pixelSize2,
                                    unit=units.arcsec)
    else:
        skyCoords1 = coord.SkyCoord(catalog1[raColStr], catalog1[decColStr], unit=unit)
        skyCoords2 = coord.SkyCoord(catalog2[raColStr], catalog2[decColStr], unit=unit)
    inds, dists, _ = coord.match_coordinates_sky(skyCoords1, skyCoords2, nthneighbor=nthNeighbor)
    if nthNeighbor > 1:
        selfMatches = [i == ind for i, ind in enumerate(inds)]
        if sum(selfMatches) > 0 and log is not None:
            log.warn("There were {} objects self-matched by "
                     "astropy.coordinates.match_coordinates_sky()").format(sum(selfMatches))
    matchedIds = dists < matchRadius*units.arcsec
    matchedIndices = inds[matchedIds]
    matchedDistances = dists[matchedIds]
    matchFirst = catalog1[matchedIds].copy(deep=True)
    matchSecond = catalog2.iloc[matchedIndices].copy(deep=True)
    matchFirst.rename(columns=lambda x: prefix1 + x, inplace=True)
    matchSecond.rename(columns=lambda x: prefix2 + x, inplace=True)
    matchFirst.index = pd.RangeIndex(len(matchFirst.index))
    matchSecond.index = pd.RangeIndex(len(matchSecond.index))
    matches = pd.concat([matchFirst, matchSecond], axis=1)
    matches["distance"] = matchedDistances.rad
    return matches


def checkIdLists(catalog1, catalog2, prefix=""):
    # Check to see if two catalogs have an identical list of objects by id
    schema1 = getSchema(catalog1)
    schema2 = getSchema(catalog2)
    idStrList = ["", ""]
    for i, schema in enumerate([schema1, schema2]):
        if "id" in schema:
            idStrList[i] = "id"
        elif "objectId" in schema:
            idStrList[i] = "objectId"
        elif prefix + "id" in schema:
            idStrList[i] = prefix + "id"
        elif prefix + "objectId" in schema:
            idStrList[i] = prefix + "objectId"
        else:
            raise RuntimeError("Cannot identify object id field (tried id, objectId, {0:}id, and "
                               "{0:}objectId)".format(prefix))
    identicalIds = np.all(catalog1[idStrList[0]] == catalog2[idStrList[1]])
    return identicalIds


def checkPatchOverlap(patchList, tractInfo):
    """Check for spatial overlap in list of patch references.

    Given a list of patch dataIds along with the associated tractInfo, check
    if any of the patches overlap.
    """
    for i, patch0 in enumerate(patchList):
        overlappingPatches = False
        patchIndex = [int(val) for val in patch0.split(",")] if isinstance(patch0, str) else patch0
        patchInfo = tractInfo.getPatchInfo(patchIndex)
        patchBBox0 = patchInfo.getOuterBBox()
        for j, patch1 in enumerate(patchList):
            if patch1 != patch0 and j > i:
                patchIndex = [int(val) for val in patch1.split(",")] if isinstance(patch1, str) else patch1
                patchInfo = tractInfo.getPatchInfo(patchIndex)
                patchBBox1 = patchInfo.getOuterBBox()
                xCen0, xCen1 = patchBBox0.getCenterX(), patchBBox1.getCenterX()
                yCen0, yCen1 = patchBBox0.getCenterY(), patchBBox1.getCenterY()
                if xCen0 == xCen1 or yCen0 == yCen1:  # omit patches that only overlap at corners
                    if patchBBox0.overlaps(patchBBox1):
                        overlappingPatches = True
                        break
        if overlappingPatches:
            break
    return overlappingPatches


def joinCatalogs(catalog1, catalog2, prefix1="cat1_", prefix2="cat2_"):
    """Join two catalogs row-by-row wiht optional prefixes.
    """
    # Make sure catalogs entries are all associated with the same object
    if not checkIdLists(catalog1, catalog2):
        raise RuntimeError("Catalogs with different sets of objects cannot be joined")

    mapperList = afwTable.SchemaMapper.join([catalog1[0].schema, catalog2[0].schema],
                                            [prefix1, prefix2])
    schema = mapperList[0].getOutputSchema()
    catalog = afwTable.BaseCatalog(schema)
    catalog.reserve(len(catalog1))
    for s1, s2 in zip(catalog1, catalog2):
        row = catalog.addNew()
        row.assign(s1, mapperList[0])
        row.assign(s2, mapperList[1])
    return catalog


def getFluxKeys(schema):
    """Retrieve the flux and flux error keys from a schema.

    Both are returned as dicts indexed on the flux name (e.g.
    "base_PsfFlux_instFlux" or "modelfit_CModel_instFlux").
    """
    if isinstance(schema, list):
        fluxKeys = {flux: flux for flux in schema if flux.endswith("_instFlux") and flux + "Err" in schema}
        errKeys = {flux + "Err": flux + "Err" for (flux, flux) in fluxKeys.items()}
    else:
        fluxTypeStr = "_instFlux"
        fluxSchemaItems = schema.extract("*" + fluxTypeStr)
        # Do not include any flag fields (as determined by their type).  Also
        # exclude slot fields, as these would effectively duplicate whatever
        # they point to.
        fluxKeys = dict((name, schemaItem.key) for name, schemaItem in list(fluxSchemaItems.items()) if
                        schemaItem.field.getTypeString() != "Flag"
                        and not name.startswith("slot"))
        errSchemaItems = schema.extract("*" + fluxTypeStr + "Err")
        errKeys = dict((name, schemaItem.key) for name, schemaItem in list(errSchemaItems.items()) if
                       name[:-len("Err")] in fluxKeys)

        # Also check for any in HSC format
        schemaKeys = dict((s.field.getName(), s.key) for s in schema)
        fluxKeysHSC = dict((name, key) for name, key in schemaKeys.items() if
                           (re.search(r"^(flux\_\w+|\w+\_flux)$", name)
                            or re.search(r"^(\w+flux\_\w+|\w+\_flux)$", name))
                           and not re.search(r"^(\w+\_apcorr)$", name) and name + "_err" in schemaKeys)
        errKeysHSC = dict((name + "_err", schemaKeys[name + "_err"]) for name in fluxKeysHSC.keys() if
                          name + "_err" in schemaKeys)
        if fluxKeysHSC:
            fluxKeys.update(fluxKeysHSC)
            errKeys.update(errKeysHSC)

    if not fluxKeys:
        raise RuntimeError("No flux keys found")

    return fluxKeys, errKeys


def addColumnsToSchema(fromCat, toCat, colNameList, prefix=""):
    """Copy columns from fromCat to new version of toCat.
    """
    fromMapper = afwTable.SchemaMapper(fromCat.schema)
    fromMapper.addMinimalSchema(toCat.schema, False)
    toMapper = afwTable.SchemaMapper(toCat.schema)
    toMapper.addMinimalSchema(toCat.schema)
    schema = fromMapper.editOutputSchema()
    for col in colNameList:
        colName = prefix + col
        fromKey = fromCat.schema.find(colName).getKey()
        fromField = fromCat.schema.find(colName).getField()
        schema.addField(fromField)
        toField = schema.find(colName).getField()
        fromMapper.addMapping(fromKey, toField, doReplace=True)

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(toCat))

    newCatalog.extend(toCat, toMapper)
    for srcFrom, srcTo in zip(fromCat, newCatalog):
        srcTo.assign(srcFrom, fromMapper)

    aliases = newCatalog.schema.getAliasMap()
    for k, v in toCat.schema.getAliasMap().items():
        aliases.set(k, v)

    return newCatalog


def addApertureFluxesHSC(catalog, prefix=""):
    mapper = afwTable.SchemaMapper(catalog[0].schema)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    apName = prefix + "base_CircularApertureFlux"
    apRadii = ["3_0", "4_5", "6_0", "9_0", "12_0", "17_0", "25_0", "35_0", "50_0", "70_0"]

    # for ia in range(len(apRadii)):
    # Just to 12 pixels for now...takes a long time...
    for ia in (4,):
        apFluxKey = schema.addField(apName + "_" + apRadii[ia] + "_instFlux", type="D",
                                    doc="flux within " + apRadii[ia].replace("_", ".") + "-pixel aperture",
                                    units="count")
        apFluxErrKey = schema.addField(apName + "_" + apRadii[ia] + "_instFluxErr", type="D",
                                       doc="1-sigma flux uncertainty")
    apFlagKey = schema.addField(apName + "_flag", type="Flag", doc="general failure flag")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))

    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        # for ia in range(len(apRadii)):
        for ia in (4,):
            row.set(apFluxKey, source[prefix+"flux_aperture"][ia])
            row.set(apFluxErrKey, source[prefix+"flux_aperture_err"][ia])
        row.set(apFlagKey, source[prefix + "flux_aperture_flag"])

    return newCatalog


def addFpPoint(det, catalog, prefix=""):
    """Compute Focal Plane coordinates for SdssCentroid of each source and add
    to schema.

    Parameters
    ----------
    det : `lsst.afw.cameraGeom.Detector`
        The detector (ccd) under consideration.
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which to add the Focal Plane point columns.
    prefix : `str`, optional
        An optional string to be prepended to the column id name.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog`
       New source catalog with the Focal Plane point and flag columns added.
    """
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    fpName = prefix + "base_FPPosition"
    fpxKey = schema.addField(fpName + "_x", type="D", doc="Position on the focal plane (in FP pixels)")
    fpyKey = schema.addField(fpName + "_y", type="D", doc="Position on the focal plane (in FP pixels)")
    fpFlag = schema.addField(fpName + "_flag", type="Flag", doc="Set to True for any fatal failure")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    xCentroidKey = catalog.schema[prefix + "base_SdssCentroid_x"].asKey()
    yCentroidKey = catalog.schema[prefix + "base_SdssCentroid_y"].asKey()
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            center = geom.Point2D(source[xCentroidKey], source[yCentroidKey])
            pixelsToFocalPlane = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
            fpPoint = pixelsToFocalPlane.applyForward(center)
        except Exception:
            fpPoint = geom.Point2D(np.nan, np.nan)
            row.set(fpFlag, True)
        row.set(fpxKey, fpPoint[0])
        row.set(fpyKey, fpPoint[1])
    return newCatalog


def addFootprintArea(catalog, fromCat=None, prefix=""):
    """Retrieve the number of pixels in an sources footprint and add to schema.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which to add the footprint area column.
    fromCat : `lsst.afw.table.SourceCatalog`, optional
        If not `None`, retrieve the footprints from this catalog.
    prefix : `str`, optional
        An optional string to be prepended to the column id name.

    Raises
    ------
    TaskError
        If lengths of ``catalog`` and ``fromCat`` (if not `None`) are not
        equal.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog`
        A new source catalog with the footprint area and flag columns added.
    """
    mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
    mapper.addMinimalSchema(catalog[0].schema)
    schema = mapper.getOutputSchema()
    fpName = prefix + "base_FootprintArea_value"
    fpFlagName = prefix + fpName[:fpName.find("value")] + "flag"
    fpKey = schema.addField(fpName, type="I",
                            doc="Area (i.e. number of pixels) in the source's detection footprint")
    fpFlag = schema.addField(fpFlagName, type="Flag", doc="Set to True for any fatal failure")
    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    if fromCat:
        if len(fromCat) != len(catalog):
            raise TaskError("Lengths of fromCat and catalog for getting footprint areas do not agree")
    if fromCat is None:
        fromCat = catalog
    for srcFrom, srcTo in zip(fromCat, catalog):
        row = newCatalog.addNew()
        row.assign(srcTo, mapper)
        try:
            footArea = srcFrom.getFootprint().getArea()
        except Exception:
            footArea = 0  # used to be np.nan, but didn't work.
            row.set(fpFlag, True)
        row.set(fpKey, footArea)
    return newCatalog


def rotatePixelCoord(src, width, height, nQuarter):
    """Rotate single (x, y) pixel coordinate such that LLC of detector in FP
    is (0, 0).
    """
    x0 = src["slot_Centroid_x"]
    y0 = src["slot_Centroid_y"]
    if nQuarter == 1:
        src["slot_Centroid_x"] = height - y0 - 1.0
        src["slot_Centroid_y"] = x0
    if nQuarter == 2:
        src["slot_Centroid_x"] = width - x0 - 1.0
        src["slot_Centroid_y"] = height - y0 - 1.0
    if nQuarter == 3:
        src["slot_Centroid_x"] = y0
        src["slot_Centroid_y"] = width - x0 - 1.0
    return src


def addRotPoint(catalog, width, height, nQuarter, prefix=""):
    """Compute rotated CCD pixel coords for comparing LSST vs HSC run
    centroids.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which to add the rotated point columns.
    width, height : `float`
        The width and height of the detector associated with the ``catalog``.
    nQuarter : `int`
        The number of 90 degree rotations of the detector associated with the
        ``catalog``.
    prefix : `str`, optional
        An optional string to be prepended to the column id name.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog`
        A new source catalog with the rotated point and flag columns added.
    """
    schema = getSchema(catalog[0])
    mapper = afwTable.SchemaMapper(schema, shareAliasMap=True)
    mapper.addMinimalSchema(schema)
    schema = mapper.getOutputSchema()
    rotName = prefix + "base_SdssCentroid_Rot"
    rotxKey = schema.addField(rotName + "_x", type="D", doc="Centroid x (in rotated pixels)")
    rotyKey = schema.addField(rotName + "_y", type="D", doc="Centroid y (in rotated pixels)")
    rotFlag = schema.addField(rotName + "_flag", type="Flag", doc="Set to True for any fatal failure")

    newCatalog = afwTable.SourceCatalog(schema)
    newCatalog.reserve(len(catalog))
    for source in catalog:
        row = newCatalog.addNew()
        row.assign(source, mapper)
        try:
            rotPoint = rotatePixelCoord(source, width, height, nQuarter).getCentroid()
        except Exception:
            rotPoint = geom.Point2D(np.nan, np.nan)
            row.set(rotFlag, True)
        row.set(rotxKey, rotPoint[0])
        row.set(rotyKey, rotPoint[1])

    return newCatalog


def makeBadArray(catalog, flagList=[], onlyReadStars=False, patchInnerOnly=True, tractInnerOnly=False,
                 useScarletModelForIsolated=False, excludeSkipped=True):
    """Create a boolean array indicating sources deemed unsuitable for qa
    analyses.

    Sets value to True for unisolated objects (deblend_nChild > 0), "sky"
    objects (merge_peak_sky), and any of the flags listed in
    config.analysis.flags.  If ``onlyReadStars`` is `True`, sets boolean as
    `True` for all galaxies classified as extended
    (base_ClassificationExtendedness_value > 0.5).  If ``patchInnerOnly`` is
    `True` (the default), sets the bad boolean array value to `True` for any
    sources for which detect_isPatchInner is `False` (to avoid duplicates in
    overlapping patches).  If ``tractInnerOnly`` is `True`, sets the bad
    boolean value to `True` for any sources for which detect_isTractInner is
    `False` (to avoid duplicates in overlapping patches).  Note, however, that
    the default for tractInnerOnly is `False` as we are currently only running
    these scripts at the per-tract level, so there are no tract duplicates (and
    omitting the "outer" ones would just leave an empty band around the tract
    edges).

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The source catalog under consideration.
    flagList : `list`, optional
        The list of flags for which, if any is set for a given source, set bad
        entry to `True` for that source.
    onlyReadStars : `bool`, optional
        Boolean indicating if you want to select objects classified as stars
        only (based on base_ClassificationExtendedness_value > 0.5).
    patchInnerOnly : `bool`, optional
        Whether to select only sources for which detect_isPatchInner is `True`.
    tractInnerOnly : `bool`, optional
        Whether to select only sources for which detect_isTractInner is `True`.
        Note that these scripts currently only ever run at the per-tract level,
        so we do not need to filter out sources for which detect_isTractInner
        is `False` as, with only one tract, there are no duplicated tract
        inner/outer sources.
    useScarletModelForIsolated : `bool`, optional
        For isolated sources, select the "model leaf" versions of the scarlet
        output if `True`.  If `False`, select the non-model, or "simple leaf"
        versions of the scarlet output for isolated sources.  Ignored if the
        ``catalog`` provided was not created using the scarlet deblender.
    excludeSkipped : `bool`, optional
        Whether to exclude objects marked as "deblend_skipped", i.e. those that
        were skipped by the deblender (e.g. because they were too big, did not
        have full-band coverage, no flux at the center of the source, etc.)

    Returns
    -------
    badArray : `numpy.ndarray`
        Boolean array with same length as catalog whose values indicate whether
        the source was deemed inappropriate for qa analyses.
    """
    schema = getSchema(catalog)
    bad = np.zeros(len(catalog), dtype=bool)
    if "detect_isPatchInner" in schema and patchInnerOnly:
        bad |= ~catalog["detect_isPatchInner"]
    if "detect_isTractInner" in schema and tractInnerOnly:
        bad |= ~catalog["detect_isTractInner"]

    if "deblend_scarletFlux" in schema:
        # TODO: The following if/else is just for backwards compatibility with
        # older scarlet catalogs that did not add the detect_isDeblendedSource
        # column.  Remove this condition when the few old catalogs in existence
        # become obsolete.
        if "detect_isDeblendedSource" in schema:
            if useScarletModelForIsolated:  # use the scarlet model-based results
                bad |= ~catalog["detect_isDeblendedModelSource"]
            else:  # use the non-model scarlet results
                bad |= ~catalog["detect_isDeblendedSource"]
                # Exclude isolated sources not passed to deblender as these
                # indicate regions where there are no scarlet models for
                # fromBlend objects in this area (because scarlet skips
                # deblending for  objects that do not have full band coverage).
                bad |= ((catalog["parent"] == 0) & (catalog["deblend_nChild"] == 0))
        else:
            # This is a pre-DM-29087 scarlet schema
            # Exclude parents that are blends of multiple sources
            bad |= ((catalog["parent"] == 0) & (catalog["deblend_nChild"] > 1))
            # Exclude isolated sources not passed to deblender
            bad |= ((catalog["parent"] == 0) & (catalog["deblend_nChild"] == 1))
    else:
        # TODO: again, this if/else is just for compatibility with older
        # catalogs that do not have the detect_isDeblendedSource column set.
        if "detect_isDeblendedSource" in schema:
            bad |= ~catalog["detect_isDeblendedSource"]
        else:
            bad |= catalog["deblend_nChild"] > 0  # Exclude non-deblended (i.e. parents)
    # Exclude parent blends that were skipped
    if excludeSkipped:
        bad |= catalog["deblend_skipped"]
    # Exclude "sky" objects from catalogs (column names differ for visit
    # and coadd catalogs).
    for skyObjectCol in ["merge_peak_sky", "sky_source"]:
        if skyObjectCol in schema:
            bad |= catalog[skyObjectCol]
    for flag in flagList:
        bad |= catalog[flag]
    if onlyReadStars and "base_ClassificationExtendedness_value" in schema:
        bad |= catalog["base_ClassificationExtendedness_value"] > 0.5
    return bad


def addFlag(catalog, badArray, flagName, doc="General failure flag"):
    """Add a flag for any sources deemed not appropriate for qa analyses.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        Source catalog to which the flag will be added.
    badArray : `numpy.ndarray`
        Boolean array with same length as catalog whose values indicate whether
        the flag flagName should be set for a given oject.
    flagName : `str`
        Name of flag to be set
    doc : `str`, optional
        Docstring for ``flagName``.

    Raises
    ------
    RuntimeError
        If lengths of ``catalog`` and ``badArray`` are not equal.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog` or
                 `pandas.core.frame.DataFrame`
        Source catalog with ``flagName`` column added.
    """
    if len(catalog) != len(badArray):
        raise RuntimeError("Lengths of catalog and bad objects array do not match.")

    if isinstance(catalog, pd.DataFrame):
        catalog[flagName] = badArray
        newCatalog = catalog
    else:
        mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
        mapper.addMinimalSchema(catalog[0].schema)
        schema = mapper.getOutputSchema()
        badFlag = schema.addField(flagName, type="Flag", doc=doc)
        newCatalog = afwTable.SourceCatalog(schema)
        newCatalog.reserve(len(catalog))
        newCatalog.extend(catalog, mapper)

        for i, row in enumerate(newCatalog):
            row.set(badFlag, bool(badArray[i]))
    return newCatalog


def addElementIdColumn(catalog, dataId, repoInfo=None):
    """Add a column indicating the image element (patch/ccd) ID.

    Parameters
    ----------
    catalog : `pandas.core.frame.DataFrame`
        Source catalog to which the column will be added.
    dataId : `dict`
        The `dict` of data id keys from which to extract the image element key.

    Raises
    ------
    RuntimeError
        If unable to determine image element (patch/ccd) key.

    Returns
    -------
    catalog : `pandas.core.frame.DataFrame`
        Source catalog with element Id column added.
    """
    if "patch" in dataId:
        elementKey = "patch"
        elementStr = "patch"
    elif repoInfo is not None:
        elementKey = repoInfo.ccdKey
        elementStr = "ccd"
    else:
        raise RuntimeError("Can't determine image element (e.g. patch/ccd) key")
    catalog[elementStr + "Id"] = dataId[elementKey]
    return catalog


def addIntFloatOrStrColumn(catalog, values, fieldName, fieldDoc, fieldUnits=""):
    """Add a column of values with name fieldName and doc fieldDoc to the
    catalog schema.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        Source catalog to which the column will be added.
    values : `list`, `numpy.ndarray`, or scalar of type `int`, `float`, or
             `str`
        The list of values to be added.  This list must have the same length as
        ``catalog`` or length 1 (to add a column with the same value for all
        objects).
    fieldName : `str`
        Name of the field to be added to the schema.
    fieldDoc : `str`
        Documentation string for the field to be added to the schema.
    fieldUnits : `str`, optional
        Units of the column to be added.

    Raises
    ------
    RuntimeError
        If type of all ``values`` is not one of `int`, `float`, or `str`.
    RuntimeError
        If length of ``values`` list is neither 1 nor equal to the ``catalog``
        length.

    Returns
    -------
    newCatalog : `lsst.afw.table.SourceCatalog` or
                 `pandas.core.frame.DataFrame`
        Source catalog with ``fieldName`` column added.
    """
    if isinstance(values, pd.Series):
        values = list(values)
    if not isinstance(values, (list, np.ndarray)):
        values = [values, ]
    if not any((isinstance(v, str) for v in values)):
        if not isinstance(values[np.where(np.isfinite(values))[0][0]], (int, float)):
            raise RuntimeError(("Have only accommodated int, float, or str types.  Type provided was : "
                                "{}.  (Note, if you want to add a boolean flag column, use the addFlag "
                                "function.)").format(type(values)))
    if len(values) not in (len(catalog), 1):
        raise RuntimeError(("Length of values list must be either 1 or equal to the catalog length "
                            "({0:d}).  Length of values list provided was: {1:d}").
                           format(len(catalog), len(values)))

    if isinstance(catalog, pd.DataFrame):
        catalog[fieldName] = values
        newCatalog = catalog
    else:
        size = None
        mapper = afwTable.SchemaMapper(catalog[0].schema, shareAliasMap=True)
        mapper.addMinimalSchema(catalog[0].schema)
        schema = mapper.getOutputSchema()

        if all(type(value) is int for value in values):
            fieldType = "I"
        elif all(type(value) is np.longlong for value in values):
            fieldType = "L"
        elif all(isinstance(value, float) for value in values):
            fieldType = "D"
        elif all(type(value) is str for value in values):
            fieldType = str
            size = len(max(values, key=len))
        else:
            raise RuntimeError(("Have only accommodated int, np.longlong, float, or str types.  Type "
                                "provided for the first element was: {} (and note that all values in the "
                                "list must have the same type.  Also note, if you want to add a boolean "
                                "flag column, use the addFlag function.)").format(type(values[0])))

        fieldKey = schema.addField(fieldName, type=fieldType, size=size, doc=fieldDoc, units=fieldUnits)

        newCatalog = afwTable.SourceCatalog(schema)
        newCatalog.reserve(len(catalog))

        newCatalog.extend(catalog, mapper)
        if len(values) == 1:
            for row in newCatalog:
                row.set(fieldKey, values[0])
        else:
            for i, row in enumerate(newCatalog):
                row.set(fieldKey, values[i])
    return newCatalog


def calibrateSourceCatalogMosaic(dataRef, catalog, fluxKeys=None, errKeys=None, zp=27.0):
    """Calibrate catalog with meas_mosaic results.

    Requires a SourceCatalog input.
    """
    result = applyMosaicResultsCatalog(dataRef, catalog, True)
    catalog = result.catalog
    ffp = result.ffp
    # Convert to constant zero point, as for the coadds
    factor = ffp.calib.getFluxMag0()[0]/10.0**(0.4*zp)

    if fluxKeys is None:
        fluxKeys, errKeys = getFluxKeys(catalog.schema)
    for fluxName, fluxKey in list(fluxKeys.items()) + list(errKeys.items()):
        if len(catalog[fluxKey].shape) > 1:
            continue
        catalog[fluxKey] /= factor
    return catalog


def calibrateSourceCatalogPhotoCalib(dataRef, catalog, photoCalibDataset, isGen3, fluxKeys=None, zp=27.0):
    """Calibrate catalog with PhotoCalib results.

    The suite of external photometric calibrations in existence include:
    fgcm, fgcm_tract, jointcal, and meas_mosaic (but the latter has been
    effectively retired).

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        The data reference for which the relevant datasets are to be retrieved.
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The source catalog to which the calibrations will be applied.
    photoCalibDataset : `str`:
        The name of the photoCalib dataset to be applied (e.g.
        "jointcal_photoCalib" or "fgcm_tract_photoCalib").
    fluxKeys : `dict` or `None`, optional
        A `dict` of the flux keys to which the photometric calibration will
        be applied.  If not provided, the getFluxKeys function will be used
        to set it.
    zp : `float`, optional
        A constant zero point magnitude to which to scale all the fluxes in
        ``fluxKeys``.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The calibrated catalog.

    Notes
    -----
    Adds columns with magnitudes and their associated errors for all flux
    fields returned by the getFluxKeys() method to the returned catalog.
    These columns are named <flux column>_mag and <flux column>_magErr,
    respectively.
    """
    if not isGen3:
        photoCalib = dataRef.get(photoCalibDataset)
    else:
        dataId = dataRef["dataId"]
        externalPhotoCalibCatalog = dataRef["butler"].get(photoCalibDataset, dataId=dataId)
        row = externalPhotoCalibCatalog.find(dataId["detector"])
        photoCalib = row.getPhotoCalib()
    schema = getSchema(catalog)
    # Scale to AB and convert to constant zero point, as for the coadds
    factor = NANOJANSKYS_PER_AB_FLUX/10.0**(0.4*zp)
    if fluxKeys is None:
        fluxKeys, errKeys = getFluxKeys(schema)

    magColsToAdd = []
    for fluxName, fluxKey in list(fluxKeys.items()):
        if len(catalog[fluxKey].shape) > 1:
            continue
        # photoCalib.instFluxToNanojansky() requires an error for each flux
        fluxErrKey = errKeys[fluxName + "Err"] if fluxName + "Err" in errKeys else None
        baseName = fluxName.replace("_instFlux", "")
        if fluxErrKey:
            if isinstance(catalog, pd.DataFrame):
                photoCalibArray, photoCalibErrArray = computePhotoCalibScaleArray(
                    photoCalib, catalog["slot_Centroid_x"].values, catalog["slot_Centroid_y"].values)
                errorHypot = np.hypot(catalog[fluxErrKey].values/catalog[fluxKey].values,
                                      photoCalibErrArray/photoCalibArray)
                catalog[fluxErrKey] = np.abs(catalog[fluxKey].values)*photoCalibArray*errorHypot
                catalog[fluxKey] *= photoCalibArray
                catalog[baseName + "_magErr"] = 2.5/np.log(10)*errorHypot
                catalog[baseName + "_mag"] = -2.5*np.log10(catalog[fluxKey])
            else:
                calibratedFluxAndErrArray = photoCalib.instFluxToNanojansky(catalog, baseName)
                catalog[fluxKey] = calibratedFluxAndErrArray[:, 0]
                catalog[fluxErrKey] = calibratedFluxAndErrArray[:, 1]
                if "Flux" in baseName:
                    magsAndErrArray = photoCalib.instFluxToMagnitude(catalog, baseName)
                    magColsToAdd.append((magsAndErrArray, baseName))
        else:
            # photoCalib requires an error for each flux, but some don't
            # have one in the schema (currently only certain deblender
            # fields, e.g. deblend_psf_instFlux), so we compute the flux
            # correction factor from any slot flux (it only depends on
            # position, so any slot with a successful measurement will do)
            # and apply that to any flux entries that do not have errors.
            for fluxSlotName in schema.extract("slot*instFlux"):
                photoCalibFactor = None
                for src in catalog:
                    if np.isfinite(src[fluxSlotName]):
                        baseSlotName = fluxSlotName.replace("_instFlux", "")
                        photoCalibFactor = (
                            photoCalib.instFluxToNanojansky(src, baseSlotName).value/src[fluxSlotName])
                        break
                if photoCalibFactor:
                    catalog[fluxKey] *= photoCalibFactor
                    break
        catalog[fluxKey] /= factor
        if fluxErrKey:
            catalog[fluxErrKey] /= factor

    for values, colName in magColsToAdd:
        fieldName = colName + "_mag"
        fieldDoc = "Magnitude calculated from " + colName
        catalog = addIntFloatOrStrColumn(catalog, values[:, 0], fieldName, fieldDoc, fieldUnits="mag")
        fieldName = colName + "_magErr"
        fieldDoc = "Magnitude error calculated from " + colName
        catalog = addIntFloatOrStrColumn(catalog, values[:, 1], fieldName, fieldDoc, fieldUnits="mag")

    return catalog


def computePhotoCalibScaleArray(photoCalib, xArray, yArray):
    """Compute the photometric calibration scaling value for a list of
    positions.

    The calibrated flux for any _instFlux or _instFluxErr value in a source
    catilog will thus be:
        _calibratedFlux[Err] = _instFlux[Err]*photoCalibScale

    Parameters
    ----------
    photoCalib : `lsst.afw.image.PhotoCalib`
        The `lsst.afw.image.PhotoCalib` object from which to calculate the
        photometric calibration scaling at the (pixel) positions in
        ``xArray`` and ``yArray``.
    xArray, yArray : `numpy.ndarray`
        1D arrays of (x, y) (pixel) positions at which to compute the
        photometric calibration scale value.

    Raises
    ------
    RuntimeError
        If lengths of ``xArray`` and ``yArray`` are not equal.

    Returns
    -------
    photoCalibScaleArray : `numpy.ndarray`
        1D array of same length as ``xArray`` and ``yArray`` consiting of
        the photometric calibration scale values associated with the
        (``xArray``, ``yArray``) positions.
    """
    if len(xArray) != len(yArray):
        raise RuntimeError("Lengths of xArray (N={}) and yArray (N={}) must be equal".
                           format(len(xArray), len(yArray)))
    boundedField = photoCalib.computeScaledCalibration()
    # The following represents the spatially constant component of the
    # calibration.
    photoCalibScaleArray = np.full_like(xArray, photoCalib.getCalibrationMean())
    photoCalibScaleErrArray = np.full_like(xArray, photoCalib.getCalibrationErr())
    # The following will be true for a variable photoCalib
    if boundedField.getBBox().getArea() > 0:
        variableScaleArray = boundedField.evaluate(xArray, yArray)
        photoCalibScaleArray *= variableScaleArray
        # TODO: decide if this scaling should be included for proper error
        # propagation.  Leaving commented out for now to mimic what is
        # currently implemented in the methods on the photoCalib object.
        # photoCalibScaleErrArray *= variableScaleArray
    return photoCalibScaleArray, photoCalibScaleErrArray


def calibrateSourceCatalog(catalog, zp):
    """Calibrate catalog to zeropoint given.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The source catalog under consideration.
    zp : `float`
        The zeropoint value to calibrate the fluxes to.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The calibrated ``catalog``.
    """
    # Convert to constant zero point, as for the coadds
    schema = getSchema(catalog)
    fluxKeys, errKeys = getFluxKeys(schema)
    factor = 10.0**(0.4*zp)
    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
        catalog[key] /= factor
    return catalog


def backoutApCorr(catalog):
    """Back out the aperture correction to all fluxes.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The source catalog under consideration.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The ``catalog`` whose flux entries have had their aperture corrections
        backed out.
    """
    ii = 0
    fluxStr = "_instFlux"
    apCorrStr = "_apCorr"
    schema = getSchema(catalog)
    if isinstance(catalog, pd.DataFrame):
        keys = {flux: flux for flux in schema if (flux.endswith(fluxStr) or flux.endswith(apCorrStr))}
    else:
        keys = schema.getNames()
    for k in keys:
        if fluxStr in k and k[:-len(fluxStr)] + apCorrStr in keys and apCorrStr not in k:
            if ii == 0:
                ii += 1
            catalog[k] /= catalog[k[:-len(fluxStr)] + apCorrStr]
    return catalog


def matchNanojanskyToAB(matches):
    """Convert catalog fluxes with unit "nanojanskys" to AB.

    Using astropy units for conversion for consistency with PhotoCalib.

    Parameters
    ----------
    matches : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The match catalog under consideration.

    Returns
    -------
    matches : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The ``matches`` catalog whose flux entries have had their aperture
        corrections backed out.
    """
    if isinstance(matches, pd.DataFrame):
        schema = getSchema(matches)
        keys = [kk for kk in schema if kk.startswith("ref_") and "_flux" in kk]
        for k in keys:
            matches[k] = matches[k].apply(lambda x: x/NANOJANSKYS_PER_AB_FLUX)
    else:
        schema = matches[0].first.schema
        keys = [schema[kk].asKey() for kk in schema.getNames() if "_flux" in kk]
        for m in matches:
            for k in keys:
                m.first[k] /= NANOJANSKYS_PER_AB_FLUX
    return matches


def checkHscStack(metadata):
    """Check to see if data were processed with the HSC stack.
    """
    try:
        hscPipe = metadata.getScalar("HSCPIPE_VERSION")
    except Exception:
        hscPipe = None
    return hscPipe


def fluxToPlotString(fluxToPlot):
    """Return a more succint string for fluxes for label plotting.
    """
    fluxStrMap = {"base_PsfFlux_instFlux": "PSF",
                  "base_PsfFlux_flux": "PSF",
                  "base_PsfFlux": "PSF",
                  "base_GaussianFlux": "Gaussian",
                  "ext_photometryKron_KronFlux": "Kron",
                  "base_InputCount_value": "InputCount",
                  "modelfit_CModel_instFlux": "CModel",
                  "modelfit_CModel_flux": "CModel",
                  "modelfit_CModel": "CModel",
                  "base_CircularApertureFlux_12_0_instFlux": "CircApRad12pix",
                  "base_CircularApertureFlux_12_0": "CircApRad12pix",
                  "base_CircularApertureFlux_9_0_instFlux": "CircApRad9pix",
                  "base_CircularApertureFlux_9_0": "CircApRad9pix",
                  "base_CircularApertureFlux_25_0_instFlux": "CircApRad25pix",
                  "base_CircularApertureFlux_25_0": "CircApRad25pix"}
    if fluxToPlot in fluxStrMap:
        return fluxStrMap[fluxToPlot]
    else:
        print("WARNING: " + fluxToPlot + " not in fluxStrMap")
        return fluxToPlot


_eups = None


def getEups():
    """Return a EUPS handle.

    We instantiate this once only, because instantiation is expensive.
    """
    global _eups
    from eups import Eups  # noqa Nothing else depends on eups, so prevent it from importing unless needed
    if not _eups:
        _eups = Eups()
    return _eups


@contextmanager
def andCatalog(version):
    eups = getEups()
    current = eups.findSetupVersion("astrometry_net_data")[0]
    eups.setup("astrometry_net_data", version, noRecursion=True)
    try:
        yield
    finally:
        eups.setup("astrometry_net_data", current, noRecursion=True)


def getRepoInfo(dataRef, coaddName=None, coaddDataset=None, catDataset="src", doApplyExternalPhotoCalib=False,
                externalPhotoCalibName="jointcal", doApplyExternalSkyWcs=False,
                externalSkyWcsName="jointcal"):
    """Obtain the relevant repository information for the given dataRef.

    Parameters
    ----------
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        The data reference for which the relevant repository information
        is to be retrieved.
    coaddName : `str`, optional
        The base name of the coadd (e.g. "deep" or "goodSeeing") if
        ``dataRef`` is for coadd level processing.
    coaddDataset : `str`, optional
        The name of the coadd dataset (e.g. "Coadd_forced_src" or
        "Coadd_meas") if ``dataRef`` is for coadd level processing.
    doApplyUberCal : `bool`, optional
        If `True`: Set the appropriate dataset type for the uber
        calibration from meas_mosaic.
        If `False`: Set the dataset type to the source catalog from single
        frame processing.

    Raises
    ------
    RuntimeError
        If one of ``coaddName`` or ``coaddDataset`` is specified but
        the other is not.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Result struct with components:

        - ``butler`` : the butler associated with ``dataRef``
          (`lsst.daf.persistence.Butler`).
        - ``camera`` : the camera associated with ``butler``
          (`lsst.afw.cameraGeom.Camera`).
        - ``dataId`` : the dataId associated with ``dataRef``
          (`lsst.daf.persistence.DataId`).
        - ``filterName`` : the name of the filter associated with ``dataRef``
          (`str`).
        - ``genericBandName`` : the generic band name for ``filterName``
          (`str`).
        - ``ccdKey`` : the ccd key associated with ``dataId`` (`str` or
          `None`).
        - ``metadata`` : the metadata associated with ``butler`` and ``dataId``
          (`lsst.daf.base.propertyContainer.PropertyList`).
        - ``hscRun`` : string representing "HSCPIPE_VERSION" fits header if
          the data associated with ``dataRef``'s ``dataset`` were processed
          with the (now obsolete, but old reruns still exist) "HSC stack",
          `None` otherwise (`str` or `None`).
        - ``dataset`` : the dataset name ("src" if ``dataRef`` is visit level,
          coaddName + coaddDataset if ``dataRef`` is a coadd (`str`)
        - ``skyMap`` : the sky map associated with ``dataRef`` if it is a
          coadd (`lsst.skymap.SkyMap` or `None`).
        - ``wcs`` : the wcs of the coadd image associated with ``dataRef``
          -- only needed as a workaround for some old coadd catalogs that were
          persisted with all nan for ra dec (`lsst.afw.geom.SkyWcs` or `None`).
        - ``tractInfo`` : the tract information associated with ``dataRef`` if
          it is a coadd (`lsst.skymap.tractInfo.ExplicitTractInfo` or `None`).
    """
    if coaddName and not coaddDataset or not coaddName and coaddDataset:
        raise RuntimeError("If one of coaddName or coaddDataset is specified, the other must be as well.")

    if hasattr(dataRef, "getButler"):
        butler = dataRef.getButler()
        camera = butler.get("camera")
        dataId = dataRef.dataId.copy()
        delimiterStr = "_"
        isGen3 = False
    else:
        butler = dataRef["butler"]
        cameraName = dataRef["camera"]
        if cameraName == "HSC":
            from lsst.obs.subaru import HyperSuprimeCam
            instrument = HyperSuprimeCam()
        elif cameraName == "LSSTCam-imSim":
            from lsst.obs.lsst.imsim import LsstCamImSim
            instrument = LsstCamImSim()
        else:
            raise RuntimeError("Unknown camera {}".format(cameraName))
        camera = instrument.getCamera()
        dataId = dataRef["dataId"].copy()
        delimiterStr = "."
        isGen3 = True
    isCoadd = True if "patch" in dataId else False
    try:
        if isGen3 and isCoadd:
            filterName = dataId["band"]
        else:
            filterName = dataId["filter"]
    except KeyError:
        if isCoadd:
            filterName = butler.get(coaddName + "Coadd_calexp_filter", dataId)
        else:
            filterName = butler.get("calexp_filter", dataId)
        dataId["filter"] = filterName
    if isCoadd:
        if not isGen3:
            bbox = butler.get(coaddName + "Coadd_bbox", dataId)
            subBbox = geom.BoxI(geom.PointI(bbox.beginX, bbox.beginY), geom.ExtentI(1, 1))
            expInfo = butler.get(coaddName + "Coadd_calexp_sub", dataId=dataId, bbox=subBbox).getInfo()
    else:
        bbox = butler.get("calexp" + delimiterStr + "bbox", dataId)
        subBbox = geom.BoxI(geom.PointI(bbox.beginX, bbox.beginY), geom.ExtentI(1, 1))
        if isGen3:
            expInfo = butler.get("calexp", dataId=dataId, parameters={'bbox': subBbox}).getInfo()
        else:
            expInfo = butler.get("calexp" + delimiterStr + "sub", dataId=dataId, bbox=subBbox).getInfo()
    if isGen3:
        ccdKey = "detector"
        genericBandName = filterName
    else:
        ccdKey = None if isCoadd else findCcdKey(dataId)
        genericBandName = expInfo.getFilterLabel().bandLabel

    try:  # Check metadata to see if stack used was HSC
        metaStr = coaddName + coaddDataset + "_md" if coaddName else "calexp_md"
        metadata = butler.get(metaStr, dataId)
        hscRun = checkHscStack(metadata)
    except (AttributeError, KeyError):
        metadata = None
        hscRun = None

    if isGen3:
        try:
            skymap = butler.get("skyMap")
        except Exception:
            skymap = None
    else:
        tempCoaddName = "deep" if coaddName is None else coaddName
        try:
            skymap = butler.get(tempCoaddName + "Coadd_skyMap")
        except Exception:
            skymap = None
    wcs = None
    tractInfo = None
    if isCoadd:
        # To get the coadd's WCS
        coaddImageName = "Coadd_calexp_hsc" if hscRun else "Coadd_calexp"
        wcs = butler.get(coaddName + coaddImageName + delimiterStr + "wcs", dataId)
        catDataset = coaddName + coaddDataset
        tractInfo = skymap[dataId["tract"]]
    photoCalibDataset = None
    if doApplyExternalPhotoCalib:
        if isGen3:
            photoCalibDataset = externalPhotoCalibName + "PhotoCalibCatalog"
        else:
            photoCalibDataset = "fcr_hsc" if hscRun else externalPhotoCalibName + "_photoCalib"
    skyWcsDataset = None
    if doApplyExternalSkyWcs:
        if isGen3:
            skyWcsDataset = externalSkyWcsName + "SkyWcsCatalog"
            skymap = skymap if skymap else butler.get("skyMap")
        else:
            skyWcsDataset = "wcs_hsc" if hscRun else externalSkyWcsName + "_wcs"
            skymap = skymap if skymap else butler.get("deepCoadd_skyMap")
    try:
        tractInfo = skymap[dataId["tract"]]
    except KeyError:
        tractInfo = None

    return Struct(
        butler=butler,
        camera=camera,
        dataId=dataId,
        filterName=filterName,
        genericBandName=genericBandName,
        ccdKey=ccdKey,
        metadata=metadata,
        hscRun=hscRun,
        catDataset=catDataset,
        photoCalibDataset=photoCalibDataset,
        skyWcsDataset=skyWcsDataset,
        skymap=skymap,
        wcs=wcs,
        tractInfo=tractInfo,
        delimiterStr=delimiterStr,
        isGen3=isGen3
    )


def findCcdKey(dataId):
    """Determine the convention for identifying a "ccd" for the current camera.

    Parameters
    ----------
    dataId : `instance` of `lsst.daf.persistence.DataId`

    Raises
    ------
    RuntimeError
        If "ccd" key could not be identified from the current hardwired list.

    Returns
    -------
    ccdKey : `str`
        The string associated with the "ccd" key.
    """
    ccdKey = None
    ccdKeyList = ["ccd", "sensor", "camcol", "detector", "ccdnum"]
    for ss in ccdKeyList:
        if ss in dataId:
            ccdKey = ss
            break
    if ccdKey is None:
        raise RuntimeError("Could not identify ccd key for dataId: %s: \nNot in list of known keys: %s" %
                           (dataId, ccdKeyList))
    return ccdKey


def getCcdNameRefList(dataRefList):
    ccdNameRefList = None
    ccdKey = findCcdKey(dataRefList[0].dataId)
    if "raft" in dataRefList[0].dataId:
        ccdNameRefList = [re.sub("[,]", "", str(dataRef.dataId["raft"]) + str(dataRef.dataId[ccdKey])) for
                          dataRef in dataRefList]
    else:
        ccdNameRefList = [dataRef.dataId[ccdKey] for dataRef in dataRefList]
    # cull multiple entries
    ccdNameRefList = list(set(ccdNameRefList))

    if ccdNameRefList is None:
        raise RuntimeError("Failed to create ccdNameRefList")
    return ccdNameRefList


def getDataExistsRefList(dataRefList, dataset):
    dataExistsRefList = []
    dataExistsCcdList = []
    dataRefTemp = dataRefList[0]
    isGen3 = not hasattr(dataRefTemp, "datasetExists")
    dataIdTemp = dataRefTemp["dataId"] if isGen3 else dataRefTemp.dataId
    ccdKey = findCcdKey(dataIdTemp)
    if not isGen3:
        dataExistsRefList = [dataRef for dataRef in dataRefList if dataRef.datasetExists(dataset)]
        for dataRef in dataRefList:
            if "raft" in dataRef.dataId:
                dataExistsCcdList = [
                    re.sub("[,]", "", str(dataRef.dataId["raft"]) + str(dataRef.dataId[ccdKey])) for
                    dataRef in dataRefList if dataRef.datasetExists(dataset)]
            else:
                dataExistsCcdList = [
                    dataRef.dataId[ccdKey] for dataRef in dataRefList if dataRef.datasetExists(dataset)]
        # cull multiple entries
        dataExistsRefList = list(set(dataExistsRefList))
        dataExistsCcdList = list(set(dataExistsCcdList))
    else:
        for dataRef in dataRefList:
            dataId = dataRef["dataId"]
            try:
                dataRef["butler"].getURI(dataset, dataId=dataId)
                dataExistsRefList.append(dataRef)
                dataExistsCcdList.append(dataRef["dataId"]["detector"])
            except LookupError:
                if dataId["detector"] != 999:
                    print("Could not find {} dataset for dataId {}".format(dataset, dataId))
                continue

    if len(dataExistsRefList) == 0:
        raise RuntimeError("dataExistsRef list is empty")
    return dataExistsRefList, dataExistsCcdList


def fLinear(p, x):
    return p[0] + p[1]*x


def fQuadratic(p, x):
    return p[0] + p[1]*x + p[2]*x**2


def fCubic(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3


def orthogonalRegression(x, y, order, initialGuess=None):
    """Perform an Orthogonal Distance Regression on the given data.

    Parameters
    ----------
    x, y : `array`
        Arrays of x and y data to fit.
    order : `int`
        Order of the polynomial to fit.
    initialGuess : `list` of `float`, optional
        List of the polynomial coefficients (highest power first) of an initial
        guess to feed to the ODR fit.  If no initialGuess is provided, a simple
        linear fit is performed and used as the guess.

    Raises
    ------
    RuntimeError
        If ``order`` is not between 1 and 3.

    Returns
    -------
    result : `list` of `float`
        List of the fit coefficients (highest power first to mimic
        `numpy.polyfit` return).
    """
    if initialGuess is None:
        linReg = scipyStats.linregress(x, y)
        initialGuess = [linReg[0], linReg[1]]
        for i in range(order - 1):  # initialGuess here is linear, so need to pad array to match order
            initialGuess.insert(0, 0.0)
    if order == 1:
        odrModel = scipyOdr.Model(fLinear)
    elif order == 2:
        odrModel = scipyOdr.Model(fQuadratic)
    elif order == 3:
        odrModel = scipyOdr.Model(fCubic)
    else:
        raise RuntimeError("Order must be between 1 and 3 (value requested, {:}, not accommodated)".
                           format(order))
    odrData = scipyOdr.Data(x, y)
    orthDist = scipyOdr.ODR(odrData, odrModel, beta0=initialGuess)
    orthRegFit = orthDist.run()

    return list(reversed(orthRegFit.beta))


def distanceSquaredToPoly(x1, y1, x2, poly):
    """Calculate the square of the distance between point (x1, y1) and poly
    at x2.

    Parameters
    ----------
    x1, y1 : `float`
        Point from which to calculate the square of the distance to the
        polynomial ``poly``.
    x2 : `float`
        Position on x axis from which to calculate the square of the distance
        between (``x1``, ``y1``) and ``poly`` (the position of the tangent of
        the polynomial curve closest to point (``x1``, ``y1``)).
    poly : `numpy.lib.polynomial.poly1d`
        Numpy polynomial fit from which to calculate the square of the distance
        to (``x1``, ``y1``) at ``x2``.

    Returns
    -------
    result : `float`
        Square of the distance between (``x1``, ``y1``) and ``poly`` at ``x2``
    """
    return (x2 - x1)**2 + (poly(x2) - y1)**2


def p1CoeffsFromP2x0y0(p2Coeffs, x0, y0):
    """Compute Ivezic P1 coefficients using the P2 coeffs and origin (x0, y0).

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    theta = arctan(mP1), where mP1 is the slope of the equivalent straight
                         line (the P1 line) from the P2 coeffs in the
                         (x, y) coordinate system and
                         x = c1 - c2, y = c2 - c3
    P1 = cos(theta)*c1
         + ((sin(theta) - cos(theta))*c2 - sin(theta)*c3
         + deltaP1
    P1 = 0 at x0, y0 ==> deltaP1 = -cos(theta)*x0 - sin(theta)*y0

    Parameters
    ----------
    p2Coeffs : `list` of `float`
        List of the four P2 coefficients from which, along with the origin
        point (``x0``, ``y0``), to compute/derive the associated P1
        coefficients.
    x0, y0 : `float`
        Coordinates at which to set P1 = 0 (i.e. the P1/P2 axis origin).

    Returns
    -------
    p1Coeffs: `list` of `float`
        The four P1 coefficients.
    """
    mP1 = p2Coeffs[0]/p2Coeffs[2]
    cosTheta = np.cos(np.arctan(mP1))
    sinTheta = np.sin(np.arctan(mP1))
    deltaP1 = -cosTheta*x0 - sinTheta*y0
    p1Coeffs = [cosTheta, sinTheta - cosTheta, -sinTheta, deltaP1]

    return p1Coeffs


def p2p1CoeffsFromLinearFit(m, b, x0, y0):
    """Derive the Ivezic et al. 2004 P2 and P1 equations based on linear fit.

    Where the linear fit is to the given region in color-color space.
    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    For y = m*x + b fit, where x = c1 - c2 and y = c2 - c3,
    P2 = (-m*c1 + (m + 1)*c2 - c3 - b)/sqrt(m**2 + 1)
    P2norm = P2/sqrt[(m**2 + (m + 1)**2 + 1**2)/(m**2 + 1)]

    P1 = cos(theta)*x + sin(theta)*y + deltaP1, theta = arctan(m)
    P1 = cos(theta)*(c1 - c2) + sin(theta)*(c2 - c3) + deltaP1
    P1 = cos(theta)*c1
         + ((sin(theta) - cos(theta))*c2 - sin(theta)*c3 + deltaP1
    P1 = 0 at x0, y0 ==> deltaP1 = -cos(theta)*x0 - sin(theta)*y0

    Parameters
    ----------
    m : `float`
        Slope of line to convert.
    b : `float`
        Intercept of line to convert.
    x0, y0 : `float`
        Coordinates at which to set P1 = 0.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Result struct with components:

        - ``p2Coeffs`` : four P2 equation coefficents (`list` of `float`).
        - ``p1Coeffs`` : four P1 equation coefficents (`list` of `float`).
    """
    # Compute Ivezic P2 coefficients using the linear fit slope and intercept
    scaleFact = np.sqrt(m**2 + 1.0)
    p2Coeffs = [-m/scaleFact, (m + 1.0)/scaleFact, -1.0/scaleFact, -b/scaleFact]
    p2Norm = 0.0
    for coeff in p2Coeffs[:-1]:  # Omit the constant normalization term
        p2Norm += coeff**2
    p2Norm = np.sqrt(p2Norm)
    p2Coeffs /= p2Norm

    # Compute Ivezic P1 coefficients equation using the linear fit slope and
    # point (x0, y0) as the origin
    p1Coeffs = p1CoeffsFromP2x0y0(p2Coeffs, x0, y0)

    return Struct(
        p2Coeffs=p2Coeffs,
        p1Coeffs=p1Coeffs,
    )


def lineFromP2Coeffs(p2Coeffs):
    """Compute P1 line in color-color space for given set P2 coefficients.

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    Parameters
    ----------
    p2Coeffs : `list` of `float`
        List of the four P2 coefficients.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Result struct with components:

        - ``mP1`` : associated slope for P1 in color-color coordinates
                   (`float`).
        - ``bP1`` : associated intercept for P1 in color-color coordinates
                   (`float`).
    """
    mP1 = p2Coeffs[0]/p2Coeffs[2]
    bP1 = -p2Coeffs[3]*np.sqrt(mP1**2 + (mP1 + 1.0)**2 + 1.0)
    return Struct(
        mP1=mP1,
        bP1=bP1,
    )


def linesFromP2P1Coeffs(p2Coeffs, p1Coeffs):
    """Derive P1/P2 axes in color-color space based on the P2 and P1 coeffs.

    Reference: Ivezic et al. 2004 (2004AN....325..583I)

    Parameters
    ----------
    p2Coeffs : `list` of `float`
        List of the four P2 coefficients.
    p1Coeffs : `list` of `float`
        List of the four P1 coefficients.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Result struct with components:

        - ``mP2``, ``mP1`` : associated slopes for P2 and P1 in color-color
                             coordinates (`float`).
        - ``bP2``, ``bP1`` : associated intercepts for P2 and P1 in color-color
                             coordinates (`float`).
        - ``x0``, ``y0`` : x and y coordinates of the P2/P1 axes origin in
                           color-color coordinates (`float`).
    """
    p1Line = lineFromP2Coeffs(p2Coeffs)
    mP1 = p1Line.mP1
    bP1 = p1Line.bP1

    cosTheta = np.cos(np.arctan(mP1))
    sinTheta = np.sin(np.arctan(mP1))

    def func2(x):
        y = [cosTheta*x[0] + sinTheta*x[1] + p1Coeffs[3], mP1*x[0] - x[1] + bP1]
        return y

    x0y0 = scipyOptimize.fsolve(func2, [1, 1])
    mP2 = -1.0/mP1
    bP2 = x0y0[1] - mP2*x0y0[0]
    return Struct(
        mP2=mP2,
        bP2=bP2,
        mP1=mP1,
        bP1=bP1,
        x0=x0y0[0],
        y0=x0y0[1],
    )


def makeEqnStr(varName, coeffList, exponentList):
    """Make a string-formatted equation.

    Parameters
    ----------
    varName : `str`
        Name of the equation to be stringified.
    coeffList : `list` of `float`
        List of equation coefficients (matched to coefficients in
        ``exponentList``).
    exponentList : `list` of `str`
        List of equation exponents (matched to coefficients in
        ``coeffList``).

    Raises
    ------
    RuntimeError
        If lengths of ``coeffList`` and ``exponentList`` are not equal.

    Returns
    -------
    eqnStr : `str`
        The stringified equation of the form:
        varName = coeffList[0]exponent[0] + ... + coeffList[n-1]exponent[n-1].
    """
    if len(coeffList) != len(exponentList):
        raise RuntimeError("Lengths of coeffList ({0:d}) and exponentList ({1:d}) are not equal".
                           format(len(coeffList), len(exponentList)))

    eqnStr = varName + " = "
    for i, (coeff, band) in enumerate(zip(coeffList, exponentList)):
        coeffStr = "{:.3f}".format(abs(coeff)) + band
        plusMinus = " $-$ " if coeff < 0.0 else " + "
        if i == 0:
            eqnStr += plusMinus.strip(" ") + coeffStr
        else:
            eqnStr += plusMinus + coeffStr

    return eqnStr


def catColors(c1, c2, magsCat, goodArray=None):
    """Compute color for a set of filters given a catalog of magnitudes by
    filter.

    Parameters
    ----------
    c1, c2 : `str`
        String representation of the filters from which to compute the color.
    magsCat : `dict` of `numpy.ndarray`
        Dict of arrays of magnitude values.  Dict keys are the string
        representation of the filters.
    goodArray : `numpy.ndarray`, optional
        Boolean array with same length as the magsCat arrays whose values
        indicate whether the source was deemed "good" for intended use.  If
        `None`, all entries are considered "good".

    Raises
    ------
    RuntimeError
        If lengths of ``goodArray`` and ``magsCat`` arrays are not equal.

    Returns
    -------
    `numpy.ndarray` of "good" colors (magnitude differeces).
    """
    if goodArray is None:
        goodArray = np.ones(len(magsCat[c1]), dtype=bool)

    if len(goodArray) != len(magsCat[c1]):
        raise RuntimeError("Lengths of goodArray ({0:d}) and magsCat ({1:d}) are not equal".
                           format(len(goodArray), len(magsCat[c1])))

    return (magsCat[c1] - magsCat[c2])[goodArray]


def setAliasMaps(catalog, aliasDictList, prefix=""):
    """Set an alias map for differing schema naming conventions.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which the mapping will be added.
    aliasDictList : `dict` of `str` or `list` of `dict` of `str`
        A `list` of `dict` or single `dict` representing the alias mappings to
        be added to ``catalog``'s schema with the key representing the new
        name to be mapped to the value which represents the old name.  Note
        that the mapping will only be added if the old name exists in
        ``catalog``'s schema.
    prefix : `str`, optional
        This `str` will be prepended to the alias names (used, e.g., in matched
        catalogs for which "src_" and "ref_" prefixes have been added to all
        schema names).  Both the old and new names have ``prefix`` associated
        with them.

    Raises
    ------
    RuntimeError
        If not all elements in ``aliasDictList`` are instances of type `dict`
        or `lsst.pex.config.dictField.Dict`.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog with the alias mappings added to the schema.
    """
    if isinstance(aliasDictList, dict):
        aliasDictList = [aliasDictList, ]
    if not all(isinstance(aliasDict, (dict, pexConfig.dictField.Dict)) for aliasDict in aliasDictList):
        raise RuntimeError("All elements in aliasDictList must be instances of type dict")
    schema = getSchema(catalog)
    aliasMap = schema.getAliasMap()
    for aliasDict in aliasDictList:
        for newName, oldName in aliasDict.items():
            if prefix + oldName in schema:
                aliasMap.set(prefix + newName, prefix + oldName)
    return catalog


def addAliasColumns(catalog, aliasDictList, prefix=""):
    """Copy columns from an alias map for differing schema naming conventions.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which the mapping columns will be added.
    aliasDictList : `dict` of `str` or `list` of `dict` of `str`
        A `list` of `dict` or single `dict` representing the column "mappings"
        to be added to ``catalog``'s schema, i.e. a new column with the "new"
        name will by added to the catalog which is simply a duplicate of the
        "old" name column.  Note that the new column will only be added if
        the column with the old name exists in ``catalog``'s schema.
    prefix : `str`, optional
        This `str` will be prepended to the alias names (used, e.g., in matched
        catalogs for which "src_" and "ref_" prefixes have been added to all
        schema names).  Both the old and new names have ``prefix`` associated
        with them.

    Raises
    ------
    RuntimeError
        If not all elements in ``aliasDictList`` are instances of type `dict`
        or `lsst.pex.config.dictField.Dict`.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        The source ``catalog`` with the "alias" columns added to the schema.
    """
    if isinstance(aliasDictList, dict):
        aliasDictList = [aliasDictList, ]
    if not all(isinstance(aliasDict, (dict, pexConfig.dictField.Dict)) for aliasDict in aliasDictList):
        raise RuntimeError("All elements in aliasDictList must be instances of type dict")
    for aliasDict in aliasDictList:
        for newName, oldName in aliasDict.items():
            if oldName in catalog.schema and newName not in catalog.schema:
                fieldDoc = catalog.schema[oldName].asField().getDoc()
                fieldUnits = catalog.schema[oldName].asField().getUnits()
                fieldType = catalog.schema[oldName].asField().getTypeString()
                if fieldType == "Flag":
                    catalog = addFlag(catalog, catalog[oldName], newName, doc=fieldDoc)
                else:
                    catalog = addIntFloatOrStrColumn(catalog, catalog[oldName], newName, fieldDoc,
                                                     fieldUnits=fieldUnits)
    return catalog


def addPreComputedColumns(catalog, fluxToPlotList, toMilli=False, unforcedCat=None):
    """Add column entries for a set of pre-computed values.

    This is for the parquet tables to facilitate the interactive
    drilldown analyses.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog to which the columns will be added.
    fluxToPlotList : `list`
        List of flux field names to make mag(fluxName) - mag(PsfFlux) columns.
    toMilli : `bool`, optional
        Whether to use units of "milli" (e.g. mmag, mas).
    unforcedCat : `lsst.afw.table.SourceCatalog`, optional
        If `catalog` is a coadd forced catalog, this is its associated unforced
        catalog for direct comparison of forced vs. unforced parameters.

    Returns
    -------
    catalog : `lsst.afw.table.SourceCatalog`
        The source catalog with the columns of pre-computed values added.
    """
    unitScale = 1000.0 if toMilli else 1.0
    fieldUnits = " (mmag)" if toMilli else " (mag)"
    schema = getSchema(catalog)
    for col in fluxToPlotList:
        colStr = fluxToPlotString(col)
        if col + "_instFlux" in schema:
            compCol = "base_PsfFlux"
            compColStr = fluxToPlotString(compCol)
            fieldName = colStr + "-" + compColStr + "_magDiff_" + fieldUnits.strip(" ()")
            fieldDoc = "Magnitude difference: " + colStr + "-" + compCol + fieldUnits
            parameterFunc = MagDiff(col + "_instFlux", compCol + "_instFlux", unitScale=unitScale)
            magDiff = parameterFunc(catalog)
            parameterFunc = MagDiffErr(col + "_instFlux", compCol + "_instFlux", unitScale=unitScale)
            magDiffErr = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, magDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))
            fieldErrName = colStr + "-" + compColStr + "_magDiffErr_" + fieldUnits.strip(" ()")
            fieldErrDoc = "Error of: " + fieldDoc
            catalog = addIntFloatOrStrColumn(catalog, magDiffErr, fieldErrName, fieldErrDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

    for compareCol, psfCompareCol, compareStr in [["base_SdssShape", "base_SdssShape_psf", "Sdss"],
                                                  ["ext_shapeHSM_HsmSourceMoments",
                                                   "ext_shapeHSM_HsmPsfMoments", "Hsm"]]:
        if compareCol + "_xx" in schema:
            # Source Trace
            fieldUnits = " (pixel)"
            fieldName = "trace" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = sqrt(0.5*(Ixx + Iyy))" + fieldUnits
            parameterFunc = TraceSize(compareCol)
            traceSize = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, traceSize, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))
            fieldName = "trace" + compareStr + "_fwhm_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = 2.0*np.sqrt(2.0*np.log(2.0))*Trace" + fieldUnits
            fwhmSize = 2.0*np.sqrt(2.0*np.log(2.0))*traceSize
            catalog = addIntFloatOrStrColumn(catalog, fwhmSize, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

            # Source Trace - PSF model Trace
            fieldUnits = " (%)"
            fieldName = "psfTrace" + compareStr + "Diff_percent"
            fieldDoc = fieldName + " = srcTrace" + compareStr + " - psfModelTrace" + compareStr + fieldUnits
            parameterFunc = PsfTraceSizeDiff(compareCol, psfCompareCol)
            psfTraceSizeDiff = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, psfTraceSizeDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

            # Source - PSF model E1/E2 Residuals
            fieldUnits = " (milli)" if toMilli else ""
            fieldName = "e1Resids" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = src(e1) - psfModel(e1), e1 = (Ixx - Iyy)/(Ixx + Iyy)" + fieldUnits
            parameterFunc = E1Resids(compareCol, psfCompareCol, unitScale)
            e1Resids = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, e1Resids, fieldName, fieldDoc)
            fieldName = "e2Resids" + compareStr + "_" + fieldUnits.strip(" ()")
            fieldDoc = fieldName + " = src(e2) - psfModel(e2), e2 = 2*Ixy/(Ixx + Iyy)" + fieldUnits
            parameterFunc = E2Resids(compareCol, psfCompareCol, unitScale)
            e2Resids = parameterFunc(catalog)
            catalog = addIntFloatOrStrColumn(catalog, e2Resids, fieldName, fieldDoc)

    if "base_SdssShape_xx" in schema:
        fieldName = "deconvMoments"
        fieldDoc = "Deconvolved moments"
        deconvMoments = deconvMom(catalog)
        catalog = addIntFloatOrStrColumn(catalog, deconvMoments, fieldName, fieldDoc)

    if unforcedCat is not None:
        fieldUnits = " (mmag)" if toMilli else " (mag)"
        for col in fluxToPlotList:
            colStr = fluxToPlotString(col)
            fieldName = "compareUnforced_" + colStr + "_magDiff_" + fieldUnits.strip(" ()")
            fieldDoc = "Compare forced - unforced" + colStr + " (mmag)"
            parameterFunc = MagDiff(col + "_instFlux", col + "_instFlux", unitScale=unitScale)
            magDiff = parameterFunc(catalog, unforcedCat)
            catalog = addIntFloatOrStrColumn(catalog, magDiff, fieldName, fieldDoc,
                                             fieldUnits=fieldUnits.strip(" ()"))

    return catalog


def addMetricMeasurement(job, metricName, metricValue, measExtrasDictList=None):
    """Add a measurement to a lsst.verify.Job instance.

    TODO: this functionality will likely be moved to MeasurementSet in
          lsst.verify per DM-12655.

    Parameters
    ----------
    job : `lsst.verify.Job`
        The verify Job to add the measurement to.
    metricName : `str`
        The name of the metric to be added.
    metricValue : `float`, `int`, `bool`, `str`
        The value of the metric to be added.
    measExtrasDictList : `list` of `dict`, optional
        A dict of key value pairs of any "extras" to be added to the
        metric measurement.  All of the following keys must be provided for
        each `dict` in the `list`: "name", "value", "label", "description".
        The "label" is meant to be a `str` suitable for plot axis labelling.

    Returns
    -------
    job : `lsst.verify.Job`
        The updated ``job`` with the new measurement added.
    """
    meas = verify.Measurement(job.metrics[metricName], metricValue)
    if measExtrasDictList:
        for extra in measExtrasDictList:
            meas.extras[extra["name"]] = verify.Datum(extra["value"], label=extra["label"],
                                                      description=extra["description"], unit="")
    job.measurements.insert(meas)
    return job


def updateVerifyJob(job, metaDict=None, specsList=None):
    """Update an lsst.verify.Job with metadata and specifications.

    Parameters
    ----------
    job : `lsst.verify.Job`
        The verify Job to add the measurement to.
    metaDict : `dict`, optional
        A dict of key value pairs of any metadata to be added to the
        verify `job`.
    specsList : `list` of `lsst.verify.Specification`, optional
        A `list` of valid `lsst.verify.Specifications`s to be added
        verify `job`.

    Returns
    -------
    job : `lsst.verify.Job`
        The updated ``job`` with the new metadata and specifications added.
    """
    if metaDict:
        for metaName, metaValue in metaDict.items():
            job.meta.update({metaName: metaValue})
    if specsList:
        for spec in specsList:
            job.specs.update(spec)
    return job


def computeMeanOfFrac(valueArray, tailStr="upper", fraction=0.1, floorFactor=1):
    """Compute the rounded mean of the upper/lower fraction of the input array.

    In other words, sort ``valueArray`` by value and compute the mean values of
    the highest[lowest] ``fraction`` of points for ``tailStr`` = upper[lower]
    and round this mean to a number of significant digits given by
    ``floorFactor``.

    E.g.
    ``floorFactor`` = 0.001, round to nearest thousandth (657.14727 -> 657.147)
    ``floorFactor`` = 0.01,  round to nearest hundredth (657.14727 -> 657.15)
    ``floorFactor`` = 0.1,   round to nearest tenth     (657.14727 -> 657.1)
    ``floorFactor`` = 1,     round to nearest integer   (657.14727 -> 657.0)
    ``floorFactor`` = 10,    round to nearest ten       (657.14727 -> 660.0)
    ``floorFactor`` = 100,   round to nearest hundred   (657.14727 -> 700.0)

    Parameters
    ----------
    valueArray : `numpy.ndarray`
        The array of values from which to compute the rounded mean.
    taiStr : `str`, optional
        Whether to compute the mean of the upper or lower ``fraction`` of
        points in ``valueArray``.
    fraction : `float`, optional
        The fraction of the upper or lower tail of the sorted values of
        ``valueArray`` to use in the calculation.
    floorFactor : `float`, optional
        Factor of 10 representing the number of significant digits to round to.
        See above for examples.

    Raises
    ------
    RuntimeError
        If ``tailStr`` is not either \"upper\" or \"lower\".

    Returns
    -------
    meanOfFrac : `float`
       The mean of the upper/lower ``fraction`` of the values in
       ``valueArray``.
    """
    valueArray = valueArray.array if hasattr(valueArray, "array") else valueArray
    pad = 0.49
    ptFrac = max(2, int(fraction*len(valueArray)))
    if tailStr == "upper":
        meanOfFrac = np.floor(
            valueArray[valueArray.argsort()[-ptFrac:]].mean()/floorFactor + pad)*floorFactor
    elif tailStr == "lower":
        meanOfFrac = np.floor(
            valueArray[valueArray.argsort()[0:ptFrac]].mean()/floorFactor - pad)*floorFactor
    else:
        raise RuntimeError("tailStr must be either \"upper\" or \"lower\" (" + tailStr
                           + "was provided")

    return meanOfFrac


def calcQuartileClippedStats(dataArray, nSigmaToClip=3.0):
    """Calculate the quartile-based clipped statistics of a data array.

    The difference between quartiles[2] and quartiles[0] is the interquartile
    distance.  0.74*interquartileDistance is an estimate of standard deviation
    so, in the case that ``dataArray`` has an approximately Gaussian
    distribution, this is equivalent to nSigma clipping.

    Parameters
    ----------
    dataArray : `list` or `numpy.ndarray` of `float`
        List or array containing the values for which the quartile-based
        clipped statistics are to be calculated.
    nSigmaToClip : `float`, optional
        Number of \"sigma\" outside of which to clip data when computing the
        statistics.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        The quartile-based clipped statistics with ``nSigmaToClip`` clipping.
        Atributes are:

        ``median``
            The median of the full ``dataArray`` (`float`).
        ``mean``
            The quartile-based clipped mean (`float`).
        ``stdDev``
            The quartile-based clipped standard deviation (`float`).
        ``rms``
            The quartile-based clipped root-mean-squared (`float`).
        ``clipValue``
            The value outside of which to clip the data before computing the
            statistics (`float`).
        ``goodArray``
            A boolean array indicating which data points in ``dataArray`` were
            used in the calculation of the statistics, where `False` indicates
            a clipped datapoint (`numpy.ndarray` of `bool`).
    """
    quartiles = np.percentile(dataArray, [25, 50, 75])
    assert len(quartiles) == 3
    median = quartiles[1]
    interQuartileDistance = quartiles[2] - quartiles[0]
    clipValue = nSigmaToClip*0.74*interQuartileDistance
    good = np.logical_not(np.abs(dataArray - median) > clipValue)
    quartileClippedMean = dataArray[good].mean()
    quartileClippedStdDev = dataArray[good].std()
    quartileClippedRms = np.sqrt(np.mean(dataArray[good]**2))

    return Struct(
        median=median,
        mean=quartileClippedMean,
        stdDev=quartileClippedStdDev,
        rms=quartileClippedRms,
        clipValue=clipValue,
        goodArray=good,
    )


def corrSpin0(ra, dec, k1, k2=None, raUnits="degrees", decUnits="degrees", **treecorrKwargs):
    """Function to compute correlations between at most two scalar fields.

    This is used to compute Rho0 statistics, given the appropriate spin-0
    (scalar) fields, usually fractional size residuals.

    Parameters
    ----------
    ra : `numpy.array`
        The right ascension values of entries in the catalog.
    dec : `numpy.array`
        The declination values of entries in the catalog.
    k1 : `numpy.array`
        The primary scalar field.
    k2 : `numpy.array`, optional
        The secondary scalar field.
        Autocorrelation of the primary field is computed if `None` (default).
    raUnits : `str`, optional
        Unit of the right ascension values.
        Valid options are "degrees", "arcmin", "arcsec", "hours" or "radians".
    decUnits : `str`, optional
        Unit of the declination values.
        Valid options are "degrees", "arcmin", "arcsec", "hours" or "radians".
    **treecorrKwargs
        Keyword arguments to be passed to `treecorr.KKCorrelation`.

    Returns
    -------
    xy : `treecorr.KKCorrelation`
        A `treecorr.KKCorrelation` object containing the correlation function.
    """

    xy = treecorr.KKCorrelation(**treecorrKwargs)
    catA = treecorr.Catalog(ra=ra, dec=dec, k=k1, ra_units=raUnits,
                            dec_units=decUnits)
    if k2 is None:
        # Calculate the auto-correlation
        xy.process(catA)
    else:
        catB = treecorr.Catalog(ra=ra, dec=dec, k=k2, ra_units=raUnits,
                                dec_units=decUnits)
        # Calculate the cross-correlation
        xy.process(catA, catB)

    return xy


def corrSpin2(ra, dec, g1a, g2a, g1b=None, g2b=None, raUnits="degrees", decUnits="degrees", **treecorrKwargs):
    """Function to compute correlations between at most two shear-like fields.

    This is used to compute Rho statistics, given the appropriate spin-2
    (shear-like) fields.

    Parameters
    ----------
    ra : `numpy.array`
        The right ascension values of entries in the catalog.
    dec : `numpy.array`
        The declination values of entries in the catalog.
    g1a : `numpy.array`
        The first component of the primary shear-like field.
    g2a : `numpy.array`
        The second component of the primary shear-like field.
    g1b : `numpy.array`, optional
        The first component of the secondary shear-like field.
        Autocorrelation of the primary field is computed if `None` (default).
    g2b : `numpy.array`, optional
        The second component of the secondary shear-like field.
        Autocorrelation of the primary field is computed if `None` (default).
    raUnits : `str`, optional
        Unit of the right ascension values.
        Valid options are "degrees", "arcmin", "arcsec", "hours" or "radians".
    decUnits : `str`, optional
        Unit of the declination values.
        Valid options are "degrees", "arcmin", "arcsec", "hours" or "radians".
    **treecorrKwargs
        Keyword arguments to be passed to `treecorr.GGCorrelation`.

    Returns
    -------
    xy : `treecorr.GGCorrelation`
        A `treecorr.GGCorrelation` object containing the correlation function.
    """
    xy = treecorr.GGCorrelation(**treecorrKwargs)
    catA = treecorr.Catalog(ra=ra, dec=dec, g1=g1a, g2=g2a, ra_units=raUnits,
                            dec_units=decUnits)
    if g1b is None or g2b is None:
        # Calculate the auto-correlation
        xy.process(catA)
    else:
        catB = treecorr.Catalog(ra=ra, dec=dec, g1=g1b, g2=g2b, ra_units=raUnits,
                                dec_units=decUnits)
        # Calculate the cross-correlation
        xy.process(catA, catB)

    return xy


def measureRhoMetrics(rhoStat, thetaExtremum=1.0, operatorStr="<="):
    """Convert Rho Statistics into a scalar metric.

    Parameters
    ----------
    rhoStat : `treecorr.GGCorrelation` or `treecorr.KKCorrelation` object
        A correlation function corresponding to a Rho statistic.
    thetaExtremum : `float`, optional
        Extremum value of angular scale (in units of arcmin) to consider
        for averaging ``rhoStat``.
    operatorStr : `str`, optional
        An optional comparison operator to specify angular scales smaller than
        or larger than ``thetaExtremum``.
        Allowed values are "<=", ">=", "<" or ">".

    Raises
    ------
    ValueError
        Raised if parameter ``operatorStr`` is not one of the allowed values.

    Returns
    -------
    avgRho : `float`
        The metric to track, defined as the mean values of the data points from
        ``rhoStat`` in the angular range specified by ``thetaExtremum`` and
        ``operatorStr``.
    """
    operations = {"<=": operator.le,
                  ">=": operator.ge,
                  "<": operator.lt,
                  ">": operator.gt}
    try:
        operation = operations[operatorStr]
    except KeyError:
        message = "{0!r} is not a supported operator".format(operatorStr)
        raise ValueError(message)

    w, = np.where(operation(rhoStat.meanr, thetaExtremum))
    try:
        xi = rhoStat.xip  # for Rho 1-5
    except AttributeError:
        xi = rhoStat.xi  # for Rho 0
    avgRho = abs(np.average(xi[w]))
    return avgRho


def getSchema(catalog):
    """Helper function to determine "schema" of catalog.

    This will be the list of columns if the catalog type is a pandas DataFrame,
    or the schema object if it is an `lsst.afw.table.SourceCatalog`.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog` or `pandas.core.frame.DataFrame`
        The source catalog under consideration.

    Returns
    -------
    schema : `list` of `str` or `lsst.afw.table.Schema`
        The schema associated with the ``catalog``.
    """
    if isinstance(catalog, pd.DataFrame):
        schema = catalog.columns
        if isinstance(schema, pd.Index):
            schema = schema.tolist()
    elif isinstance(catalog, pd.Series):
        schema = [catalog.name, ]
    else:
        schema = catalog.schema
    return schema


def loadRefCat(matchMeta, refObjLoader, padRadiusFactor=1.05, epoch=None):
    """Function to load a reference catalog based on search coordinates
    stored in the meta data provided.

    During single frame processing calibration, a list of reference-to-source
    matches list is persisted as a "normalized" catalog of IDs produced by
    afw.table.packMatches(), with match metadata (as returned by the astrometry
    tasks) in the catalog's metadata attribute.  This method uses the
    information in the metadata to load in the reference sources covering the
    entire area that was considered in the calibration.

    When loading in reference objects for a region defined by an image's
    current WCS estimate, the reference object loader grows the bbox by the
    config parameter pixelMargin.  This is set to 300 by default but this is
    not reflected by the radius parameter set in the metadata, so some matches
    may reside outside the circle searched within this radius.  Thus, we
    increase the radius set in the metadata by the factor ``padRadiusFactor``
    to accommodate.

    Parameters
    ----------
    matchMeta : `lsst.daf.base.propertyContainer.PropertyList`
        Metadata providing the search center and radius parameters.
    refObjLoader :
           `lsst.meas.algorithms.loadReferenceObjects.LoadReferenceObjectsTask`
        Reference object loader to read in the reference catalogs.
    padRadiusFactor : `float`, optional
        Factor by which to "pad" (increase) the sky circle radius to be loaded
        from that stored in the metadata.
    epoch : `astropy.time.Time`, optional
        Epoch in MJD to which to correct the reference catalog coordinates for
        proper motions, or `None`.  If `None`, an attempt will be made to
        derive the epoch to use from `matchMeta`.  If none is found, `epoch`
        will default to `None` and no corrections will be applied.  Note that
        even if an epoch is provided, corrections can only be made for
        reference catalogs that have proper motions (this is currently only
        true for the Gaia reference catalogs of those that are in regular use
        with LSST pipelines processing).

    Returns
    -------
    refCat : `lsst.afw.table.SimpleCatalog`
        The loaded catalog of reference sources.
    """
    version = matchMeta.getInt("SMATCHV")
    if version != 1:
        raise ValueError("SourceMatchVector version number is {:}, not 1.".format(version))
    filterName = "g" if "gaia" in refObjLoader.ref_dataset_name else matchMeta.getString("FILTER").strip()
    if epoch is None:
        try:
            epoch = matchMeta.getDouble("EPOCH")
        except (pexExceptions.NotFoundError, pexExceptions.TypeError):
            epoch = None  # Not present, or not correct type means it's not set
    epoch = astropy.time.Time(epoch, format="mjd", scale="tai") if epoch is not None else None
    if "RADIUS" in matchMeta:
        # This is a circle style metadata, call loadSkyCircle
        ctrCoord = geom.SpherePoint(matchMeta.getDouble("RA"), matchMeta.getDouble("DEC"), geom.degrees)
        rad = matchMeta.getDouble("RADIUS")*padRadiusFactor*geom.degrees
        refCat = refObjLoader.loadSkyCircle(ctrCoord, rad, filterName, epoch=epoch).refCat
    elif "INNER_UPPER_LEFT_RA" in matchMeta:
        # This is the sky box type (only triggers in the LoadReferenceObject
        # class, not task).  Only the outer box is required to be loaded to get
        # the maximum region, all filtering will be done below on the
        # unpackedMatches catalog, and no spatial filtering needs to be done by
        # the refObjLoader.
        box = []
        for place in ("UPPER_LEFT", "UPPER_RIGHT", "LOWER_LEFT", "LOWER_RIGHT"):
            coord = geom.SpherePoint(matchMeta.getDouble(f"OUTER_{place}_RA"),
                                     matchMeta.getDouble(f"OUTER_{place}_DEC"), geom.degrees).getVector()
            box.append(coord)
        outerBox = sphgeom.ConvexPolygon(box)
        refCat = refObjLoader.loadRegion(outerBox, filterName=filterName, epoch=epoch).refCat
    return refCat


def loadDenormalizeAndUnpackMatches(catalog, packedMatches, refObjLoader, epoch=None, padRadiusFactor=1.05,
                                    calibKey="calib_astrometry_used", log=None):
    """Function to load and denormalize a catalog of packed matches.

    A match list is persisted and unpersisted as a catalog of IDs produced by
    afw.table.packMatches(), with match metadata (as returned by the astrometry
    tasks) in the catalog's metadata attribute. This method converts such a
    match catalog into a match list, with links to source records and reference
    object records.

    NOTE: the refObjLoader.ref_dataset_name here must match that which was used
    in processing.

    Parameters
    ----------
    catalog : `'pandas.core.frame.DataFrame`
        The source catalog linked to the ``patchMatches`` catalog.
    packedMatches : `lsst.afw.table.BaseCatalog`
        Catalog of packed matches to be denormalized (i.e. load in associated
        reference catalogs with full column information).
    refObjLoader :
           `lsst.meas.algorithms.loadReferenceObjects.LoadReferenceObjectsTask`
        Reference object loader to read in the reference catalogs.
    padRadiusFactor : `float`, optional
        Factor by which to "pad" (increase) the sky circle radius to be loaded
        from that stored in the metadata associated with `packedMatches`.
    calibKey : `str`, optional
       The packed matches persisted in srcMatch represent those that were used
       in the SFM astrometric calibration, so we may as well clip the src
       catalog to just those with that flag set before joining the catalogs.
    log : `lsst.log.Log`, optional
        Logger object for logging messages.

    Returns
    -------
    unpackedMatches : `pandas.core.frame.DataFrame`
        The unpacked match catalog (i.e. contains all fields from the original
        source and external reference catalogs (but with "src_" and "ref_"
        prefixes on the column names).
    """
    matchMeta = packedMatches.table.getMetadata()
    refCat = loadRefCat(matchMeta, refObjLoader, epoch=epoch, padRadiusFactor=padRadiusFactor)
    refCat = refCat.asAstropy().to_pandas().set_index("id")
    packedMatches = packedMatches.asAstropy().to_pandas().set_index("first")
    denormMatches = pd.merge(packedMatches, refCat, left_index=True, right_index=True)
    # Raise if no matches were found
    if denormMatches.empty:
        if log is not None:
            logStr = ("No matches were found.  You are using {} as the reference catalog.  Are you "
                      "sure this is the same as the one used in the processing that produced the "
                      "[src/deepCoadd]Match catlogs?  Any plots based on the calib_astrometry_used flags "
                      "will use the generic matched catalog.".format(refObjLoader.ref_dataset_name))
            log.warn(logStr)
        return None
    # Check that matches were found for all obects in patchedMatches catalog
    numUnmatched = len(packedMatches.index) - len(denormMatches)
    if numUnmatched > 0 and log is not None:
        logStr = ("No match found for N={} objects (out of {}) in the packedMatch catalog. "
                  "Try increasing padRadiusFactor (currently = {}) to load sources over a "
                  "wider area?".format(numUnmatched, len(packedMatches.index), padRadiusFactor))
        log.warn(logStr)
    if calibKey is not None:
        catalogCopy = catalog[catalog[calibKey]].copy(deep=True)
    else:
        catalogCopy = catalog.copy(deep=True)
    catalogCopy.rename(columns=lambda x: "src_" + x, inplace=True)
    denormMatches.rename(columns=lambda x: "ref_" + x if x != "distance" else x, inplace=True)
    unpackedMatches = catalogCopy.join(denormMatches.set_index("ref_second"), on="src_id")
    unpackedMatches = unpackedMatches[unpackedMatches["distance"].notnull()]

    return unpackedMatches


def loadReferencesAndMatchToCatalog(catalog, matchMeta, refObjLoader, epoch=None, padRadiusFactor=1.05,
                                    matchRadius=0.5, matchFlagList=[], goodFlagList=[], minSrcSn=30.0,
                                    log=None):
    """Function to load a reference catalog and match it to a source catalog.

    When loading in reference objects for a region defined by an image's
    current WCS estimate, the reference object loader grows the bbox by the
    config parameter pixelMargin.  This is set to 300 by default but this is
    not reflected by the radius parameter set in the metadata, so some matches
    may reside outside the circle searched within this radius.  Thus, we
    increase the radius set in the metadata by the factor ``padRadiusFactor``
    to accommodate.

    Parameters
    ----------
    catalog : `'pandas.core.frame.DataFrame`
        The source catalog to which the maching will be done.
    matchMeta : `lsst.daf.base.propertyContainer.PropertyList`
        Metadata providing the search center and radius parameters.
    refObjLoader :
           `lsst.meas.algorithms.loadReferenceObjects.LoadReferenceObjectsTask`
        Reference object loader to read in the reference catalogs.
    padRadiusFactor : `float`, optional
        Factor by which to "pad" (increase) the sky circle radius to be loaded
        from that stored in the metadata.
    matchRadius : `float`, optional
        The radius within which to consider two objects a match in units of
        arcsec.
    matchFlagList : `list` of `str`, optional
        List of column flag names for which to cull sources before matching to
        the reference catalag if any are set to `True`.  An exception is made
        for any sources that were used in the SFM calibration (identified by
        the "calib_*_used" flags).  The later are all retained for matching
        regardless of any other flags being set.
    goodFlagList : `list` of `str`, optional
        List of column flag names for which to retain catalog sources having
        any one of them set to `True`, regardless of any other flags being set.
        For example, it may be desireable to keep all sources that were used in
        the SFM calibration (identified by the "calib_*_used" flags).
    minSrcSn : `float`, optional
        Minimum signal-to-noise ratio for sources in `catalog` to be considered
        in matching.
    log : `lsst.log.Log`, optional
        Logger object for logging messages.

    Returns
    -------
    matches : `pandas.core.frame.DataFrame`
        The unpacked match catalog of all matches (i.e. contains all fields
        from the original source and external reference catalogs (but with
        "src_" and "ref_" prefixes on the column names).
    """
    refCat = loadRefCat(matchMeta, refObjLoader, epoch=epoch, padRadiusFactor=padRadiusFactor)
    refCat = refCat.asAstropy().to_pandas().set_index("id")
    schema = getSchema(catalog)
    flagList = []
    for flag in matchFlagList:
        if flag in schema:
            flagList.append(flag)
        else:
            if log is not None:
                log.warn("Did not find column {:} in catalog so it will not be added to the list of "
                         "flags for culling the source catalog prior to the generic matching.".format(flag))
    # Cull on bad sources from the catalogs as these should not be
    # considered in our match-to-reference catalog metric.  However, we allow
    # an option to explicitly leave in sources for which any of the flags in
    # goodFlagList are set to True.
    bad = makeBadArray(catalog, flagList=flagList)
    bad |= catalog["base_PsfFlux_instFlux"].values/catalog["base_PsfFlux_instFluxErr"].values < minSrcSn
    good = ~bad
    for goodFlag in goodFlagList:
        good |= catalog[goodFlag].values
    goodCatalog = catalog[good].copy(deep=True)
    matches = matchAndJoinCatalogs(goodCatalog, refCat, matchRadius, prefix1="src_", prefix2="ref_", log=log)
    return matches


def computeAreaDict(repoInfo, dataRefList, dataset="", fakeCat=None, raFakesCol="raJ2000",
                    decFakesCol="decJ2000", toMaskList=["BAD", "NO_DATA"]):
    """Compute the effective area of each image element (detector or patch).

    The effective area is computed while masking out the pixels with the
    masks planes included in ``toMaskList``.

    Parameters
    ----------
    repoInfo : `lsst.pipe.base.Struct`
        A struct containing relevant information about the repository under
        study.  Elements used here include the key name associated with a
        detector and the butler (`lsst.daf.persistence.Butler`) associated
        with the dataset.
    dataRefList : `list` of
                  `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        The `list` of butler data references under consideration.
    dataset : `str`, optional
        Name of the ``dataset`` to be used in order to set up a Fits reader
        (`lsst.afw.image.readers.ExposureFitsReader`) via the ``dataset``s
        URI obtained from the butler (`lsst.daf.persistence.Butler`) stored
        in ``repoInfo`` (e.g. "deepCoadd" for coadds, blank `str` for visits).
    fakeCat : `pandas.core.frame.DataFrame` or `None`, optional
        Catalog of fake sources.  If not `None`, to which
        a column (onCcd) is added with the ccd number if the fake
        source overlaps a ccd and np.nan if it does not.
    raFakesCol : `str`, optional
        The RA column to use from the fakes catalog.
    decFakesCol : `str`, optional
        The Dec. column to use from the fakes catalog.
    toMaskList : `list` of `str`, optional
        The `list` of mask plane names to be ignored in the effectie are
        computation.

    Raises
    ------
    RuntimeError
        If it cannot be established whether this is coadd or visit catalog data
        based on the information in the ``repoInfo`` object.

    Returns
    -------
    areaDict : `dict`
        A `dict` containing the area and corner locations of each element
        (detector for visits, patch for coadds).
        Examples of keys: there is one of these for every element ID specified
        when the code is called.
            ``"elementId"``
                The effective area of the the element (i.e where none of the
                mask bits of the planes in ``toMaskList`` are set), in
                arcseconds.
            ``"corners_elemenId"``
                A `list` of `lsst.geom.SpherePoint`s providing the corners
                of the element in degrees.
    fakeCat : `pandas.core.frame.DataFrame` or `None`
        If the input ``fakeCat`` is not `None`, the updated catalog of fake
        sources in which a column (onElement) has been added with the element
        id number if the fake source overlaps the given element and `numpy.nan`
        if it does not.
    """
    isPatch = True if "patch" in repoInfo.dataId else False
    isCcd = True if repoInfo.ccdKey in repoInfo.dataId else False
    if (not isPatch and not isCcd) or (isPatch and isCcd):
        raise RuntimeError("Cannot establish whether this is coadd or visit catalog data")
    areaDict = {}
    dataset = dataset + "_" if dataset != "" and dataset[-1] != "_" else dataset
    elementKey = "patch" if isPatch else repoInfo.ccdKey
    for dataRef in dataRefList:
        # getUri is less safe but enables us to use an efficient
        # ExposureFitsReader.
        if repoInfo.isGen3:
            dataId = dataRef["dataId"].copy()
            uri = repoInfo.butler.getURI(dataset + "calexp", dataId)
            fname = uri.path
            fname = fname.replace("%2C", ",")  # hack for gen2-gen3 converted repos
            ccdId = dataId[elementKey]
        else:
            fname = repoInfo.butler.getUri(dataset + "calexp", dataRef.dataId)
            ccdId = dataRef.dataId[elementKey]
        reader = afwImage.ExposureFitsReader(fname)
        if repoInfo.skyWcsDataset is not None and isCcd:
            if not repoInfo.isGen3:
                wcs = dataRef.get(repoInfo.skyWcsDataset)
            else:
                externalSkyWcsCatalog = dataRef["butler"].get(repoInfo.skyWcsDataset, dataId=dataId)
                row = externalSkyWcsCatalog.find(dataId["detector"])
                wcs = row.getWcs()
        else:
            wcs = reader.readWcs()
        mask = reader.readMask()
        maskedPixels = None
        for maskPlane in toMaskList:
            maskBad = mask.array & 2**mask.getMaskPlaneDict()[maskPlane]
            maskedPixels = maskBad if maskedPixels is None else maskedPixels + maskBad
        numGoodPix = np.count_nonzero(maskedPixels == 0)
        if isCcd:
            detector = reader.readDetector()
            pixScale = wcs.getPixelScale(detector.getCenter(cameraGeom.PIXELS)).asArcseconds()
            corners = wcs.pixelToSky(detector.getCorners(cameraGeom.PIXELS))
        else:
            pixScale = wcs.getPixelScale(reader.readBBox().getCenter()).asArcseconds()
            corners = wcs.pixelToSky(geom.Box2D(reader.readBBox()).getCorners())
        areaDict["corners_" + str(ccdId)] = corners
        area = numGoodPix*pixScale**2
        areaDict[ccdId] = area

        if fakeCat is not None:
            fakeCat = determineIfSrcOnElement(fakeCat, dataRef, corners, wcs, elementKey, mask=mask,
                                              reader=reader, raCol=raFakesCol, decCol=decFakesCol)
    return areaDict, fakeCat


def determineIfSrcOnElement(catalog, dataRef, corners, wcs, elementKey, mask=None, reader=None,
                            raCol="raJ2000", decCol="decJ2000"):
    """Determine which sources in a catalog lie on a given image element.

    A new column is added to ``catalog`` with name on\"ElementKey\" (with the
    first letter capitalized, e.g. onPatch, onCcd) and is given the value
    of the element ID if the source lies within the ``dataRef``s validPolygon
    or `numpy.nan` if it does not.

    Parameters
    ----------
    catalog : `pandas.core.frame.DataFrame`
        Catalog of sources under consideration to which a column (onElement) is
        added.  For a given source, the value the element Id if the source
        overlaps the element associate with ``dataRef`` or `numpy.nan` if it
        does not.
    dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
        The butler data reference under consideration.
    corners : `list` of `lsst.geom.SpherePoint`
        A `list` of the four corners of the element (in degrees).
    wcs : `lsst.afw.geom.SkyWcs`
        The WCS solution associated with ``dataRef``.
    elementKey : `str`
       The column name or key associated with the image element.
    mask : `lsst.afw.image.MaskX`, or `None`, optional
       The mask plane associated with ``dataRef``.
    raCol : `str`, optional
        The RA column to use from the fakes catalog.
    decCol : `str`, optional
        The Dec column to use from the fakes catalog.
    toMaskList : `list` of `str`, optional
        The `list` of mask plane names to be ignored in the effectie are
        computation.

    Raises
    ------
    RuntimeError
        If the data are coadd catalogs (as indicated by having the string
        \"patch\" in ``elementKey``), but no ``mask`` from which to compute the
        valid polygon was provided.

    Returns
    -------
    catalog : `pandas.core.frame.DataFrame` or `None`
        If the input ``catalog`` is not `None`, the updated catalog of fake
        sources in which a column (onElement) has been added with the element
        id number if the fake source overlaps the given element and `numpy.nan`
        if it does not.
    """
    isPatch = True if "patch" in elementKey else False
    if isPatch and mask is None:
        raise RuntimeError("Must provide a mask if isPatch is True")
    # Check which fake sources fall in the patch
    cornerRas = [cx.asRadians() for (cx, cy) in corners]
    cornerDecs = [cy.asRadians() for (cx, cy) in corners]
    posOnElement = np.where((catalog[raCol].values > np.min(cornerRas))
                            & (catalog[raCol].values < np.max(cornerRas))
                            & (catalog[decCol].values > np.min(cornerDecs))
                            & (catalog[decCol].values < np.max(cornerDecs)))[0]
    if "on" + elementKey.capitalize() not in catalog.columns:
        catalog["on" + elementKey.capitalize()] = [np.nan]*len(catalog)
    if isPatch:
        validPolygon = mask.getInfo().getValidPolygon()
    else:
        validPolygon = reader.readExposureInfo().getValidPolygon()

    onElementList = []
    for rowId in posOnElement:
        skyCoord = geom.SpherePoint(catalog[raCol].values[rowId],
                                    catalog[decCol].values[rowId], geom.radians)
        pixCoord = wcs.skyToPixel(skyCoord)
        onElement = validPolygon.contains(pixCoord)
        if onElement:
            onElementList.append(rowId)
    catalog["on" + elementKey.capitalize()].iloc[np.array(onElementList)] = dataRef.dataId[elementKey]
    return catalog


def getParquetColumnsList(pqTable, dfDataset=None, filterName=None):
    """Determine the list of columns for a (multilevel)parquet table.

    NOTE: for a multilevel table, we are assuming it has 3 levels with names
    "dataset", "filter", "column", as is the format persisted in the "_obj"
    coadd catalogs from pipe_tasks' postprocess.py.  The returned list of
    column names is that associated with the level specified by ``dfDataset``
    and ``filterName``.

    Parameters
    ----------
    parquetTable : `lsst.pipe.tasks.parquetTable.ParquetTable` or
                   `lsst.pipe.tasks.parquetTable.MultilevelParquetTable`
        The parquet table from which to extract the column names
    dfDataset : `str`, optional
        Name of the parquet table "dataset" level for which the columns
        list is to be derived.  If ``parquetTable`` is multilevel, this is
        actually not optional.  For single level catalogs, it is not relevant.
    filterName : `str`, optional
        Name of the parquet table "filter" level for which the columns
        list is to be derived.  If ``parquetTable`` is multilevel, this is
        actually not optional.  For single level catalogs, it is not relevant.

    Raises
    ------
    RuntimeError
        If ``parquetTable`` is multilevel but does not have the expected 3
        levels.
    RuntimeError
        If ``parquetTable`` is multilevel but either ``dfDataset`` or
        ``filterName`` are not set.

    Returns
    -------
    catColumnsList : `list` of `str`
        The list of column names for the ``parquetTable``.  For multilevel
        parquet tables, the columns list will be associated with the level
        specified by ``dfDataset`` and ``filterName``.
    """
    nLevels = 3
    if isinstance(pqTable, MultilevelParquetTable):
        if len(pqTable.columnLevels) != nLevels:
            raise RuntimeError("Unknown multilevel parquet table: expect {:} levels but got {:}".
                               format(nLevels, len(pqTable.columnLevels)))
        if dfDataset is None or filterName is None:
            raise RuntimeError("Both dfDataset and filterName must be set for multilevel parquet tables: "
                               "but got {:} and {:}, respectively".format(dfDataset, filterName))
        pqDatasetNames, pqFilterNames, pqColumnNames = pqTable.columnIndex.levels
        catColumnsList = [pqColumnNames[icx] for idx, ifx, icx in zip(*pqTable.columnIndex.codes)
                          if idx == list(pqDatasetNames).index(dfDataset)
                          and ifx == list(pqFilterNames).index(filterName)]
    elif isinstance(pqTable.columns, pd.MultiIndex):
        if len(pqTable.columns.levels) != nLevels:
            raise RuntimeError("Unknown multiIndex dataFrame: expect {:} levels but got {:}".
                               format(nLevels, len(pqTable.columns.levels)))
        if dfDataset is None or filterName is None:
            raise RuntimeError("Both dfDataset and filterName must be set for multiIndex dataFrames: "
                               "but got {:} and {:}, respectively".format(dfDataset, filterName))
        catColumnsList = [tup[2] for tup in pqTable.columns if tup[0] == dfDataset and tup[1] == filterName]
    else:
        catColumnsList = pqTable.columns
    return catColumnsList
