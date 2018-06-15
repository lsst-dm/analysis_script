#!/usr/bin/env python

import matplotlib
matplotlib.use("Agg")  # noqa #402
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")  # noqa #402
import functools
import os
import scipy.stats as scipyStats

from collections import defaultdict

from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from .analysis import Analysis, AnalysisConfig
from .coaddAnalysis import CoaddAnalysisTask
from .utils import (Filenamer, Enforcer, concatenateCatalogs, checkIdLists, getFluxKeys, addFlag,
                    addIntOrFloatColumn, addPatchColumn, calibrateCoaddSourceCatalog, fluxToPlotString,
                    writeParquet, getRepoInfo, orthogonalRegression, distanceSquaredToPoly,
                    p2p1CoeffsFromLinearFit, p1CoeffsFromP2x0y0, linesFromP2P1Coeffs, makeEqnStr,
                    catColors)
from .plotUtils import AllLabeller, OverlapsStarGalaxyLabeller, plotText, labelCamera, setPtSize

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

__all__ = ["ColorTransform", "ivezicTransformsSDSS", "ivezicTransformsHSC", "straightTransforms",
           "NumStarLabeller", "ColorValueInFitRange", "ColorValueInPerpRange", "GalaxyColor",
           "ColorAnalysisConfig", "ColorAnalysisRunner", "ColorAnalysisTask", "ColorColorDistance",
           "SkyAnalysisRunner", "SkyAnalysisTask"]


class ColorTransform(Config):
    description = Field(dtype=str, doc="Description of the color transform")
    subDescription = Field(dtype=str, doc="Sub-description of the color transform (added detail)")
    plot = Field(dtype=bool, default=True, doc="Plot this color?")
    coeffs = DictField(keytype=str, itemtype=float, doc="Coefficients for each filter")
    x0 = Field(dtype=float, default=None, optional=True,
               doc="x Origin of P1/P2 axis on the color-color plane")
    y0 = Field(dtype=float, default=None, optional=True,
               doc="y Origin of P1/P2 axis on the color-color plane")
    requireGreater = DictField(keytype=str, itemtype=float, default={},
                               doc="Minimum values for colors so that this is useful")
    requireLess = DictField(keytype=str, itemtype=float, default={},
                            doc="Maximum values for colors so that this is useful")
    fitLineSlope = Field(dtype=float, default=None, optional=True, doc="Slope for fit line limits")
    fitLineUpperIncpt = Field(dtype=float, default=None, optional=True,
                              doc="Intercept for upper fit line limits")
    fitLineLowerIncpt = Field(dtype=float, default=None, optional=True,
                              doc="Intercept for lower fit line limits")

    @classmethod
    def fromValues(cls, description, subDescription, plot, coeffs, x0=None, y0=None, requireGreater={},
                   requireLess={}, fitLineSlope=None, fitLineUpperIncpt=None, fitLineLowerIncpt=None):
        self = cls()
        self.description = description
        self.subDescription = subDescription
        self.plot = plot
        self.coeffs = coeffs
        self.x0 = x0
        self.y0 = y0
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.fitLineSlope = fitLineSlope
        self.fitLineUpperIncpt = fitLineUpperIncpt
        self.fitLineLowerIncpt = fitLineLowerIncpt
        return self


ivezicTransformsSDSS = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"SDSS-G": -0.227, "SDSS-R": 0.792, "SDSS-I": -0.567, "": 0.050},
                                       x0 = 0.4250,  y0 = 0.0818,
                                       requireGreater={"wPara": -0.2}, requireLess={"wPara": 0.6}),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"SDSS-G": 0.707, "SDSS-R": -0.707, "": -0.988},
                                       requireGreater={"xPara": 0.8}, requireLess={"xPara": 1.6}),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"SDSS-R": -0.270, "SDSS-I": 0.800, "SDSS-Z": -0.534, "": 0.054},
                                       x0 = 0.5763, y0 = 0.1900,
                                       requireGreater={"yPara": 0.1}, requireLess={"yPara": 1.2}),
    "wPara": ColorTransform.fromValues("Ivezic w parallel", " (griBlue)", False,
                                       {"SDSS-G": 0.928, "SDSS-R": -0.556, "SDSS-I": -0.372, "": -0.425}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", " (griRed)", False,
                                       {"SDSS-R": 1.0, "SDSS-I": -1.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", " (rizRed)", False,
                                       {"SDSS-R": 0.895, "SDSS-I": -0.448, "SDSS-Z": -0.447, "": -0.600}),
}

ivezicTransformsHSC = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", " (griBlue)", True,
                                       {"HSC-G": -0.272, "HSC-R": 0.803, "HSC-I": -0.531, "": 0.036},
                                       x0 = 0.4481, y0 = 0.1546,
                                       requireGreater={"wPara": -0.2}, requireLess={"wPara": 0.6},
                                       fitLineSlope=-1/0.51, fitLineUpperIncpt=2.40, fitLineLowerIncpt=0.68),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", " (griRed)", True,
                                       {"HSC-G": 0.678, "HSC-R": -0.733, "HSC-I": 0.055, "": -0.792},
                                       x0 = 1.2654, y0 = 1.3675,
                                       requireGreater={"xPara": 0.8}, requireLess={"xPara": 1.6},
                                       fitLineSlope=-1/11.4, fitLineUpperIncpt=1.73, fitLineLowerIncpt=0.87),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", " (rizRed)", True,
                                       {"HSC-R": -0.227, "HSC-I": 0.793, "HSC-Z": -0.566, "": -0.012},
                                       x0 = 1.2219, y0 = 0.5183,
                                       requireGreater={"yPara": 0.1}, requireLess={"yPara": 1.2},
                                       fitLineSlope=-1/0.40, fitLineUpperIncpt=5.5, fitLineLowerIncpt=2.7),
    # The following still default to the SDSS values.  HSC coeffs will be derived on a subsequent
    # commit
    "wPara": ColorTransform.fromValues("Ivezic w parallel", " (griBlue)", False,
                                       {"HSC-G": 0.89, "HSC-R": -0.43, "HSC-I": -0.46, "": -0.52}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", " (griRed)", False,
                                       {"HSC-G": 0.0, "HSC-R": 1.0, "HSC-I": -1.0, "": 0.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", " (rizRed)", False,
                                       {"HSC-R": 0.928, "HSC-I": -0.555, "HSC-Z": -0.373, "": -1.400}),
    # The following three entries were derived in the process of calibrating the above coeffs (all three
    # RC2 tracts gave effectively the same fits).  May remove later if deemed no longer useful.
    "wFit": ColorTransform.fromValues("Straight line fit for wPerp range", " (griBlue)", False,
                                      {"HSC-G": 0.51, "HSC-R": -0.51, "": -0.07}),
    "xFit": ColorTransform.fromValues("Straight line fit for xperp range", " (griRed)", False,
                                      {"HSC-G": 11.4, "HSC-R": -11.4, "": -13.3}),
    "yFit": ColorTransform.fromValues("Straight line fit for yPerp range", " (rizRed)", False,
                                      {"HSC-R": 0.40, "HSC-I": -0.40, "": 0.02}),
}

straightTransforms = {
    "g": ColorTransform.fromValues("HSC-G", "", True, {"HSC-G": 1.0}),
    "r": ColorTransform.fromValues("HSC-R", "", True, {"HSC-R": 1.0}),
    "i": ColorTransform.fromValues("HSC-I", "", True, {"HSC-I": 1.0}),
    "z": ColorTransform.fromValues("HSC-Z", "", True, {"HSC-Z": 1.0}),
    "y": ColorTransform.fromValues("HSC-Y", "", True, {"HSC-Y": 1.0}),
    "n921": ColorTransform.fromValues("NB0921", "", True, {"NB0921": 1.0}),
}


class NumStarLabeller(object):
    labels = {"star": 0, "maybe": 1, "notStar": 2}
    plot = ["star", "maybe"]

    def __init__(self, numBands):
        self.numBands = numBands

    def __call__(self, catalog):
        return np.array([0 if nn >= self.numBands else 2 if nn == 0 else 1 for nn in catalog["numStarFlags"]])


class ColorValueInFitRange(object):
    """Functor to produce color value if in the appropriate range

    Here the range is set by upper and lower lines roughly perpendicular to the
    fit where those lines cross the fit.  These numbers were previously determined
    and are (currently) hard-wired in the fitLineSlope, fitLineUpperIncpt, and
    fitLineLowerIncpt parameters in the ivezicTransformsHSC dict.
    """
    def __init__(self, column, xColor, yColor, fitLineSlope=None, fitLineUpperIncpt=None,
                 fitLineLowerIncpt=None, unitScale=1.0):
        self.column = column
        self.xColor = xColor
        self.yColor = yColor
        self.fitLineSlope = fitLineSlope
        self.fitLineUpperIncpt = fitLineUpperIncpt
        self.fitLineLowerIncpt = fitLineLowerIncpt
        self.unitScale = unitScale

    def __call__(self, principalColCats):
        good = self.yColor > self.fitLineLowerIncpt + self.fitLineSlope*self.xColor
        good &= self.yColor < self.fitLineUpperIncpt + self.fitLineSlope*self.xColor
        return np.where(good, principalColCats[self.column], np.nan)*self.unitScale


class ColorValueInPerpRange(object):
    """Functor to produce color value if in the appropriate range

    Here the range is set by the Ivezic etal. P1 limits provided in the requireGreater
    and requireLess parameters in the ivezicTransforms dict.
    """
    def __init__(self, column, requireGreater, requireLess, unitScale=1.0):
        self.column = column
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.unitScale = unitScale

    def __call__(self, principalColCats):
        good = np.ones(len(principalColCats), dtype=bool)
        for col, value in self.requireGreater.items():
            good &= principalColCats[col] > value
        for col, value in self.requireLess.items():
            good &= principalColCats[col] < value
        return np.where(good, principalColCats[self.column], np.nan)*self.unitScale


class GalaxyColor(object):
    """Functor to produce difference between galaxy color calculated by different algorithms"""
    def __init__(self, alg1, alg2, prefix1, prefix2):
        self.alg1 = alg1
        self.alg2 = alg2
        self.prefix1 = prefix1
        self.prefix2 = prefix2

    def __call__(self, catalog):
        color1 = -2.5*np.log10(catalog[self.prefix1 + self.alg1]/catalog[self.prefix2 + self.alg1])
        color2 = -2.5*np.log10(catalog[self.prefix1 + self.alg2]/catalog[self.prefix2 + self.alg2])
        return color1 - color2


class ColorAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["slot_Centroid_flag", "slot_Shape_flag",
                               "base_PsfFlux_flag", "modelfit_CModel_flag",
                               "base_PixelFlags_flag_saturated", "base_ClassificationExtendedness_flag"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc=("Filter to use for plotting against magnitude and "
                                                        "setting star/galaxy classification"))
    fluxFilterGeneric = Field(dtype=str, default="i", doc=("Filter to use for plotting against magnitude "
                                                           "and setting star/galaxy classification"))
    srcSchemaMap = DictField(keytype=str, itemtype=str, default=None, optional=True,
                             doc="Mapping between different stack (e.g. HSC vs. LSST) schema names")
    extinctionCoeffs = DictField(keytype=str, itemtype=float, default=None, optional=True,
                                 doc="Dictionary of extinction coefficients for conversion from E(B-V) "
                                 "to extinction, A_filter")
    correctForGalacticExtinction = Field(dtype=bool, default=True,
                                         doc="Correct flux fields for Galactic Extinction?  Must have "
                                         "extinctionCoeffs config setup.")
    toMilli = Field(dtype=bool, default=True, doc="Print stats in milli units (i.e. mas, mmag)?")
    doPlotPrincipalColors = Field(dtype=bool, default=True,
                                  doc="Create the Ivezic Principal Color offset plots?")
    doPlotGalacticExtinction = Field(dtype=bool, default=True, doc="Create Galactic Extinction plots?")
    writeParquetOnly = Field(dtype=bool, default=False,
                             doc="Only write out Parquet tables (i.e. do not produce any plots)?")
    doWriteParquetTables = Field(dtype=bool, default=True,
                                 doc=("Write out Parquet tables (for subsequent interactive analysis)?"
                                      "\nNOTE: if True but fastparquet package is unavailable, a warning "
                                      "is issued and table writing is skipped."))
    plotRanges = DictField(keytype=str, itemtype=float,
                           default={"griX0": -0.6, "griX1": 2.0, "griY0": -0.6, "griY1": 3.0,
                                    "rizX0": -0.4, "rizX1": 3.0, "rizY0": -0.2, "rizY1": 1.5,
                                    "izyX0": -0.5, "izyX1": 1.3, "izyY0": -0.4, "izyY1": 0.8,
                                    "z9yX0": -0.3, "z9yX1": 0.45, "z9yY0": -0.2, "z9yY1": 0.5},
                           doc="Plot Ranges for various color-color combinations")

    def setDefaults(self):
        Config.setDefaults(self)
        self.transforms = ivezicTransformsHSC
        self.analysis.flags = []  # We remove bad source ourself
        self.analysis.magThreshold = 22.0  # RHL requested this limit
        if self.correctForGalacticExtinction:
            self.flags += ["galacticExtinction_flag"]

    def validate(self):
        Config.validate(self)
        if self.writeParquetOnly and not self.doWriteParquetTables:
            raise ValueError("Cannot writeParquetOnly if doWriteParquetTables is False")
        if self.correctForGalacticExtinction and self.extinctionCoeffs is None:
            raise ValueError("Must set appropriate extinctionCoeffs config.  See "
                             "config/hsc/extinctionCoeffs.py in obs_subaru for an example.")

        # If a wired origin was included in the config, check that it actually lies on the wired P1 line
        if self.transforms:
            for col, transform in self.transforms.items():
                if transform.plot and transform.x0 is not None and transform.y0 is not None:
                    transformPerp = self.transforms[col]
                    transformPara = self.transforms[col[0] + "Para"]
                    p1p2Lines = linesFromP2P1Coeffs(list(transformPerp.coeffs.values()),
                                                    list(transformPara.coeffs.values()))
                    if (np.abs((p1p2Lines.mP1 - p1p2Lines.mP2)*transformPerp.x0 +
                               (p1p2Lines.bP1 - p1p2Lines.bP2)) > 1e-4):
                        raise ValueError(("Wired origin for {} does not lie on line associated with wired "
                                          "PCA coefficients.  Check that the wired values are correct.").
                                         format(col))


class ColorAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        FilterRefsDict = functools.partial(defaultdict, list)  # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict)  # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            # Make sure the actual input file requested exists (i.e. do not follow the parent chain)
            inputDataFile = patchRef.get("deepCoadd_forced_src_filename")[0]
            if parsedCmd.input not in parsedCmd.output:
                inputDataFile = inputDataFile.replace(parsedCmd.output, parsedCmd.input)
            if os.path.exists(inputDataFile):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                tractFilterRefs[tract][filterName].append(patchRef)

        # Find tract,patch with full colour coverage (makes combining catalogs easier)
        bad = []
        for tract in tractFilterRefs:
            filterRefs = tractFilterRefs[tract]
            patchesForFilters = [set(patchRef.dataId["patch"] for patchRef in patchRefList) for
                                 patchRefList in filterRefs.values()]
            if not patchesForFilters:
                parsedCmd.log.warn("No input data found for tract {:d}".format(tract))
                bad.append(tract)
                continue
            keep = set.intersection(*patchesForFilters)  # Patches with full colour coverage
            tractFilterRefs[tract] = {ff: [patchRef for patchRef in filterRefs[ff] if
                                           patchRef.dataId["patch"] in keep] for ff in filterRefs}
        for tract in bad:
            del tractFilterRefs[tract]

        # List of filters included on the command line
        parsedFilterList = [dataId["filter"] for dataId in parsedCmd.id.idList]
        for tract in tractFilterRefs:
            numFilters = 0
            for ff in parsedFilterList:
                if ff in tractFilterRefs[tract].keys():
                    numFilters += 1
                else:
                    parsedCmd.log.warn("No input data found for filter {0:s} of tract {1:d}".
                                       format(ff, tract))
            if numFilters < 3:
                parsedCmd.log.warn("Must have at least 3 filters with data existing in the input repo. "
                                   "Only {0:d} exist of those requested ({1:}) for tract {2:d}. "
                                   "Skipping tract.".format(numFilters, parsedFilterList, tract))
                del tractFilterRefs[tract]
            if not tractFilterRefs[tract]:
                raise RuntimeError("No suitable datasets found.")

        return [(filterRefs, kwargs) for filterRefs in tractFilterRefs.values()]


class ColorAnalysisTask(CmdLineTask):
    ConfigClass = ColorAnalysisConfig
    RunnerClass = ColorAnalysisRunner
    AnalysisClass = Analysis
    _DefaultName = "colorAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_forced_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.unitScale = 1000.0 if self.config.toMilli else 1.0

    def run(self, patchRefsByFilter):
        patchList = []
        repoInfo = None
        self.fluxFilter = None
        for patchRefList in patchRefsByFilter.values():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.config.fluxFilter:
                    self.fluxFilter = self.config.fluxFilter
                    break
                if dataRef.dataId["filter"] == self.config.fluxFilterGeneric:
                    self.fluxFilter = self.config.fluxFilterGeneric
                    break
        if self.fluxFilter is None:
            raise TaskError("Flux filter from config not found (neither {0:s} nor the generic {1:s}".
                            format(self.config.fluxFilter, self.config.fluxFilterGeneric))
        self.log.info("Flux filter for plotting and primary star/galaxy classifiation is: {0:s}".
                      format(self.fluxFilter))
        for patchRefList in patchRefsByFilter.values():
            for dataRef in patchRefList:
                if dataRef.dataId["filter"] == self.fluxFilter:
                    patchList.append(dataRef.dataId["patch"])
                    if repoInfo is None:
                        repoInfo = getRepoInfo(dataRef, coaddName=self.config.coaddName,
                                               coaddDataset="Coadd_forced_src")
        self.log.info("Size of patchList with full color coverage: {:d}".format(len(patchList)))

        # Only adjust the schema names necessary here (rather than attaching the full alias schema map)
        self.fluxColumn = self.config.analysis.fluxColumn
        self.classificationColumn = "base_ClassificationExtendedness_value"
        self.flags = self.config.flags
        if repoInfo.hscRun is not None:
            self.fluxColumn = self.config.srcSchemaMap[self.config.analysis.fluxColumn] + "_flux"
            self.classificationColumn = self.config.srcSchemaMap[self.classificationColumn]
            self.flags = [self.config.srcSchemaMap[flag] for flag in self.flags]

        filenamer = Filenamer(repoInfo.butler, "plotColor", repoInfo.dataId)
        byFilterForcedCats = {ff: self.readCatalogs(patchRefList,
                                                    self.config.coaddName + "Coadd_forced_src") for
                              ff, patchRefList in patchRefsByFilter.items()}

        for cat in byFilterForcedCats.values():
            calibrateCoaddSourceCatalog(cat, self.config.analysis.coaddZp)

        geLabel = "None"
        doPlotGalacticExtinction = False
        if self.correctForGalacticExtinction:
            # The per-object Galactic Extinction correction currently requires sims_catUtils to be setup
            # as it uses the EBVbase class to obtain E(B-V).  Putting this in a try/except to fall back
            # to the per-field correction until we can access the EBVbase class from an lsst_distrib
            # installation.
            try:
                byFilterForcedCats = self.correctForGalacticExtinction(byFilterForcedCats, repoInfo.tractInfo)
                doPlotGalacticExtinction = True
                geLabel = "Per Object"
            except Exception:
                byFilterForcedCats = self.correctFieldForGalacticExtinction(byFilterForcedCats,
                                                                            repoInfo.tractInfo)
                geLabel = "Per Field"

        geLabel = "GE applied: " + geLabel
        if self.config.doPlotGalacticExtinction and doPlotGalacticExtinction:
            self.plotGalacticExtinction(byFilterForcedCats, filenamer, repoInfo.dataId,
                                        camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                        patchList=patchList, hscRun=repoInfo.hscRun, geLabel=geLabel)

        # self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        principalColCats = self.transformCatalogs(byFilterForcedCats, self.config.transforms,
                                                  hscRun=repoInfo.hscRun)

        # Create and write parquet tables
        if self.config.doWriteParquetTables:
            tableFilenamer = Filenamer(repoInfo.butler, 'qaTableColor', repoInfo.dataId)
            writeParquet(principalColCats, tableFilenamer(repoInfo.dataId, description='forced'))
            if self.config.writeParquetOnly:
                self.log.info("Exiting after writing Parquet tables.  No plots generated.")
                return

        if self.config.doPlotPrincipalColors:
            self.plotStarPrincipalColors(principalColCats, byFilterForcedCats, filenamer,
                                         NumStarLabeller(3), repoInfo.dataId, camera=repoInfo.camera,
                                         tractInfo=repoInfo.tractInfo, patchList=patchList,
                                         hscRun=repoInfo.hscRun, geLabel=geLabel)

        for fluxColumn in ["base_PsfFlux_flux", "modelfit_CModel_flux"]:
            self.plotStarColorColor(principalColCats, byFilterForcedCats, filenamer, repoInfo.dataId,
                                    fluxColumn, camera=repoInfo.camera, tractInfo=repoInfo.tractInfo,
                                    patchList=patchList, hscRun=repoInfo.hscRun, geLabel=geLabel)

    def readCatalogs(self, patchRefList, dataset):
        """Read in and concatenate catalogs of type dataset in lists of data references

        If self.config.doWriteParquetTables is True, before appending each catalog to a single
        list, an extra column indicating the patch is added to the catalog.  This is useful for
        the subsequent interactive QA analysis.

        Parameters
        ----------
        patchRefList : `list` of `lsst.daf.persistence.butlerSubset.ButlerDataRef`
           A list of butler data references whose catalogs of dataset type are to be read in
        dataset : `str`
           Name of the catalog dataset to be read in

        Raises
        ------
        `TaskError`
           If no data is read in for the dataRefList

        Returns
        -------
        `list` of concatenated `lsst.afw.table.source.source.SourceCatalog`s
        """
        catList = []
        for patchRef in patchRefList:
            if patchRef.datasetExists(dataset):
                cat = patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_HEAVY_FOOTPRINTS)
                if self.config.doWriteParquetTables:
                    cat = addPatchColumn(cat, patchRef.dataId["patch"])
                catList.append(cat)
        if not catList:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        return concatenateCatalogs(catList)

    def correctForGalacticExtinction(self, catalog, tractInfo):
        """Correct all fluxes for each object for Galactic Extinction

        This function uses the EBVbase class from lsst.sims.catUtils.dust.EBV, so lsst.sims.catUtils must
        be setup and accessible for use.

        Parameters
        ----------
        catalog : `lsst.afw.table.source.source.SourceCatalog`
           The source catalog for which to apply the per-object Galactic Extinction correction to all fluxes.
           Catalog is corrected in place and a Galactic Extinction applied and flag columns are added.
        tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
           TractInfo object associated with catalog

        Raises
        ------
        `ImportError`
           If lsst.sims.catUtils.dust.EBV could not be imported.

        Returns
        -------
        Updated `lsst.afw.table.source.source.SourceCatalog` catalog with fluxes corrected for
        Galactic Extinction with a column added indicating correction applied and a flag indicating
        if the correction failed (in the context having a non-np.isfinite value).
        """
        try:
            from lsst.sims.catUtils.dust.EBV import EBVbase as ebv
        except ImportError:
            raise ImportError("lsst.sims.catUtils.dust.EBV could not be imported.  Cannot use "
                              "correctForGalacticExtinction function without it.")

        for ff in catalog.keys():
            if ff in self.config.extinctionCoeffs:
                raList = catalog[ff]["coord_ra"]
                decList = catalog[ff]["coord_dec"]
                ebvObject = ebv()
                ebvValues = ebvObject.calculateEbv(equatorialCoordinates=np.array([raList, decList]))
                galacticExtinction = ebvValues*self.config.extinctionCoeffs[ff]
                bad = ~np.isfinite(galacticExtinction)
                if ~np.isfinite(galacticExtinction).all():
                    self.log.warn("Could not compute {0:s} band Galactic Extinction for "
                                  "{1:d} out of {2:d} sources.  Flag will be set.".
                                  format(ff, len(raList[bad]), len(raList)))
                factor = 10.0**(0.4*galacticExtinction)
                fluxKeys, errKeys = getFluxKeys(catalog[ff].schema)
                self.log.info("Applying per-object Galactic Extinction correction for filter {0:s}.  "
                              "Catalog mean A_{0:s} = {1:.3f}". format(ff, galacticExtinction[~bad].mean()))
                for name, key in list(fluxKeys.items()) + list(errKeys.items()):
                    catalog[ff][key] *= factor
            else:
                self.log.warn("Do not have A_X/E(B-V) for filter {0:s}.  "
                              "No Galactic Extinction correction applied for that filter.  "
                              "Flag will be set".format(ff))
                bad = np.ones(len(catalog[list(catalog.keys())[0]]), dtype=bool)
            # Add column of Galactic Extinction value applied to the catalog and a flag for the sources
            # for which it could not be computed
            catalog[ff] = addIntOrFloatColumn(catalog[ff], galacticExtinction, "A_" + str(ff),
                                              "Galactic Extinction (in mags) applied "
                                              "(based on SFD 1998 maps)")
            catalog[ff] = addFlag(catalog[ff], bad, "galacticExtinction_flag",
                                  "True if Galactic Extinction failed")

        return catalog

    def correctFieldForGalacticExtinction(self, catalog, tractInfo):
        """Apply a per-field correction for Galactic Extinction using hard-wired values

        These numbers for E(B-V) are based on the Schlegel et al. 1998 (ApJ 500, 525, SFD98)
        dust maps and were obtained from:
        http://irsa.ipac.caltech.edu/applications/DUST/

        Note that the only fields included are the 5 tracts in the RC + RC2 datasets.
        This is just a placeholder until a per-object implementation is added in DM-13519
        """
        ebvValues = {"UD_COSMOS_9813": {"centerCoord": afwGeom.SpherePoint(150.25, 2.23, afwGeom.degrees),
                                        "EBmV": 0.0165},
                     "WIDE_VVDS_9796": {"centerCoord": afwGeom.SpherePoint(337.78, 0.74, afwGeom.degrees),
                                        "EBmV": 0.0748},
                     "WIDE_GAMMA15H_9615": {"centerCoord": afwGeom.SpherePoint(216.3, 0.74, afwGeom.degrees),
                                            "EBmV": 0.0281},
                     "WIDE_8766": {"centerCoord": afwGeom.SpherePoint(35.70, -3.72, afwGeom.degrees),
                                   "EBmV": 0.0246},
                     "WIDE_8767": {"centerCoord": afwGeom.SpherePoint(37.19, -3.72, afwGeom.degrees),
                                   "EBmV": 0.0268}}

        geFound = False
        for fieldName, geEntry in ebvValues.items():
            if tractInfo.contains(geEntry["centerCoord"]):
                ebvValue = ebvValues[fieldName]["EBmV"]
                geFound = True
                break
        if geFound:
            for ff in catalog.keys():
                if ff in self.config.extinctionCoeffs:
                    fluxKeys, errKeys = getFluxKeys(catalog[ff].schema)
                    galacticExtinction = ebvValue*self.config.extinctionCoeffs[ff]
                    self.log.info("Applying Per-Field Galactic Extinction correction A_{0:s} = {1:.3f}".
                                  format(ff, galacticExtinction))
                    factor = 10.0**(0.4*galacticExtinction)
                    for name, key in list(fluxKeys.items()) + list(errKeys.items()):
                        catalog[ff][key] *= factor
                    # Add column of Galactic Extinction value applied to the catalog
                    catalog[ff] = addIntOrFloatColumn(catalog[ff], [galacticExtinction, ], "A_" + str(ff),
                                                      "Galactic Extinction applied (based on SFD 1998 maps)")
                else:
                    self.log.warn("Do not have A_X/E(B-V) for filter {0:s}.  "
                                  "No Galactic Extinction correction applied for that filter".format(ff))
        else:
            self.log.warn("Do not have Galactic Extinction for tract {0:d} at {1:s}.  "
                          "No Galactic Extinction correction applied".
                          format(tractInfo.getId(), str(tractInfo.getCtrCoord())))
        return catalog

    def transformCatalogs(self, catalogs, transforms, flagsCats=None, hscRun=None):
        """
        Transform catalog entries according to the color transform given

        Parameters
        ----------
        catalogs : `dict` of `lsst.afw.table.source.source.SourceCatalog`s
           One dict entry per filter
        transforms : `dict` of `lsst.pipe.analysis.colorAnalysis.ColorTransform`s
           One dict entry per filter-dependent transform definition
        flagsCats : `dict` of `lsst.afw.table.source.source.SourceCatalog`s
           One dict entry per filter.  Source lists must be identical to those in catalogs.
           This is to provide a way to use a different catalog containing the flags of interest
           for source filtering (e.g. forced catalogs do not have all the flags defined in unforced
           catalogs, but the source lists are identical)
        hscRun : `str` or `NoneType`
           A string representing "HSCPIPE_VERSION" fits header if the data were processed with
           the (now obsolete, but old reruns still exist) "HSC stack", None otherwise
        """
        if flagsCats is None:
            flagsCats = catalogs

        template = list(catalogs.values())[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.values())

        mapper = afwTable.SchemaMapper(template.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        schema = mapper.getOutputSchema()

        for col in transforms:
            doAdd = True
            for ff in transforms[col].coeffs:
                if ff != "" and ff not in catalogs:
                    doAdd = False
            if doAdd:
                schema.addField(col, float, transforms[col].description + transforms[col].subDescription)
        schema.addField("numStarFlags", type=np.int32, doc="Number of times source was flagged as star")
        badKey = schema.addField("qaBad_flag", type="Flag", doc="Is this a bad source for color qa analyses?")
        schema.addField(self.fluxColumn, type=np.float64, doc="Flux from filter " + self.fluxFilter)

        # Copy basics (id, RA, Dec)
        new = afwTable.SourceCatalog(schema)
        new.reserve(num)
        new.extend(template, mapper)

        # Set transformed colors
        for col, transform in transforms.items():
            if col not in schema:
                continue
            value = np.ones(num)*transform.coeffs[""] if "" in transform.coeffs else np.zeros(num)
            for ff, coeff in transform.coeffs.items():
                if ff == "":  # Constant: already done
                    continue
                cat = catalogs[ff]
                mag = -2.5*np.log10(cat[self.fluxColumn])
                value += mag*coeff
            new[col][:] = value

        # Flag bad values
        bad = np.zeros(num, dtype=bool)
        for dataCat, flagsCat in zip(catalogs.values(), flagsCats.values()):
            if not checkIdLists(dataCat, flagsCat):
                raise RuntimeError(
                    "Catalog being used for flags does not have the same object list as the data catalog")
            for flag in self.flags:
                if flag in flagsCat.schema:
                    bad |= flagsCat[flag]
        # Can't set column for flags; do row-by-row
        for row, badValue in zip(new, bad):
            row.setFlag(badKey, bool(badValue))

        # Star/galaxy
        numStarFlags = np.zeros(num)
        for cat in catalogs.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)
        new["numStarFlags"][:] = numStarFlags

        new[self.fluxColumn][:] = catalogs[self.fluxFilter][self.fluxColumn]

        return new

    def plotGalacticExtinction(self, byFilterCats, filenamer, dataId, butler=None,
                               camera=None, tractInfo=None, patchList=None, hscRun=None, geLabel=None):
        for ff in byFilterCats:
            qMin = np.nanmean(byFilterCats[ff]["A_" + ff]) - 6.0*np.nanstd(byFilterCats[ff]["A_" + ff])
            qMax = np.nanmean(byFilterCats[ff]["A_" + ff]) + 6.0*np.nanstd(byFilterCats[ff]["A_" + ff])
            shortName = "galacticExtinction_" + ff
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(byFilterCats[ff], byFilterCats[ff]["A_" + ff],
                               "%s (%s)" % ("Galactic Extinction:  A_" + ff, "mag"),
                               shortName, self.config.analysis, flags=["galacticExtinction_flag"],
                               labeller=AllLabeller(), qMin=qMin, qMax=qMax, magThreshold=99.0,
                               ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                         zpLabel=geLabel, plotRunStats=False)

    def plotGalaxyColors(self, catalogs, filenamer, dataId):
        filters = set(catalogs.keys())
        if filters.issuperset(set(("HSC-G", "HSC-I"))):
            gg = catalogs["HSC-G"]
            ii = catalogs["HSC-I"]
            assert len(gg) == len(ii)
            mapperList = afwTable.SchemaMapper.join([gg.schema, ii.schema],
                                                    ["g_", "i_"])
            catalog = afwTable.BaseCatalog(mapperList[0].getOutputSchema())
            catalog.reserve(len(gg))
            for gRow, iRow in zip(gg, ii):
                row = catalog.addNew()
                row.assign(gRow, mapperList[0])
                row.assign(iRow, mapperList[1])

            catalog.writeFits("gi.fits")
            shortName = "galaxy-TEST"
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(catalog,
                               GalaxyColor("modelfit_CModel_flux", "slot_CalibFlux_flux", "g_", "i_"),
                               "(g-i)_cmodel - (g-i)_CalibFlux", shortName, self.config.analysis,
                               flags=["modelfit_CModel_flag", "slot_CalibFlux_flag"], prefix="i_",
                               labeller=OverlapsStarGalaxyLabeller("g_", "i_"),
                               qMin=-0.5, qMax=0.5,).plotAll(dataId, filenamer, self.log)

    def plotStarPrincipalColors(self, principalColCats, byFilterCats, filenamer, labeller, dataId,
                                butler=None, camera=None, tractInfo=None, patchList=None, hscRun=None,
                                geLabel=None):
        mags = {ff: -2.5*np.log10(byFilterCats[ff]["base_PsfFlux_flux"]) for ff in byFilterCats}
        unitStr = "mmag" if self.config.toMilli else "mag"
        for col, transform in self.config.transforms.items():
            if not transform.plot or col not in principalColCats.schema:
                continue
            if self.config.transforms == ivezicTransformsHSC:
                if col == "wPerp" or col == "xPerp":
                    colStr1, colStr2, colStr3 = "HSC-G", "HSC-R", "HSC-I"
                    filterStrList = ["g", "r", "i", ""]
                elif col == "yPerp":
                    colStr1, colStr2, colStr3 = "HSC-R", "HSC-I", "HSC-Z"
                    filterStrList = ["r", "i", "z", ""]
                else:
                    raise RuntimeError("Unknown transformation name: {:s}.  Either set transform.plot "
                                       "to False for that transform or provide accommodations for "
                                       "plotting it in the plotStarPrincipalColors function".format(col))
                xColor = catColors(colStr1, colStr2, mags)
                yColor = catColors(colStr2, colStr3, mags)
                filtersStr = filterStrList[0] + filterStrList[1] + filterStrList[2]
                xRange = (self.config.plotRanges[filtersStr + "X0"],
                          self.config.plotRanges[filtersStr + "X1"])
                yRange = (self.config.plotRanges[filtersStr + "Y0"],
                          self.config.plotRanges[filtersStr + "Y1"])
                paraCol = col[0] + "Para"
                principalColorStrs = []
                for pColStr in [paraCol, col]:
                    transformForStr = self.config.transforms[pColStr]
                    pColStr = makeEqnStr(pColStr, transformForStr.coeffs.values(), filterStrList)
                    principalColorStrs.append(pColStr)
                colorsInFitRange = ColorValueInFitRange(col, xColor, yColor,
                                                        transform.fitLineSlope, transform.fitLineUpperIncpt,
                                                        transform.fitLineLowerIncpt, unitScale=self.unitScale)
                colorsInPerpRange = ColorValueInPerpRange(col, transform.requireGreater,
                                                          transform.requireLess, unitScale=self.unitScale)
                colorsInRange = colorsInFitRange
            elif self.config.transforms == ivezicTransformsSDSS:
                colorsInRange = ColorValueInPerpRange(col, transform.requireGreater, transform.requireLess,
                                                      unitScale=self.unitScale)
            else:
                raise RuntimeError("Unknown transformation: {:s}".format(self.config.transforms))

            shortName = "color_" + col
            self.log.info("shortName = {:s}".format(shortName + transform.subDescription))
            self.AnalysisClass(principalColCats, colorsInRange, "%s (%s)" % (col + transform.subDescription,
                                                                             unitStr),
                               shortName, self.config.analysis, flags=["qaBad_flag"], labeller=labeller,
                               qMin=-0.2, qMax=0.2, magThreshold=self.config.analysis.magThreshold,
                               ).plotAll(dataId, filenamer, self.log, butler=butler, camera=camera,
                                         tractInfo=tractInfo, patchList=patchList, hscRun=hscRun,
                                         zpLabel=geLabel, plotRunStats=False, extraLabels=principalColorStrs)

            # Plot selections of stars for different criteria
            if self.config.transforms == ivezicTransformsHSC:
                filename = filenamer(dataId, description=filtersStr + fluxToPlotString("base_PsfFlux_flux"),
                                     style=col+"Selections")
                qaGood = np.logical_and(np.logical_not(principalColCats["qaBad_flag"]),
                                        principalColCats["numStarFlags"] >= 3)
                qaGood = np.logical_and(qaGood, mags[self.fluxFilter] < self.config.analysis.magThreshold)
                inFitGood = np.logical_and(np.isfinite(colorsInFitRange(principalColCats)), qaGood)
                inPerpGood = np.logical_and(np.isfinite(colorsInPerpRange(principalColCats)), qaGood)
                xColor = catColors(colStr1, colStr2, mags)
                yColor = catColors(colStr2, colStr3, mags)
                fig, axes = plt.subplots(1, 1)
                axes.tick_params(which="both", direction="in", labelsize=9)
                axes.set_xlim(*xRange)
                axes.set_ylim(*yRange)

                deltaX = abs(xRange[1] - xRange[0])
                deltaY = abs(yRange[1] - yRange[0])
                lineOffset = [0.15, 0.15]
                if col == "wPerp":
                    lineOffset = [0.30, 0.30]
                if col == "xPerp":
                    lineOffset = [0.60, 0.15]
                if col == "yPerp":
                    lineOffset = [0.30, 0.30]
                lineFitSlope = self.config.transforms[col[0] + "Fit"].coeffs[colStr1]
                lineFitIncpt = self.config.transforms[col[0] + "Fit"].coeffs[""]
                xLine = np.linspace(xRange[0] + lineOffset[0]*deltaX, xRange[1]-lineOffset[1]*deltaX, 100)
                yLineUpper = transform.fitLineUpperIncpt + transform.fitLineSlope*xLine
                yLineLower = transform.fitLineLowerIncpt + transform.fitLineSlope*xLine
                yLineFit = lineFitSlope*xLine + lineFitIncpt
                axes.plot(xLine, yLineUpper, "g--", alpha=0.5)
                axes.plot(xLine, yLineLower, "g--", alpha=0.5)
                axes.plot(xLine, yLineFit, "m--", alpha=0.5)

                ptSize = max(1, setPtSize(len(xColor)) - 2)

                axes.scatter(xColor[qaGood], yColor[qaGood], label="all", color="black", alpha=0.4,
                             marker="o", s=ptSize + 2)
                axes.scatter(xColor[inFitGood], yColor[inFitGood], label="inFit", color="blue",
                             marker="o", s=ptSize + 1)
                axes.scatter(xColor[inPerpGood], yColor[inPerpGood], label="inPerp", color="red",
                             marker="x", s=ptSize, lw=0.5)
                axes.set_xlabel(colStr1 + " $-$ " + colStr2)
                axes.set_ylabel(colStr2 + " $-$ " + colStr3, labelpad=-1)

                # Label total number of objects of each data type
                lenNumObj = max(len(str(len(xColor[qaGood]))), len(str(len(xColor[inFitGood]))),
                                len(str(len(xColor[inPerpGood]))))
                fdx = max((min(0.09*lenNumObj, 0.9), 0.42))
                xLoc, yLoc = xRange[0] + 0.03*deltaX, yRange[1] - 0.038*deltaY
                kwargs = dict(va="center", fontsize=8)
                axes.text(xLoc, yLoc, "NqaGood  =", ha="left", color="black", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[qaGood])) + " [" + self.fluxFilter +
                          " < " + str(self.config.analysis.magThreshold) + "]", ha="right", color="black",
                          **kwargs)
                yLoc -= 0.05*deltaY
                axes.text(xLoc, yLoc, "NinFitGood =", ha="left", color="blue", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[inFitGood])) + " [" + self.fluxFilter +
                          " < " + str(self.config.analysis.magThreshold) + "]", ha="right", color="blue",
                          **kwargs)
                yLoc -= 0.05*deltaY
                axes.text(xLoc, yLoc, "NinPerpGood =", ha="left", color="red", **kwargs)
                axes.text(xLoc + fdx*deltaX, yLoc, str(len(xColor[inPerpGood])) + " [" + self.fluxFilter +
                          " < " + str(self.config.analysis.magThreshold) + "]", ha="right", color="red",
                          **kwargs)
                if camera is not None:
                    labelCamera(camera, plt, axes, 0.5, 1.09)
                if geLabel is not None:
                    plotText(geLabel, plt, axes, 0.09, -0.11, color="green")
                if hscRun is not None:
                    axes.set_title("HSC stack run: " + hscRun, color="#800080")

                tractStr = "tract: {:d}".format(dataId["tract"])
                axes.annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                              fontsize=10, color="green")

                fig.savefig(filename, dpi=120)
                plt.close(fig)

    def plotStarColorColor(self, principalColCats, byFilterCats, filenamer, dataId, fluxColumn, butler=None,
                           camera=None, tractInfo=None, patchList=None, hscRun=None, geLabel=None):
        num = len(list(byFilterCats.values())[0])
        zp = 0.0
        mags = {ff: zp - 2.5*np.log10(byFilterCats[ff][fluxColumn]) for ff in byFilterCats}

        bad = np.zeros(num, dtype=bool)
        for cat in byFilterCats.values():
            for flag in self.flags:
                if flag in cat.schema:
                    bad |= cat[flag]

        bright = mags[self.fluxFilter] < self.config.analysis.magThreshold
        prettyBrightThreshold = self.config.analysis.magThreshold
        prettyBright = mags[self.fluxFilter] < prettyBrightThreshold

        # Determine number of filters object is classified as a star
        numStarFlags = np.zeros(num)
        for cat in byFilterCats.values():
            numStarFlags += np.where(cat[self.classificationColumn] < 0.5, 1, 0)

        # Select as a star if classified as such in self.config.fluxFilter
        isStarFlag = byFilterCats[self.fluxFilter][self.classificationColumn] < 0.5
        # Require stellar classification in self.fluxFilter and at least one other filter for fits
        good = isStarFlag & (numStarFlags >= 2) & ~bad & bright
        goodCombined = isStarFlag & (numStarFlags >= 2) & ~bad
        decentStars = isStarFlag & ~bad & prettyBright
        decentGalaxies = ~isStarFlag & ~bad & prettyBright

        # The combined catalog is only used in the Distance (from the poly fit) AnalysisClass plots
        combined = (self.transformCatalogs(byFilterCats, straightTransforms, hscRun=hscRun)[goodCombined].
                    copy(True))
        filters = set(byFilterCats.keys())
        goodMags = {ff: mags[ff][good] for ff in byFilterCats}
        decentStarsMag = mags[self.fluxFilter][decentStars]
        decentGalaxiesMag = mags[self.fluxFilter][decentGalaxies]
        unitStr = "mmag" if self.config.toMilli else "mag"
        fluxColStr = fluxToPlotString(fluxColumn)
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transformPerp = self.config.transforms["wPerp"]
            transformPara = self.config.transforms["wPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            filtersStr = "gri"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr + "-wFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            wPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, catColors("HSC-G", "HSC-R", mags, good),
                                             catColors("HSC-R", "HSC-I", mags, good),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["wPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(0.3, 1.12), yFitRange=(0.04, 0.5),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            transformPerp = self.config.transforms["xPerp"]
            transformPara = self.config.transforms["xPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            nameStr = filtersStr + fluxColStr + "-xFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            xPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, catColors("HSC-G", "HSC-R", mags, good),
                                             catColors("HSC-R", "HSC-I", mags, good),
                                             "g - r  [{0:s}]".format(fluxColStr),
                                             "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["xPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(1.05, 1.45), yFitRange=(0.78, 1.65),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale,
                                             closeToVertical=True)
            # Lower branch only; upper branch is noisy due to astrophysics
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [1.21, -0.55]
            fitLineLower = [0.21, -0.36]
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, catColors("HSC-G", "HSC-R", mags, good),
                                         catColors("HSC-R", "HSC-I", mags, good),
                                         "g - r  [{0:s}]".format(fluxColStr),
                                         "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=3,
                                         xFitRange=(0.23, 1.2), yFitRange=(0.05, 0.6),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               catColors("HSC-G", "HSC-R", mags, decentStars),
                               catColors("HSC-R", "HSC-I", mags, decentStars),
                               catColors("HSC-G", "HSC-R", mags, decentGalaxies),
                               catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                               decentStarsMag, decentGalaxiesMag,
                               "g - r  [{0:s}]".format(fluxColStr),
                               "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=(xRange[0], xRange[1] + 0.6), yRange=yRange,
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               geLabel=geLabel, unitScale=self.unitScale)
                colorColor4MagPlots(dataId, filenamer(dataId, description=nameStr, style="noFitMagBins"),
                                    self.log, catColors("HSC-G", "HSC-R", mags, decentStars),
                                    catColors("HSC-R", "HSC-I", mags, decentStars),
                                    catColors("HSC-G", "HSC-R", mags, decentGalaxies),
                                    catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                    decentStarsMag, decentGalaxiesMag,
                                    "g - r  [{0:s}]".format(fluxColStr),
                                    "r - i  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                    xRange=(xRange[0], xRange[1] + 0.6), yRange=yRange,
                                    magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                                    geLabel=geLabel, unitScale=self.unitScale)

            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("g", "r", "i", poly, unitScale=self.unitScale,
                                                            fitLineUpper=fitLineUpper,
                                                            fitLineLower=fitLineLower),
                               filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr), shortName,
                               self.config.analysis, flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               magThreshold=prettyBrightThreshold, labeller=NumStarLabeller(2),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}}),
                                         camera=camera, tractInfo=tractInfo, patchList=patchList,
                                         hscRun=hscRun, zpLabel=geLabel)
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            # Do a linear fit to regions defined in Ivezic transforms
            transformPerp = self.config.transforms["yPerp"]
            transformPara = self.config.transforms["yPara"]
            fitLineUpper = [transformPerp.fitLineUpperIncpt, transformPerp.fitLineSlope]
            fitLineLower = [transformPerp.fitLineLowerIncpt, transformPerp.fitLineSlope]
            filtersStr = "riz"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr + "-yFit"
            self.log.info("nameStr = {:s}".format(nameStr))
            yPerpFit = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                             self.log, catColors("HSC-R", "HSC-I", mags, good),
                                             catColors("HSC-I", "HSC-Z", mags, good),
                                             "r - i  [{0:s}]".format(fluxColStr),
                                             "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                             transformPerp=transformPerp, transformPara=transformPara,
                                             mags=goodMags, principalCol=principalColCats["yPerp"][good],
                                             xRange=xRange, yRange=yRange, order=1,
                                             xFitRange=(0.92, 2.02), yFitRange=(0.4, 0.83),
                                             fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                             magThreshold=self.config.analysis.magThreshold, camera=camera,
                                             hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            nameStr = filtersStr + fluxColStr
            fitLineUpper = [0.94, -0.27]
            fitLineLower = [0.046, -0.55]
            self.log.info("nameStr = {:s}".format(nameStr))
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, catColors("HSC-R", "HSC-I", mags, good),
                                         catColors("HSC-I", "HSC-Z", mags, good),
                                         "r - i  [{0:s}]".format(fluxColStr),
                                         "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=2,
                                         xFitRange=(0.0, 1.6), yFitRange=(-0.03, 0.7),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               catColors("HSC-R", "HSC-I", mags, decentStars),
                               catColors("HSC-I", "HSC-Z", mags, decentStars),
                               catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                               catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                               decentStarsMag, decentGalaxiesMag,
                               "r - i  [{0:s}]".format(fluxColStr),
                               "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               geLabel=geLabel, unitScale=self.unitScale)
                colorColor4MagPlots(dataId, filenamer(dataId, description=nameStr, style="noFitMagBins"),
                                    self.log, catColors("HSC-R", "HSC-I", mags, decentStars),
                                    catColors("HSC-I", "HSC-Z", mags, decentStars),
                                    catColors("HSC-R", "HSC-I", mags, decentGalaxies),
                                    catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                    decentStarsMag, decentGalaxiesMag,
                                    "r - i  [{0:s}]".format(fluxColStr),
                                    "i - z  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                    xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                    magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                                    geLabel=geLabel, unitScale=self.unitScale)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly, unitScale=self.unitScale,
                                                            fitLineUpper=fitLineUpper,
                                                            fitLineLower=fitLineLower),
                               filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr), shortName,
                               self.config.analysis, flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               magThreshold=prettyBrightThreshold, labeller=NumStarLabeller(2),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}}),
                                         camera=camera, tractInfo=tractInfo, patchList=patchList,
                                         hscRun=hscRun, zpLabel=geLabel)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            filtersStr = "izy"
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [0.56, -0.32]
            fitLineLower = [-0.014, -0.39]
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, catColors("HSC-I", "HSC-Z", mags, good),
                                         catColors("HSC-Z", "HSC-Y", mags, good),
                                         "i - z  [{0:s}]".format(fluxColStr),
                                         "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange, order=2,
                                         xFitRange=(-0.05, 0.8), yFitRange=(-0.06, 0.3),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               catColors("HSC-I", "HSC-Z", mags, decentStars),
                               catColors("HSC-Z", "HSC-Y", mags, decentStars),
                               catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                               catColors("HSC-Z", "HSC-Y", mags, decentGalaxies),
                               decentStarsMag, decentGalaxiesMag,
                               "i - z  [{0:s}]".format(fluxColStr),
                               "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               geLabel=geLabel, unitScale=self.unitScale)
                colorColor4MagPlots(dataId, filenamer(dataId, description=nameStr, style="noFitMagBins"),
                                    self.log, catColors("HSC-I", "HSC-Z", mags, decentStars),
                                    catColors("HSC-Z", "HSC-Y", mags, decentStars),
                                    catColors("HSC-I", "HSC-Z", mags, decentGalaxies),
                                    catColors("HSC-Z", "HSC-Y", mags, decentGalaxies),
                                    decentStarsMag, decentGalaxiesMag,
                                    "i - z  [{0:s}]".format(fluxColStr),
                                    "z - y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                    xRange=xRange, yRange=(yRange[0], yRange[1] + 0.2),
                                    magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                                    geLabel=geLabel, unitScale=self.unitScale)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("i", "z", "y", poly, unitScale=self.unitScale,
                                                            fitLineUpper=fitLineUpper,
                                                            fitLineLower=fitLineLower),
                               filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr), shortName,
                               self.config.analysis, flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               magThreshold=prettyBrightThreshold, labeller=NumStarLabeller(2),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}}),
                                         camera=camera, tractInfo=tractInfo, patchList=patchList,
                                         hscRun=hscRun, zpLabel=geLabel)

        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            filtersStr = "z9y"
            xRange = (self.config.plotRanges[filtersStr + "X0"],
                      self.config.plotRanges[filtersStr + "X1"])
            yRange = (self.config.plotRanges[filtersStr + "Y0"],
                      self.config.plotRanges[filtersStr + "Y1"])
            nameStr = filtersStr + fluxColStr
            self.log.info("nameStr = {:s}".format(nameStr))
            fitLineUpper = [0.20, -0.3]
            fitLineLower = [0.01, -0.29]
            poly = colorColorPolyFitPlot(dataId, filenamer(dataId, description=nameStr, style="fit"),
                                         self.log, catColors("HSC-Z", "NB0921", mags, good),
                                         catColors("NB0921", "HSC-Y", mags, good),
                                         "z-n921  [{0:s}]".format(fluxColStr),
                                         "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter,
                                         xRange=xRange, yRange=yRange,
                                         order=2, xFitRange=(-0.09, 0.16), yFitRange=(0.003, 0.18),
                                         fitLineUpper=fitLineUpper, fitLineLower=fitLineLower,
                                         magThreshold=self.config.analysis.magThreshold, camera=camera,
                                         hscRun=hscRun, geLabel=geLabel, unitScale=self.unitScale)
            # Make a color-color plot with both stars and galaxies, less pruning, and no fit
            if fluxColumn is not "base_PsfFlux_flux":
                self.log.info("nameStr: noFit ({1:s}) = {0:s}".format(nameStr, fluxColumn))
                colorColorPlot(dataId, filenamer(dataId, description=nameStr, style="noFit"), self.log,
                               catColors("HSC-Z", "NB0921", mags, decentStars),
                               catColors("NB0921", "HSC-Y", mags, decentStars),
                               catColors("HSC-Z", "NB0921", mags, decentGalaxies),
                               catColors("NB0921", "HSC-Y", mags, decentGalaxies),
                               decentStarsMag, decentGalaxiesMag,
                               "z-n921  [{0:s}]".format(fluxColStr),
                               "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                               xRange=xRange, yRange=(yRange[0] - 0.05, yRange[1] + 0.05),
                               magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                               geLabel=geLabel, unitScale=self.unitScale)
                colorColor4MagPlots(dataId, filenamer(dataId, description=nameStr, style="noFitMagBins"),
                                    self.log, catColors("HSC-Z", "NB0921", mags, decentStars),
                                    catColors("NB0921", "HSC-Y", mags, decentStars),
                                    catColors("HSC-Z", "NB0921", mags, decentGalaxies),
                                    catColors("NB0921", "HSC-Y", mags, decentGalaxies),
                                    decentStarsMag, decentGalaxiesMag,
                                    "z-n921  [{0:s}]".format(fluxColStr),
                                    "n921-y  [{0:s}]".format(fluxColStr), self.fluxFilter, fluxColStr,
                                    xRange=xRange, yRange=(yRange[0] - 0.05, yRange[1] + 0.05),
                                    magThreshold=prettyBrightThreshold, camera=camera, hscRun=hscRun,
                                    geLabel=geLabel, unitScale=self.unitScale)
            shortName = filtersStr + "Distance" + fluxColStr
            self.log.info("shortName = {:s}".format(shortName))
            self.AnalysisClass(combined, ColorColorDistance("z", "n921", "y", poly, unitScale=self.unitScale,
                                                            fitLineUpper=fitLineUpper,
                                                            fitLineLower=fitLineLower),
                               filtersStr + "Distance [%s] (%s)" % (fluxColStr, unitStr), shortName,
                               self.config.analysis, flags=["qaBad_flag"], qMin=-0.1, qMax=0.1,
                               magThreshold=prettyBrightThreshold, labeller=NumStarLabeller(2),
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={"star": {"stdev": 0.03*self.unitScale}}),
                                         camera=camera, tractInfo=tractInfo, patchList=patchList,
                                         hscRun=hscRun, zpLabel=geLabel)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None


def colorColorPolyFitPlot(dataId, filename, log, xx, yy, xLabel, yLabel, filterStr, transformPerp=None,
                          transformPara=None, mags=None, principalCol=None, xRange=None, yRange=None,
                          order=1, iterations=3, rej=3.0, xFitRange=None, yFitRange=None,
                          fitLineUpper=None, fitLineLower=None, numBins="auto", hscRun=None, geLabel=None,
                          logger=None, magThreshold=99.9, camera=None, unitScale=1.0, closeToVertical=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.46, bottom=0.15, left=0.11, right=0.96, top=0.9)
    axes[0].tick_params(which="both", direction="in", labelsize=9)
    axes[1].tick_params(which="both", direction="in", labelsize=9)

    good = np.logical_and(np.isfinite(xx), np.isfinite(yy))
    xx, yy = xx[good], yy[good]
    numGood = len(xx)
    fitP2 = None
    if mags is not None:
        mags = {ff: mags[ff][good] for ff in mags.keys()}
    if principalCol is not None:
        principalColor = principalCol[good].copy()*unitScale

    if xRange:
        axes[0].set_xlim(*xRange)
    else:
        xRange = (0.9*xx.min(), 1.1*xx.max())
    if yRange:
        axes[0].set_ylim(*yRange)
    else:
        yRange = (0.9*yy.min(), 1.1*yy.max())

    xLine = np.linspace(xRange[0], xRange[1], 1000)
    if fitLineUpper:
        yLineUpper = fitLineUpper[0] + fitLineUpper[1]*xLine
    if fitLineLower:
        yLineLower = fitLineLower[0] + fitLineLower[1]*xLine

    # Include vertical xFitRange for clipping points in the first iteration
    selectXRange = np.ones_like(xx, dtype=bool) if not xFitRange else ((xx > xFitRange[0]) &
                                                                       (xx < xFitRange[1]))
    # Include horizontal yFitRange for clipping points in the first iteration
    selectYRange = np.ones_like(xx, dtype=bool) if not yFitRange else ((yy > yFitRange[0]) &
                                                                       (yy < yFitRange[1]))
    selectUpper = np.ones_like(xx, dtype=bool) if not fitLineUpper else (yy <
                                                                         fitLineUpper[0] + fitLineUpper[1]*xx)
    selectLower = np.ones_like(xx, dtype=bool) if not fitLineLower else (yy >
                                                                         fitLineLower[0] + fitLineLower[1]*xx)

    # Pad vertical and horizontal fit ranges for use after the first fit iteration
    if xFitRange:
        xMinPad = xFitRange[0] - 0.07*(xFitRange[1] - xFitRange[0])
        xMaxPad = xFitRange[1] + 0.07*(xFitRange[1] - xFitRange[0])
    if yFitRange:
        yMinPad = yFitRange[0] - 0.07*(yFitRange[1] - yFitRange[0])
        yMaxPad = yFitRange[1] + 0.07*(yFitRange[1] - yFitRange[0])

    select = np.ones_like(xx, dtype=bool)
    for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
        select &= sel

    keep = np.ones_like(xx, dtype=bool)
    # Perform a polynomial fit using np.polyfit to use as an initial guess for the Orthoganl Regression
    if closeToVertical:
        # Force the initial guess for near-vertical distributions (np.polyfit cannot handle vertical fits)
        keep &= select
        poly = [10.0, -10.0*(xFitRange[0] + (xFitRange[1]-xFitRange[0])/3.0)]
    else:
        for ii in range(iterations):
            keep &= select
            poly = np.polyfit(xx[keep], yy[keep], order)
            dy = yy - np.polyval(poly, xx)
            q1, q3 = np.percentile(dy[keep], [25, 75])
            clip = rej*0.74*(q3 - q1)
            keep = np.logical_not(np.abs(dy) > clip)
            # After the first iteration, reset the vertical and horizontal clipping to be less restrictive
            if ii == 0:
                selectXRange = selectXRange if not xFitRange else ((xx > xMinPad) & (xx < xMaxPad))
                selectYRange = selectYRange if not yFitRange else ((yy > yMinPad) & (yy < yMaxPad))
                for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
                    select &= sel

        log.info("Number of iterations in polynomial fit: {:d}".format(ii + 1))
        keep &= select
        nKeep = np.sum(keep)
        if nKeep < order:
            raise RuntimeError(
                "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".format(nKeep, order))

        poly = np.polyfit(xx[keep], yy[keep], order)

    # Calculate the point density
    xyKeep = np.vstack([xx[keep], yy[keep]])
    zKeep = scipyStats.gaussian_kde(xyKeep)(xyKeep)
    xyOther = np.vstack([xx[~keep], yy[~keep]])
    zOther = scipyStats.gaussian_kde(xyOther)(xyOther)
    idxHighDensity = np.argmax(zKeep)
    xHighDensity = xx[keep][idxHighDensity]
    yHighDensity = yy[keep][idxHighDensity]
    log.info("Highest Density point x, y: {0:.2f} {1:.2f}".format(xHighDensity, yHighDensity))

    initialGuess = list(reversed(poly))
    keepOdr = keep.copy()
    orthRegCoeffs = orthogonalRegression(xx[keepOdr], yy[keepOdr], order, initialGuess)
    for ii in range(iterations - 1):
        initialGuess = list(reversed(orthRegCoeffs))
        dy = yy - np.polyval(orthRegCoeffs, xx)
        q1, q3 = np.percentile(dy[keepOdr], [25, 75])
        clip = rej*0.74*(q3 - q1)
        keepOdr = np.logical_not(np.abs(dy) > clip) & np.isfinite(xx) & np.isfinite(yy)
        # After the first iteration, reset the vertical and horizontal clipping to be less restrictive
        if ii == 0:
            selectXRange = selectXRange if not xFitRange else ((xx > xMinPad) & (xx < xMaxPad))
            selectYRange = selectYRange if not yFitRange else ((yy > yMinPad) & (yy < yMaxPad))
            for sel in [selectXRange, selectYRange, selectUpper, selectLower]:
                keepOdr &= sel
        nKeepOdr = np.sum(keepOdr)
        if nKeepOdr < order:
            raise RuntimeError(
                "Not enough good data points ({0:d}) for polynomial fit of order {1:d}".
                format(nKeepOdr, order))
        orthRegCoeffs = orthogonalRegression(xx[keepOdr], yy[keepOdr], order, initialGuess)
    yOrthLine = np.polyval(orthRegCoeffs, xLine)

    # Find index where poly and fit range intersect -- to calculate the local slopes of the fit to make
    # sure it is close to the fitLines (log a warning if they are not within 5%)
    message = ("{0:s} branch of the hard-coded lines for object selection does not cross the "
               "current polynomial fit.\nUsing the xFitRange {1:} to compute the local slope")
    try:
        crossIdxUpper = (np.argwhere(np.diff(np.sign(yOrthLine - yLineUpper)) != 0).reshape(-1) + 0)[0]
    except Exception:
        log.warnf(message, "Upper", xFitRange[1])
        crossIdxUpper = (np.abs(xLine - xFitRange[1])).argmin()
    try:
        crossIdxLower = (np.argwhere(np.diff(np.sign(yOrthLine - yLineLower)) != 0).reshape(-1) + 0)[0]
    except Exception:
        log.warnf(message, "Lower", xFitRange[0])
        crossIdxLower = (np.abs(xLine - xFitRange[0])).argmin()

    # Compute the slope of the two pixels +/-1% of line length from crossing point
    yOffset = int(0.01*len(yOrthLine))
    mUpper = ((yOrthLine[crossIdxUpper + yOffset] - yOrthLine[crossIdxUpper - yOffset])/
              (xLine[crossIdxUpper + yOffset] - xLine[crossIdxUpper - yOffset]))
    mLower = ((yOrthLine[crossIdxLower + yOffset] - yOrthLine[crossIdxLower - yOffset])/
              (xLine[crossIdxLower + yOffset] - xLine[crossIdxLower - yOffset]))
    bUpper = -yOrthLine[crossIdxUpper] - mUpper*xLine[crossIdxUpper]
    bLower = -yOrthLine[crossIdxLower] - mLower*xLine[crossIdxLower]
    # Rotate slope by 90 degrees for source selection lines
    mUpper = -1.0/mUpper
    mLower = -1.0/mLower
    bUpper = yOrthLine[crossIdxUpper] - mUpper*xLine[crossIdxUpper]
    bLower = yOrthLine[crossIdxLower] - mLower*xLine[crossIdxLower]
    message = ("{0:s} branch of the hard-coded lines for object selection does not match the local\nslope of "
               "the current polynomial fit.\n  --> Consider replacing {1:} with [{2:.3f}, {3:.3f}] "
               "(Line crosses fit at x = {4:.2f})")
    if (abs(200*(fitLineUpper[0] - bUpper)/(fitLineUpper[0] + bUpper)) > 5.0 or
            abs(200*(fitLineUpper[1] - mUpper)/(fitLineUpper[1] + mUpper)) > 5.0):
        log.warn(message.format("Upper", fitLineUpper, bUpper, mUpper, xLine[crossIdxUpper]))
    if (abs(200*(fitLineLower[0] - bLower)/(fitLineLower[0] + bLower)) > 5.0 or
            abs(200*(fitLineLower[1] - mLower)/(fitLineLower[1] + mLower)) > 5.0):
        log.warn(message.format("Lower", fitLineLower, bLower, mLower, xLine[crossIdxLower]))
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])

    # Find some sensible plotting limits for the P1 line fit
    frac = 0.26
    crossIdxMid = crossIdxLower + int(0.5*(crossIdxUpper - crossIdxLower))
    fracIdx = min(int(frac*len(xLine)), len(xLine) - 1 - crossIdxMid)
    yAtCrossIdxMid = yOrthLine[crossIdxMid]
    midCrossPlusFracIdx = np.abs(yOrthLine - (yAtCrossIdxMid + frac*deltaY)).argmin()
    yAtFracIdx = yOrthLine[crossIdxMid + fracIdx]
    idxP1 = (crossIdxMid + fracIdx) if yAtFracIdx < (yAtCrossIdxMid + frac*deltaY) else midCrossPlusFracIdx
    deltaIdxP1 = idxP1 - crossIdxMid
    xP1Line = xLine[crossIdxMid - deltaIdxP1:crossIdxMid + deltaIdxP1]
    yP1Line = yOrthLine[crossIdxMid - deltaIdxP1:crossIdxMid + deltaIdxP1]
    axes[0].plot(xP1Line, yP1Line, "g--", lw=0.75)

    kwargs = dict(s=3, marker="o", lw=0, alpha=0.4)
    axes[0].scatter(xx[~keep], yy[~keep], c=zOther, cmap="gray", label="other", **kwargs)
    axes[0].scatter(xx[keep], yy[keep], c=zKeep, cmap="jet", label="used", **kwargs)
    axes[0].set_xlabel(xLabel)
    axes[0].set_ylabel(yLabel, labelpad=-1)

    mappableKeep = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=zKeep.min(), vmax=zKeep.max()))
    mappableKeep._A = []        # fake up the array of the scalar mappable. Urgh...
    caxKeep = plt.axes([0.46, 0.15, 0.022, 0.75])
    cbKeep = plt.colorbar(mappableKeep, cax=caxKeep)
    cbKeep.ax.tick_params(labelsize=6)
    labelPadShift = len(str(zKeep.max()//10)) if zKeep.max()//10 > 0 else 0
    cbKeep.set_label("Number Density", rotation=270, labelpad=-15 - labelPadShift, fontsize=7)

    if xFitRange:
        # Shade region outside xFitRange
        kwargs = dict(facecolor="k", edgecolor="none", alpha=0.05)
        axes[0].axvspan(axes[0].get_xlim()[0], xFitRange[0], **kwargs)
        axes[0].axvspan(xFitRange[1], axes[0].get_xlim()[1], **kwargs)
        # Looser range after fisrt iteration
        axes[0].axvspan(axes[0].get_xlim()[0], xMinPad, **kwargs)
        axes[0].axvspan(xMaxPad, axes[0].get_xlim()[1], **kwargs)
    if yFitRange:
        # Shade region outside yFitRange
        xMin = abs(xFitRange[0] - xRange[0])/deltaX if xFitRange else 1
        xMax = abs(xFitRange[1] - xRange[0])/deltaX if xFitRange else 1
        axes[0].axhspan(axes[0].get_ylim()[0], yFitRange[0], xmin=xMin, xmax=xMax, **kwargs)
        axes[0].axhspan(yFitRange[1], axes[0].get_ylim()[1], xmin=xMin, xmax=xMax, **kwargs)
        # Looser range after fisrt iteration
        xMin = abs(xMinPad - xRange[0])/deltaX if xFitRange else 1
        xMax = abs(xMaxPad - xRange[0])/deltaX if xFitRange else 1
        axes[0].axhspan(axes[0].get_ylim()[0], yMinPad, xmin=xMin, xmax=xMax, **kwargs)
        axes[0].axhspan(yMaxPad, axes[0].get_ylim()[1], xmin=xMin, xmax=xMax, **kwargs)
    if fitLineUpper:
        scaleLine = 0.05*deltaX*max(1.0, min(3.0, abs(1.0/fitLineUpper[1])))
        xLineUpper = np.linspace(xLine[crossIdxUpper] - scaleLine, xLine[crossIdxUpper] + scaleLine, 100)
        yLineUpper = fitLineUpper[0] + fitLineUpper[1]*xLineUpper
        axes[0].plot(xLineUpper, yLineUpper, "r--")
    if fitLineLower:
        scaleLine = 0.05*deltaX*max(1.0, min(3.0, abs(1.0/fitLineLower[1])))
        xLineLower = np.linspace(xLine[crossIdxLower] - scaleLine, xLine[crossIdxLower] + scaleLine, 100)
        yLineLower = fitLineLower[0] + fitLineLower[1]*xLineLower
        axes[0].plot(xLineLower, yLineLower, "r--")

    # Label total number of objects of each data type
    kwargs = dict(va="center", fontsize=7)
    lenNumObj = max(len(str(len(xx[keepOdr]))), len(str(len(xx))))
    fdx = max((min(0.08*lenNumObj, 0.6), 0.32))
    xLoc, yLoc = xRange[0] + 0.05*deltaX, yRange[1] - 0.036*deltaY
    axes[0].text(xLoc, yLoc, "N$_{used }$ =", ha="left", color="blue", **kwargs)
    axes[0].text(xLoc + fdx*deltaX, yLoc, str(len(xx[keepOdr])), ha="right", color="blue", **kwargs)
    axes[0].text(xRange[1] - 0.03*deltaX, yLoc, " [" + filterStr + " < " + str(magThreshold) + "]",
                 ha="right", color="blue", **kwargs)
    yLoc -= 0.044*(yRange[1] - yRange[0])
    axes[0].text(xLoc, yLoc, "N$_{total}$ =", ha="left", color="black", **kwargs)
    axes[0].text(xLoc + fdx*deltaX, yLoc, str(len(xx)), ha="right", color="black", **kwargs)

    unitStr = "mmag" if unitScale == 1000 else "mag"
    axes[1].set_xlabel("Distance to polynomial fit ({:s})".format(unitStr))
    axes[1].set_ylabel("Number")
    axes[1].set_yscale("log", nonposy="clip")

    # Label orthogonal polynomial fit parameters to 2 decimal places
    xLoc = xRange[0] + 0.045*deltaX
    polyColor = "green"
    polyFit = orthRegCoeffs
    polyStr = "odr"
    kept = keepOdr

    polyStr = "y$_{" + polyStr + "}$" + " = {:.2f}".format(polyFit[len(polyFit) - 1])
    for i in range(1, len(polyFit)):
        index = len(polyFit) - 1 - i
        exponent = "$^{" + str(i) + "}$" if i > 1 else ""
        coeffStr = "{:.2f}".format(abs(polyFit[index])) + "x" + exponent
        plusMinus = " $-$ " if polyFit[index] < 0.0 else " + "
        if i == 0:
            polyStr += plusMinus.strip(" ") + coeffStr
        else:
            polyStr += plusMinus + coeffStr
    yLoc -= 0.05*deltaY
    kwargs = dict(ha="left", va="center", color=polyColor)
    axes[0].text(xLoc, yLoc, polyStr, fontsize=8, **kwargs)

    if "odr" in polyStr and order == 1:
        m, b = polyFit[0], polyFit[1]
        # Closest point on line to highest density point
        xHighDensity0 = (xHighDensity + m*(yHighDensity - b))/(m**2.0 + 1.0)
        yHighDensity0 = (m*(xHighDensity + m*yHighDensity) + b)/(m**2.0 + 1.0)
        bP2 = yHighDensity0 + (1.0/m)*xHighDensity0
        yP2Line = (-1.0/m)*xLine + bP2
        # Find some sensible plotting limits for the P2 line fit
        frac = 0.15
        idxHd = np.abs(yP2Line - yHighDensity0).argmin()
        idxFrac = idxHd - int(frac*len(xLine))
        fracIdx = max(idxHd - int(frac*len(xLine)), 0)
        yAtIdxFrac = yP2Line[idxFrac]
        idxHdPlusFrac = np.abs(yP2Line - (yHighDensity0 + frac*deltaY)).argmin()
        yAtHdPlusFrac = yP2Line[idxHdPlusFrac]
        idxP2 = idxFrac if yAtIdxFrac < yAtHdPlusFrac else idxHdPlusFrac
        deltaIdxP2 = idxHd - idxP2
        xP2Line = xLine[idxHd - deltaIdxP2:idxHd + deltaIdxP2]
        yP2Line = yP2Line[idxHd - deltaIdxP2:idxHd + deltaIdxP2]
        axes[0].plot(xP2Line, yP2Line, "g--", lw=0.75)
        plotText("P2$_{fit}$", plt, axes[0], xP2Line[0] - 0.022*deltaX, yP2Line[0] + 0.02*deltaY,
                 fontSize=8, color="green", coordSys="data")
        plotText("P1$_{fit}$", plt, axes[0], xP1Line[0] - 0.07*deltaX, yP1Line[0] + 0.01*deltaY,
                 fontSize=8, color="green", coordSys="data")

        # Derive Ivezic P2 and P1 equations based on linear fit and highest density position (where P1 = 0)
        pColCoeffs = p2p1CoeffsFromLinearFit(m, b, xHighDensity0, yHighDensity0)

        perpIndex = filename.find("Fit-fit")
        perpIndexStr = filename[perpIndex - 1:perpIndex]
        if perpIndexStr in ("w", "x"):
            perpFilters = ["g", "r", "i", ""]
        elif perpIndexStr == "y":
            perpFilters = ["r", "i", "z", ""]
        else:
            raise RuntimeError("Unknown Principal Color: {0:s}Perp".format(perpIndexStr))

        log.info("{0:s}Perp: P1/P2 origin x, y: {1:.2f} {2:.2f}".format(perpIndexStr,
                                                                        xHighDensity0, yHighDensity0))

        paraStr = "{0:s}Para{1:s}".format(perpIndexStr, "$_{fit}$")
        paraStr = makeEqnStr(paraStr, pColCoeffs.p1Coeffs, perpFilters)
        perpStr = "{0:s}Perp{1:s}".format(perpIndexStr, "$_{fit}$")
        perpStr = makeEqnStr(perpStr, pColCoeffs.p2Coeffs, perpFilters)

        # Also label plot with hardwired numbers
        principalColorStrs = []
        for transform, pCol in zip([transformPerp, transformPara],
                                   [perpIndexStr + "Perp", perpIndexStr[0:1] + "Para"]):
            principalColorStr = "{0:s}{1:s}".format(pCol, "$_{wired}$")
            principalColorStr = makeEqnStr(principalColorStr, transform.coeffs.values(), perpFilters)
            principalColorStrs.append(principalColorStr)

        xLoc = xRange[1] - 0.03*deltaX
        yLoc -= 0.05*deltaY
        axes[0].text(xLoc, yLoc, perpStr, fontsize=6, ha="right", va="center", color="magenta")
        yLoc -= 0.04*deltaY
        axes[0].text(xLoc, yLoc, principalColorStrs[0], fontsize=6, ha="right", va="center",
                     color="blue", alpha=0.8)
        yLoc -= 0.05*deltaY
        axes[0].text(xLoc, yLoc, paraStr, fontsize=6, ha="right", va="center", color="magenta")
        yLoc -= 0.04*deltaY
        axes[0].text(xLoc, yLoc, principalColorStrs[1], fontsize=6, ha="right", va="center",
                     color="blue", alpha=0.8)
        log.info("{0:s}".format("".join(x for x in perpStr if x not in "{}$")))
        log.info("{0:s}".format("".join(x for x in paraStr if x not in "{}$")))

        # Compute fitted P2 for each object
        if transform is not None:
            fitP2 = np.ones(numGood)*pColCoeffs.p2Coeffs[3]
            for i, ff in enumerate(transform.coeffs.keys()):
                if ff != "":
                    fitP2 += mags[ff]*pColCoeffs.p2Coeffs[i]
            fitP2 *= unitScale

    # Determine quality of locus
    distance2 = []
    polyFit = np.poly1d(polyFit)
    polyDeriv = np.polyder(polyFit)
    for x, y in zip(xx[kept], yy[kept]):
        roots = np.roots(np.poly1d((1, -x)) + (polyFit - y)*polyDeriv)
        distance2.append(min(
            distanceSquaredToPoly(x, y, np.real(rr), polyFit) for rr in roots if np.real(rr) == rr))
    distance = np.sqrt(distance2)
    distance *= np.where(yy[kept] >= polyFit(xx[kept]), 1.0, -1.0)
    distance *= unitScale
    q1, median, q3 = np.percentile(distance, [25, 50, 75])
    good = np.logical_not(np.abs(distance - median) > 3.0*0.74*(q3 - q1))
    mean = distance[good].mean()
    stdDev = distance[good].std()
    rms = np.sqrt(np.mean(distance[good]**2))
    # Get rid of LaTeX-specific characters for log message printing
    log.info("Polynomial fit: {:2}".format("".join(x for x in polyStr if x not in "{}$")))
    log.info(("Statistics from {0:} of Distance to polynomial ({9:s}): {7:s}\'star\': " +
              "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f}; clip={6:.4f})" +
              "{8:s}").format(dataId, mean, stdDev, len(xx[keep]), len(xx), np.median(distance[good]),
                              3.0*0.74*(q3 - q1), "{", "}", unitStr))
    meanStr = "mean = {0:5.2f}".format(mean)
    stdStr = "  std = {0:5.2f}".format(stdDev)
    rmsStr = "  rms = {0:5.2f}".format(rms)

    count, bins, ignored = axes[1].hist(distance[good], bins=numBins, range=(-4.0*stdDev, 4.0*stdDev),
                                        normed=True, color=polyColor, alpha=0.5)
    axes[1].plot(bins, 1/(stdDev*np.sqrt(2*np.pi))*np.exp(-(bins-mean)**2/(2*stdDev**2)),
                 color=polyColor)
    axes[1].axvline(x=mean, color=polyColor, linestyle=":")
    kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color=polyColor)
    axes[1].annotate(meanStr, xy=(0.34, 0.965), **kwargs)
    axes[1].annotate(stdStr, xy=(0.34, 0.93), **kwargs)
    axes[1].annotate(rmsStr, xy=(0.34, 0.895), **kwargs)

    axes[1].axvline(x=0.0, color="black", linestyle="--")
    tractStr = "tract: {:d}".format(dataId["tract"])
    axes[1].annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                     fontsize=10, color="green")

    # Plot hardwired principal color distributions
    if principalCol is not None:
        pCmean = principalColor[kept].mean()
        pCstdDev = principalColor[kept].std()
        count, nBins, ignored = axes[1].hist(principalColor[kept], bins=bins, range=(-4.0*stdDev, 4.0*stdDev),
                                             normed=True, color="blue", alpha=0.5)
        axes[1].plot(bins, 1/(pCstdDev*np.sqrt(2*np.pi))*np.exp(-(bins-pCmean)**2/(2*pCstdDev**2)),
                     color="blue")
        axes[1].axvline(x=pCmean, color="blue", linestyle=":")
        pCmeanStr = "{0:s}{1:s} = {2:5.2f}".format(perpStr[0:5], "$_{wired}$", pCmean)
        pCstdStr = "  std = {0:5.2f}".format(pCstdDev)
        kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color="blue")
        axes[1].annotate(pCmeanStr, xy=(0.97, 0.965), **kwargs)
        axes[1].annotate(pCstdStr, xy=(0.97, 0.93), **kwargs)
        log.info(("Statistics from {0:} of {9:s}Perp_wired ({8:s}): {6:s}\'star\': " +
                  "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f})" +
                  "{7:s}").format(dataId, pCmean, pCstdDev, len(principalColor[kept]), len(principalColor),
                                  np.median(principalColor[kept]), "{", "}", unitStr, perpIndexStr))
    # Plot fitted principal color distributions
    if fitP2 is not None:
        fitP2mean = fitP2[kept].mean()
        fitP2stdDev = fitP2[kept].std()
        count, nBins, ignored = axes[1].hist(fitP2[kept], bins=bins, range=(-4.0*stdDev, 4.0*stdDev),
                                             normed=True, color="magenta", alpha=0.5)
        axes[1].plot(bins, 1/(fitP2stdDev*np.sqrt(2*np.pi))*np.exp(-(bins-fitP2mean)**2/(2*fitP2stdDev**2)),
                     color="magenta")
        axes[1].axvline(x=fitP2mean, color="magenta", linestyle=":")
        fitP2meanStr = "{0:s}{1:s} = {2:5.2f}".format(perpStr[0:5], "$_{fit}$", fitP2mean)
        fitP2stdStr = "  std = {0:5.2f}".format(fitP2stdDev)
        kwargs = dict(xycoords="axes fraction", ha="right", va="center", fontsize=7, color="magenta")
        axes[1].annotate(fitP2meanStr, xy=(0.97, 0.895), **kwargs)
        axes[1].annotate(fitP2stdStr, xy=(0.97, 0.86), **kwargs)
        log.info(("Statistics from {0:} of {9:s}Perp_fit ({8:s}): {6:s}\'star\': " +
                  "Stats(mean={1:.4f}; stdev={2:.4f}; num={3:d}; total={4:d}; median={5:.4f})" +
                  "{7:s}").format(dataId, fitP2mean, fitP2stdDev, len(fitP2[kept]), len(fitP2),
                                  np.median(fitP2[kept]), "{", "}", unitStr, perpIndexStr))

    axes[1].set_ylim(axes[1].get_ylim()[0], axes[1].get_ylim()[1]*2.5)

    if camera is not None:
        labelCamera(camera, plt, axes[0], 0.5, 1.04)
    if geLabel is not None:
        plotText(geLabel, plt, axes[0], 0.16, -0.15, fontSize=10, color="green")
    if hscRun is not None:
        axes[0].set_title("HSC stack run: " + hscRun, color="#800080")

    fig.savefig(filename, dpi=120)
    plt.close(fig)

    return orthRegCoeffs


def colorColorPlot(dataId, filename, log, xStars, yStars, xGalaxies, yGalaxies, magStars, magGalaxies,
                   xLabel, yLabel, filterStr, fluxColStr, xRange=None, yRange=None, hscRun=None,
                   geLabel=None, logger=None, magThreshold=99.9, camera=None, unitScale=1.0):
    fig, axes = plt.subplots(1, 1)
    axes.tick_params(which="both", direction="in", labelsize=9)

    if xRange:
        axes.set_xlim(*xRange)
    else:
        xRange = (0.9*xStars.min(), 1.1*xStars.max())
    if yRange:
        axes.set_ylim(*yRange)

    vMin = min(magStars.min(), magGalaxies.min())
    vMax = min(magStars.max(), magGalaxies.max())

    ptSize = max(1, setPtSize(len(xGalaxies)) - 2)

    kwargs = dict(s=ptSize, marker="o", lw=0, vmin=vMin, vmax=vMax)
    axes.scatter(xGalaxies, yGalaxies, c=magGalaxies, cmap="autumn", label="galaxies", **kwargs)
    axes.scatter(xStars, yStars, c=magStars, cmap="winter", label="stars", **kwargs)
    axes.set_xlabel(xLabel)
    axes.set_ylabel(yLabel, labelpad=-1)

    # Label total number of objects of each data type
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])
    lenNumObj = max(len(str(len(xStars))), len(str(len(xGalaxies))))
    fdx = max((min(0.095*lenNumObj, 0.9), 0.42))
    xLoc, yLoc = xRange[0] + 0.03*deltaX, yRange[1] - 0.038*deltaY
    kwargs = dict(va="center", fontsize=8)
    axes.text(xLoc, yLoc, "Ngals  =", ha="left", color="red", **kwargs)
    axes.text(xLoc + fdx*deltaX, yLoc, str(len(xGalaxies)) +
              " [" + filterStr + " < " + str(magThreshold) + "]", ha="right", color="red", **kwargs)
    axes.text(xLoc, 0.94*yLoc, "Nstars =", ha="left", color="blue", **kwargs)
    axes.text(xLoc + fdx*deltaX, 0.94*yLoc, str(len(xStars)) +
              " [" + filterStr + " < " + str(magThreshold) + "]", ha="right", color="blue", **kwargs)
    if camera is not None:
        labelCamera(camera, plt, axes, 0.5, 1.09)
    if geLabel is not None:
        plotText(geLabel, plt, axes, 0.09, -0.11, color="green")
    if hscRun is not None:
        axes.set_title("HSC stack run: " + hscRun, color="#800080")

    tractStr = "tract: {:d}".format(dataId["tract"])
    axes.annotate(tractStr, xy=(0.5, 1.04), xycoords="axes fraction", ha="center", va="center",
                  fontsize=10, color="green")

    mappableStars = plt.cm.ScalarMappable(cmap="winter_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableStars._A = []        # fake up the array of the scalar mappable. Urgh...
    cbStars = plt.colorbar(mappableStars, aspect=14, pad=-0.09)
    cbStars.ax.tick_params(labelsize=8)
    cbStars.set_label(filterStr + " [" + fluxColStr + "]: stars", rotation=270, labelpad=-24, fontsize=9)
    mappableGalaxies = plt.cm.ScalarMappable(cmap="autumn_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableGalaxies._A = []      # fake up the array of the scalar mappable. Urgh...
    cbGalaxies = plt.colorbar(mappableGalaxies, aspect=14)
    cbGalaxies.set_ticks([])
    cbGalaxies.set_label(filterStr + " [" + fluxColStr + "]: galaxies", rotation=270, labelpad=-6, fontsize=9)

    fig.savefig(filename, dpi=120)
    plt.close(fig)

    return None


def colorColor4MagPlots(dataId, filename, log, xStars, yStars, xGalaxies, yGalaxies, magStars, magGalaxies,
                        xLabel, yLabel, filterStr, fluxColStr, xRange=None, yRange=None, hscRun=None,
                        geLabel=None, logger=None, magThreshold=99.9, camera=None, unitScale=1.0):

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0, bottom=0.1, right=0.82, top=0.9)

    xRange = ((xRange[0] + 0.01, xRange[1] - 0.01) if xRange is not None
              else (0.9*xStars.min(), 1.1*xStars.max()))
    yRange = ((yRange[0] + 0.01, yRange[1] - 0.01) if yRange is not None
              else (0.9*yStars.min(), 1.1*yStars.max()))
    deltaX = abs(xRange[1] - xRange[0])
    deltaY = abs(yRange[1] - yRange[0])
    vMin = round(min(magStars.min(), magGalaxies.min()) - 0.05, 1)
    vMax = round(max(magStars.max(), magGalaxies.max()) + 0.05, 1)

    ptSize = max(1, setPtSize(len(xGalaxies)) - 2)

    # Divide stars and galaxies into 4 magnitude bins
    binEdges = np.hstack((vMin, np.arange(magThreshold - 3, magThreshold, 1)))
    binIdxStars = np.digitize(magStars, binEdges)
    binIdxGalaxies = np.digitize(magGalaxies, binEdges)
    # The following is for ease of printing the bin ranges in the following loop
    binEdges = [bin for bin in binEdges]
    binEdges.append(magThreshold)

    for i, ax in enumerate(axes.flat[:]):
        index = 4 - i
        ax.tick_params(which="both", direction="in", labelsize=7)
        ax.set_xlim(*xRange)
        ax.set_ylim(*yRange)

        kwargs = dict(s=ptSize, marker="o", lw=0, vmin=vMin, vmax=vMax)
        ax.scatter(xGalaxies[binIdxGalaxies == index], yGalaxies[binIdxGalaxies == index],
                   c=magGalaxies[binIdxGalaxies == index], cmap="autumn", label="galaxies", **kwargs)
        ax.scatter(xStars[binIdxStars == index], yStars[binIdxStars == index],
                   c=magStars[binIdxStars == index], cmap="winter", label="stars", **kwargs)
        if i in (2, 3):
            ax.set_xlabel(xLabel)
        if i in (0, 2):
            ax.set_ylabel(yLabel)

        # Label total number of objects of each data type
        xLoc, yLoc = xRange[0] + 0.05*deltaX, yRange[1] - 0.06*deltaY
        kwargs = dict(va="center", fontsize=7)
        ax.text(xLoc, yLoc, "Ngals  =", ha="left", color="red", **kwargs)
        ax.text(xRange[1] - 0.03*deltaX, yLoc, str(len(xGalaxies[binIdxGalaxies == index])) +
                " [" + str(binEdges[index - 1]) + " <= " + filterStr + " < " + str(binEdges[index]) + "]",
                ha="right", color="red", **kwargs)
        ax.text(xLoc, 0.92*yLoc, "Nstars =", ha="left", va="center", fontsize=7, color="blue")
        ax.text(xRange[1] - 0.03*deltaX, 0.92*yLoc, str(len(xStars[binIdxStars == index])) +
                " [" + str(binEdges[index - 1]) + " <= " + filterStr + " < " + str(binEdges[index]) + "]",
                ha="right", color="blue", **kwargs)

    mappableStars = plt.cm.ScalarMappable(cmap="winter_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableStars._A = []        # fake up the array of the scalar mappable. Urgh...
    caxStars = plt.axes([0.88, 0.1, 0.04, 0.8])
    caxGalaxies = plt.axes([0.84, 0.1, 0.04, 0.8])
    cbStars = plt.colorbar(mappableStars, cax=caxStars)
    cbStars.ax.tick_params(labelsize=8)
    cbStars.set_label(filterStr + "[" + fluxColStr + "] :stars", rotation=270, labelpad=-24, fontsize=9)
    mappableGalaxies = plt.cm.ScalarMappable(cmap="autumn_r", norm=plt.Normalize(vmin=vMin, vmax=vMax))
    mappableGalaxies._A = []      # fake up the array of the scalar mappable. Urgh...
    cbGalaxies = plt.colorbar(mappableGalaxies, cax=caxGalaxies)
    cbGalaxies.set_ticks([])
    cbGalaxies.set_label(filterStr + " [" + fluxColStr + "]: galaxies", rotation=270, labelpad=-6, fontsize=9)

    if camera is not None:
        labelCamera(camera, plt, axes[0, 0], 1.05, 1.14)
    if geLabel is not None:
        plotText(geLabel, plt, axes[0, 0], 0.09, -0.11, color="green")
    if hscRun is not None:
        axes.set_title("HSC stack run: " + hscRun, color="#800080")

    tractStr = "tract: {:d}".format(dataId["tract"])
    axes[0, 0].annotate(tractStr, xy=(1.05, 1.06), xycoords="axes fraction", ha="center", va="center",
                        fontsize=9, color="green")

    fig.savefig(filename, dpi=120)
    plt.close(fig)

    return None


class ColorColorDistance(object):
    """Functor to calculate distance from stellar locus in color-color plot"""
    def __init__(self, band1, band2, band3, poly, unitScale=1.0, xMin=None, xMax=None,
                 fitLineUpper=None, fitLineLower=None):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        if isinstance(poly, np.lib.polynomial.poly1d):
            self.poly = poly
        else:
            self.poly = np.poly1d(poly)
        self.unitScale = unitScale
        self.xMin = xMin
        self.xMax = xMax
        self.fitLineUpper = fitLineUpper
        self.fitLineLower = fitLineLower

    def __call__(self, catalog):
        xx = catalog[self.band1] - catalog[self.band2]
        yy = catalog[self.band2] - catalog[self.band3]
        polyDeriv = np.polyder(self.poly)
        distance2 = np.ones_like(xx)*np.nan
        for i, (x, y) in enumerate(zip(xx, yy)):
            if (not np.isfinite(x) or not np.isfinite(y) or (self.xMin is not None and x < self.xMin) or
                (self.xMax is not None and x > self.xMax) or
                (self.fitLineUpper is not None and y > self.fitLineUpper[0] + self.fitLineUpper[1]*x) or
                    (self.fitLineLower is not None and y < self.fitLineLower[0] + self.fitLineLower[1]*x)):
                distance2[i] = np.nan
                continue
            roots = np.roots(np.poly1d((1, -x)) + (self.poly - y)*polyDeriv)
            distance2[i] = min(distanceSquaredToPoly(x, y, np.real(rr), self.poly) for
                               rr in roots if np.real(rr) == rr)
        return np.sqrt(distance2)*np.where(yy >= self.poly(xx), 1.0, -1.0)*self.unitScale


class SkyAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by filter
        filterRefs = defaultdict(list)  # filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_meas"):
                filterName = patchRef.dataId["filter"]
                filterRefs[filterName].append(patchRef)

        return [(refList, kwargs) for refList in filterRefs.values()]


class SkyAnalysisTask(CoaddAnalysisTask):
    """Version of CoaddAnalysisTask that runs on all inputs simultaneously

    This is most useful for utilising overlaps between tracts.
    """
    _DefaultName = "skyAnalysis"
    RunnerClass = SkyAnalysisRunner
    outputDataset = "plotSky"
