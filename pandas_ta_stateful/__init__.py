# -*- coding: utf-8 -*-
from importlib.metadata import version
version = version("pandas-ta")

from pandas_ta_stateful.maps import EXCHANGE_TZ, RATE, Category, Imports
from pandas_ta_stateful.utils import *
from pandas_ta_stateful.utils import __all__ as utils_all
from pandas_ta_stateful.stateful import *
from pandas_ta_stateful.stateful import __all__ as stateful_all

# Flat Structure. Supports ta.ema() or ta.overlap.ema()
from pandas_ta_stateful.candle import *
from pandas_ta_stateful.cycle import *
from pandas_ta_stateful.momentum import *
from pandas_ta_stateful.overlap import *
from pandas_ta_stateful.performance import *
from pandas_ta_stateful.statistics import *
from pandas_ta_stateful.trend import *
from pandas_ta_stateful.volatility import *
from pandas_ta_stateful.volume import *
from pandas_ta_stateful.candle import __all__ as candle_all
from pandas_ta_stateful.cycle import __all__ as cycle_all
from pandas_ta_stateful.momentum import __all__ as momentum_all
from pandas_ta_stateful.overlap import __all__ as overlap_all
from pandas_ta_stateful.performance import __all__ as performance_all
from pandas_ta_stateful.statistics import __all__ as statistics_all
from pandas_ta_stateful.trend import __all__ as trend_all
from pandas_ta_stateful.volatility import __all__ as volatility_all
from pandas_ta_stateful.volume import __all__ as volume_all

# Common Averages useful for Indicators
# with a mamode argument, like ta.adx()
from pandas_ta_stateful.ma import ma

# Custom External Directory Commands. See help(import_dir)
from pandas_ta_stateful.custom import create_dir, import_dir

# Enable "ta" DataFrame Extension
from pandas_ta_stateful.core import AnalysisIndicators

__all__ = [
    # "name",
    "EXCHANGE_TZ",
    "RATE",
    "Category",
    "Imports",
    "version",
    "ma",
    "create_dir",
    "import_dir",
    "AnalysisIndicators",
    "AllStudy",
    "CommonStudy",
]

__all__ += [
    utils_all
    + candle_all
    + cycle_all
    + momentum_all
    + overlap_all
    + performance_all
    + statistics_all
    + trend_all
    + volatility_all
    + volume_all
    + stateful_all
]
