"""Time feature extraction utilities.

Adapted from Informer official implementation.
Generates temporal features (minute, hour, day, week, month) from timestamps.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature(ABC):
    @abstractmethod
    def __call__(self, index):
        """Return one normalized temporal feature for a DatetimeIndex."""

    def __repr__(self):
        return self.__class__.__name__


class MinuteOfHour(TimeFeature):
    def __call__(self, index):
        return index.minute / 59.0 - 0.5


class SecondOfMinute(TimeFeature):
    def __call__(self, index):
        return index.second / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index):
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index):
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index):
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index):
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index):
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index):
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def _freq_map(freq):
    """Map frequency string to a list of TimeFeature objects."""
    offset = to_offset(freq)
    features_by_offset = (
        ((offsets.YearBegin, offsets.YearEnd), []),
        ((offsets.QuarterBegin, offsets.QuarterEnd, offsets.MonthBegin, offsets.MonthEnd), [MonthOfYear()]),
        ((offsets.Week,), [DayOfMonth(), WeekOfYear()]),
        ((offsets.Day, offsets.BusinessDay), [DayOfWeek(), DayOfMonth(), DayOfYear()]),
        ((offsets.Hour,), [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]),
        ((offsets.Minute,), [MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]),
        (
            (offsets.Second,),
            [SecondOfMinute(), MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()],
        ),
    )
    for offset_types, features in features_by_offset:
        if isinstance(offset, offset_types):
            return features
    raise ValueError(f"Unsupported time frequency: {freq!r}")


def time_features(dates, freq="h"):
    """Generate temporal feature array from timestamps.

    Args:
        dates: DatetimeIndex or list of timestamps.
        freq: frequency string (e.g., 'h', 't', 's').

    Returns:
        (N, D) ndarray of normalized time features [-0.5, 0.5].
    """
    features = _freq_map(freq)
    if isinstance(dates, pd.DatetimeIndex):
        index = dates
    else:
        index = pd.to_datetime(dates)

    if not features:
        return np.empty((len(index), 0), dtype=np.float32)
    return np.column_stack([feat(index) for feat in features]).astype(np.float32)
