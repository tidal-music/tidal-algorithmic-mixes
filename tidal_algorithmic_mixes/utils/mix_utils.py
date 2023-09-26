from typing import List, Any

import pyspark.sql.functions as F
import pyspark.sql.types as T
import tidal_algorithmic_mixes.utils.constants as c


def at_date(date):
    """
    Standardized function for generating the atDate columns in DynamoDB mix tables.

    :type date: datetime.datetime
    :rtype: str
    """
    return date.isoformat(timespec="seconds")


def updated(time):
    """
    Standardized function for generating the updated columns in the DynamoDB mix tables.

    :type time: time.time
    :rtype:     int
    """
    return int(round(time * 1000))


def mix_id(col1, col2):
    """
    Standardized function for generating the mixId column in the DynamoDB mix tables.

    :param col1:      The first combination key (e.g. a timestamp)
    :type col1:       pyspark.sql.Column
    :param col2:      The column containing the userId
    :type col2:       pyspark.sql.Column
    :rtype:           pyspark.sql.Column
    """
    return F.substring(F.md5(F.concat(col1, col2)), 0, c.MIX_ID_LENGTH)


def mix_id_stable(col1):
    """
    Standardized function for generating the mixId column in the DynamoDB mix tables when IDs should be stable.

    :param col1:        The key used to generate the ID (e.g. artistId or contributorMixType)
    :rtype:             pyspark.sql.Column
    """
    return F.substring(F.md5(F.col(col1)), 0, c.MIX_ID_LENGTH)


@F.udf(T.ArrayType(T.StructType([
        T.StructField(c.ID, T.IntegerType(), False),
        T.StructField(c.NAME, T.StringType(), False),
        T.StructField(c.IMAGE, T.StringType(), False)])))
def pick_top_artists_udf(artists, num_artists):
    distinct, seen = [], {}
    for a in artists:
        if not a[c.ID] in seen and a[c.IMAGE]:
            seen.update({a[c.ID]: True})
            distinct.append(a)
    return distinct[:num_artists]


@F.udf(T.ArrayType(T.StructType([
        T.StructField(c.ID, T.IntegerType(), False),
        T.StructField(c.NAME, T.StringType(), False),
        T.StructField(c.IMAGE, T.StringType(), False),
        T.StructField(c.COVER, T.StringType(), False)])))
def pick_top_artists_with_cover(artists, num_artists):
    distinct, seen = [], {}
    for a in artists:
        if not a[c.ID] in seen and a[c.IMAGE] and a[c.COVER]:
            seen.update({a[c.ID]: True})
            distinct.append(a)
    return distinct[:num_artists]


@F.udf(returnType=T.ArrayType(T.IntegerType()))
def last_n_items(items: List[Any], n):
    """ F.slice does not work for large negative numbers (e.g. last 100 items) and returns empty lists!"""
    return items[-n:]
