from enum import StrEnum


class RegionTag(StrEnum):
    """Enumeration of short serialization tags for region types.

    Members use the short string values that are persisted in serialized
    representations (so existing on-disk/json formats remain compact).
    """

    SPAN = "s"
    RECT = "rr"
    POLY = "pr"
    LINE = "lr"
    POLYLINE = "pl"
