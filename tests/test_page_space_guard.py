import pytest

from src.s2doc.errors import IncompatibleError
from src.s2doc.page import Page


def test_factor_between_spaces_zero_dims():
    p = Page("p1", 1)
    # default page spaces have zero dimensions; conversion should be guarded
    with pytest.raises(IncompatibleError):
        p.factor_between_spaces("img", "xml")
