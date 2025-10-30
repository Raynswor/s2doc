class DocumentError(Exception):
    """Base exception for Document related errors"""

    pass


class PageNotFoundError(DocumentError):
    """Raised when a page cannot be found"""

    pass


class ElementNotFoundError(DocumentError):
    """Raised when an element cannot be found"""

    pass


class ExistenceError(DocumentError):
    """Raised when existence of object is questioned"""

    pass


class LoadFromDictError(DocumentError):
    """Raised when loading from a dictionary fails"""

    # standard message failed to load {cls} from dict: {error}
    def __init__(self, cls: str, error: str):
        super().__init__(f"Failed to load {cls} from dict: {error}")


class IncompatibleError(DocumentError):
    """Raised when an object is incompatible with another"""

    # standard message incompatible {cls} with {other}
    def __init__(self, error: str, cls: str, other: str):
        super().__init__(f"Incompatible ({error}): {cls} -> {other}")
