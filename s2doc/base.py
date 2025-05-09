import abc


class DocObj(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: dict) -> "DocObj":
        pass

    @abc.abstractmethod
    def to_obj(self) -> dict:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
