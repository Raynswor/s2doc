from abc import ABC, abstractmethod


class DocObj(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "DocObj":
        pass

    @abstractmethod
    def to_obj(self) -> dict:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
