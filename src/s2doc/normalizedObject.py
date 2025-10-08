from collections.abc import ItemsView, Iterator, ValuesView
from typing import Generic, TypeVar

from .element import Element
from .errors import ExistenceError, LoadFromDictError
from .page import Page
from .util import build_compressed_trie, flatten_compressed_trie

T = TypeVar("T", Page, Element)


class NormalizedObj(Generic[T]):
    def __init__(self, byId: dict[str, T] | None = None):
        self.byId: dict[str, T] = byId if byId else dict()
        self.allIds: list[str] = list(self.byId.keys())

    def values(self) -> ValuesView[T]:
        return self.byId.values()

    def items(self) -> ItemsView[str, T]:
        return self.byId.items()

    # dict .get method
    def get(self, key: str, default: T | None = None) -> T | None:
        return self.byId.get(key, default)

    def __getitem__(self, key: str) -> T:
        ret = None
        if isinstance(key, int):
            ret = self.byId.get(self.allIds[key])
        else:
            ret = self.byId.get(key)
        if ret is None:
            raise ExistenceError(f"Key {key} not found in NormalizedObj")
        return ret

    def __setitem__(self, key: str, value: T) -> None:
        if key not in self.byId:
            self.allIds.append(key)
        self.byId[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.byId

    def __len__(self) -> int:
        return len(self.byId)

    def __iter__(self) -> Iterator[T]:
        return iter(self.byId.values())

    def _sync_ids(self) -> None:
        """Ensure allIds matches byId keys"""
        self.allIds = list(self.byId.keys())

    def add(self, obj: T) -> None:
        """Add an object to the collection (replaces append)"""
        if obj.oid in self.byId:
            raise ExistenceError(f"Object with id {obj.oid} already exists")
        self.byId[obj.oid] = obj
        self.allIds.append(obj.oid)

    def add_with_key(self, key: str, obj: T) -> None:
        """
        Adds an object to the collection with the specified key.

        If the key already exists in the collection:
            - If `append` is True, the object is appended to the existing value.
              If the existing value is not a list, it is converted into a list
              containing the existing value and the new object.
            - If `append` is False, an `ExistsError` is raised.

        If the key does not exist, the object is added to the collection, and
        the key is appended to the list of all IDs.

        Args:
            key (str): The key to associate with the object.
            obj (T): The object to add to the collection.
            append (bool, optional): Whether to append the object if the key
                already exists. Defaults to True.

        Raises:
            ExistsError: If the key already exists and `append` is False.
        """
        if key in self.byId:
            raise ExistenceError(f"Object with id {key} already exists")
        else:
            self.byId[key] = obj
            self.allIds.append(key)

    def remove(self, obj: T | str) -> None:
        oid = obj.oid if not isinstance(obj, str) else obj
        if oid in self.byId:
            del self.byId[oid]
        self._sync_ids()

    def remove_multiple(self, objs: list[T] | list[str]) -> None:
        for obj in objs:
            oid = obj.oid if not isinstance(obj, str) else obj
            if oid in self.byId:
                del self.byId[oid]
        self._sync_ids()

    @classmethod
    def from_dict(cls, dic: dict, what: str) -> "NormalizedObj":
        cls_map = {"elements": Element, "pages": Page}
        obj_cls = cls_map.get(what)
        if obj_cls is None:
            raise LoadFromDictError(what, "Unknown object type")

        # check if it is trie - first element should not contain oid
        if "oid" not in next(iter(dic.values())).keys():
            flat_dic = flatten_compressed_trie(dic)
        else:
            flat_dic = dic

        def rec_obj_creation(part: dict, obj_cls) -> dict[str, T]:
            return (
                {k: obj_cls.from_dict(v) for k, v in part.items()} if obj_cls else part
            )

        try:
            byId = rec_obj_creation(flat_dic, obj_cls) if obj_cls else flat_dic
            return cls(byId)
        except Exception as e:
            raise LoadFromDictError(what, str(e))

    def to_obj(self, as_trie: bool = False) -> dict:
        def get_child(key):
            return (
                self.byId[key].to_obj()
                if hasattr(self.byId[key], "to_obj")
                else self.byId[key]
            )
        if as_trie:
            return build_compressed_trie(self.byId, get_child)
        return {k: get_child(k) for k in self.byId}
