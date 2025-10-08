import base64
import io

from PIL.Image import Image, open


def base64_to_img(base64_string: str) -> Image:
    try:
        header, encoded = base64_string.split(",", 1)
    except ValueError:
        encoded = base64_string

    image_bytes = base64.b64decode(encoded)
    return open(io.BytesIO(image_bytes))


def img_to_base64(image: Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def flatten_compressed_trie(trie: dict) -> dict:
    """
    Flattens a compressed trie structure into a flat dictionary.

    Args:
        trie (dict): The compressed trie to flatten.

    Returns:
        dict: A flat dictionary representation of the trie.
    """
    flat = {}

    def recurse(node: dict, prefix: str):
        for key, value in node.items():
            if isinstance(value, dict):
                recurse(value, prefix + key)
            else:
                flat[prefix + key] = value

    recurse(trie, "")
    return flat


def build_compressed_trie(data: dict, get_child) -> dict:
    """
    Builds a compressed trie from a flat dictionary.

    Args:
        data (dict): A flat dictionary to convert into a trie.

    Returns:
        dict: A compressed trie representation of the dictionary.
    """
    trie = {}
    for key in data.keys():
        current = trie
        for char in key:
            current = current.setdefault(char, {})
        current[None] = get_child(key)

    def compress_trie_remove_none(node: dict) -> dict:
        if None in node:
            return node[None]

        for key, value in list(node.items()):
            compressed_child = compress_trie_remove_none(value)
            if len(compressed_child) == 1:
                if isinstance(compressed_child, set):
                    pass
                new_key = key + list(compressed_child.keys())[0]
                node[new_key] = compressed_child[list(compressed_child.keys())[0]]
                del node[key]
            else:
                node[key] = compressed_child

        return node

    return compress_trie_remove_none(trie)


def compress_ids_front_coding(ids: set[str]) -> dict | list[str]:
    if not ids:
        return []

    sorted_ids = sorted(ids)
    first = sorted_ids[0]
    prefix_len = len(first)
    
    # Find longest common prefix
    for other in sorted_ids[1:]:
        i = 0
        while i < prefix_len and i < len(other) and other[i] == first[i]:
            i += 1
        prefix_len = i
        if prefix_len == 0:
            break

    if prefix_len == 0:
        return sorted_ids  # no compression gain

    base = first[:prefix_len]
    suffixes = [s[prefix_len:] for s in sorted_ids]
    return {"base": base, "suffixes": suffixes}


def decompress_ids_front_coding(data: dict | list[str]) -> set[str]:
    if isinstance(data, list):
        return set(data)
    base = data["base"]
    return {base + suffix for suffix in data["suffixes"]}
