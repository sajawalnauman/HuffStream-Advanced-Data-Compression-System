from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """

    freq_dict = {}
    for elem in text:
        if elem not in freq_dict:
            freq_dict[elem] = 1
        else:
            freq_dict[elem] += 1

    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """

    if len(freq_dict) == 0:
        return HuffmanTree(None)

    elif len(freq_dict) == 1:
        key = list(freq_dict.keys())[0]
        any_valid_byte = (key + 1) % 256
        dummy_tree = HuffmanTree(any_valid_byte)
        result = HuffmanTree(freq_dict[key], HuffmanTree(key), dummy_tree)
        return result

    else:
        freq_tree = []
        for symbol in freq_dict:
            freq_tree.append((freq_dict[symbol], HuffmanTree(symbol)))

        while len(freq_tree) > 1:
            freq_tree.sort()
            tree_left = freq_tree.pop(0)
            tree_right = freq_tree.pop(0)
            tree = HuffmanTree(None, tree_left[1], tree_right[1])
            freq_tree.append((tree_left[0] + tree_right[0], tree))

        return freq_tree.pop()[1]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """

    if tree.is_leaf():
        return {}
    else:
        left_dict = {}
        right_dict = {}

        if tree.left.is_leaf():
            left_dict[tree.left.symbol] = "0"
        else:
            left_dict = get_codes(tree.left)
            for key in left_dict:
                left_dict[key] = "0" + left_dict[key]

        if tree.right.is_leaf():
            right_dict[tree.right.symbol] = "1"
        else:
            right_dict = get_codes(tree.right)
            for key in right_dict:
                right_dict[key] = "1" + right_dict[key]

        final_dict = {**left_dict, **right_dict}

        return final_dict


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """

    __number_nodes_helper(tree, 0)


def __number_nodes_helper(tree: HuffmanTree, counter: int) -> int:
    """
    Helper method for number_nodes.
    """
    if tree is not None and tree.left and tree.right:

        left = __number_nodes_helper(tree.left, counter)
        right = __number_nodes_helper(tree.right, left)

        counter = right
        tree.number = counter
        counter += 1

    return counter


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """

    codes = get_codes(tree)

    sum_code = 0
    sum_freq = 0

    for key in freq_dict:
        sum_code += len(codes[key]) * freq_dict[key]
        sum_freq += freq_dict[key]

    return sum_code / sum_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """

    out = ""
    byte_list = []

    for symbol in text:
        out += codes[symbol]

    for i in range(0, len(out), 8):
        byte_list.append(bits_to_byte(out[i: i + 8]))

    return bytes(byte_list)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """

    left_list = []
    right_list = []
    left = []
    right = []

    if tree is not None and tree.left and tree.right:

        if tree.left.is_leaf():
            left_list.append(0)
            left_list.append(tree.left.symbol)
        else:
            left_list.append(1)
            left_list.append(tree.left.number)
            left = tree_to_bytes(tree.left)

        if tree.right.is_leaf():
            right_list.append(0)
            right_list.append(tree.right.symbol)
        else:
            right_list.append(1)
            right_list.append(tree.right.number)
            right = tree_to_bytes(tree.right)

    return bytes(left) + bytes(right) + bytes(left_list) + bytes(right_list)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """

    root_node = node_lst[root_index]

    if root_node.l_type == 0:
        left = HuffmanTree(root_node.l_data)
    else:
        left = generate_tree_general(node_lst, root_node.l_data)

    if root_node.r_type == 0:
        right = HuffmanTree(root_node.r_data)
    else:
        right = generate_tree_general(node_lst, root_node.r_data)

    tree = HuffmanTree(None, left, right)
    tree.number = root_index
    return tree


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """

    stack = [root_index]
    stack.pop()

    while len(node_lst) > 0:
        node = node_lst.pop(0)

        if node.l_type == 0:
            left_tree = HuffmanTree(node.l_data)
        else:
            left_tree = stack.pop(-2)

        if node.r_type == 0:
            right_tree = HuffmanTree(node.r_data)
        else:
            right_tree = stack.pop()

        tree = HuffmanTree(None, left_tree, right_tree)

        stack.append(tree)
    return stack.pop()


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """

    encoded = ""
    decoded = []

    symbol_dict = get_codes(tree)
    code_dict = {v: k for k, v in symbol_dict.items()}

    for byte in text:
        b_int = byte_to_bits(byte)
        encoded += b_int

    search_bit = ""
    for bit in encoded:
        search_bit += bit
        if search_bit in code_dict:
            decoded.append(code_dict[search_bit])
            search_bit = ""

    out = decoded[:size]
    return bytes(out)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    freq_vals = [(freq_dict[key], key) for key in freq_dict]
    freq_vals.sort()
    __tree_swap_helper(tree, freq_vals)


def __tree_swap_helper(tree: HuffmanTree, freq_vals: list[tuple[int, int]]) -> \
        None:
    """
    Helper method for improve_tree.
    """
    if tree is not None and tree.left and tree.right:

        if tree.left.symbol:
            tree.left.symbol = freq_vals.pop()[1]

        if tree.right.symbol:
            tree.right.symbol = freq_vals.pop()[1]

        __tree_swap_helper(tree.left, freq_vals)
        __tree_swap_helper(tree.right, freq_vals)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
