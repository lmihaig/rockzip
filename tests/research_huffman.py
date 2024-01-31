from collections import defaultdict
import heapq

from bitstring import BitStream

length_code_ranges = [
    [257, 0, 3, 3],
    [258, 0, 4, 4],
    [259, 0, 5, 5],
    [260, 0, 6, 6],
    [261, 0, 7, 7],
    [262, 0, 8, 8],
    [263, 0, 9, 9],
    [264, 0, 10, 10],
    [265, 1, 11, 12],
    [266, 1, 13, 14],
    [267, 1, 15, 16],
    [268, 1, 17, 18],
    [269, 2, 19, 22],
    [270, 2, 23, 26],
    [271, 2, 27, 30],
    [272, 2, 31, 34],
    [273, 3, 35, 42],
    [274, 3, 43, 50],
    [275, 3, 51, 58],
    [276, 3, 59, 66],
    [277, 4, 67, 82],
    [278, 4, 83, 98],
    [279, 4, 99, 114],
    [280, 4, 115, 130],
    [281, 5, 131, 162],
    [282, 5, 163, 194],
    [283, 5, 195, 226],
    [284, 5, 227, 257],
    [285, 0, 258, 258],
]


# Compact representation of the distance code value (0-31), distance range and number
# of extra bits to use in LZ77 compression (See Section 3.2.5 of RFC 1951)
#   [code, num_bits_extra, lower_bound, upper_bound]
distance_code_ranges = [
    [0, 0, 1, 1],
    [1, 0, 2, 2],
    [2, 0, 3, 3],
    [3, 0, 4, 4],
    [4, 1, 5, 6],
    [5, 1, 7, 8],
    [6, 2, 9, 12],
    [7, 2, 13, 16],
    [8, 3, 17, 24],
    [9, 3, 25, 32],
    [10, 4, 33, 48],
    [11, 4, 49, 64],
    [12, 5, 65, 96],
    [13, 5, 97, 128],
    [14, 6, 129, 192],
    [15, 6, 193, 256],
    [16, 7, 257, 384],
    [17, 7, 385, 512],
    [18, 8, 513, 768],
    [19, 8, 769, 1024],
    [20, 9, 1025, 1536],
    [21, 9, 1537, 2048],
    [22, 10, 2049, 3072],
    [23, 10, 3073, 4096],
    [24, 11, 4097, 6144],
    [25, 11, 6145, 8192],
    [26, 12, 8193, 12288],
    [27, 12, 12289, 16384],
    [28, 13, 16385, 24576],
    [29, 13, 24577, 32768],
]


def init_values(code_ranges):
    # example for length but similar for distances
    #        value : symbol, num_bits_extra, extra_bits
    #         "20" : 269,     2,             '01'
    codes = defaultdict(Code_Data)
    for symbol, num_bits_extra, lower_bound, upper_bound in code_ranges:
        # code of symbol, number of extra bits, value of extra bits
        for i in range(lower_bound, upper_bound + 1):
            # to better explain this, for the LL symbols map I had to somehow
            # encode both the LENGTH symbols and the LITERALS symbols
            # i decided that LENGTHS are actually str(value)
            # and literals are represented by their extended ascii codes, so a int(0) - int(255) are chr()
            # and "3" - "287" are str() that represent the length of the match,
            # this is so I don't have an extra map for LENGTH_VALUE to SYMBOL
            # EXCEPTION is 256 which is the EOB symbol but I made it a LITERAL because ughhhhhh
            codes[str(i)] = Code_Data(
                symbol=symbol,
                offset_bits=bin(i - lower_bound)[2:].zfill(num_bits_extra)
                if num_bits_extra > 0
                else None,
            )

    return codes


class LZ77Encoder:
    def __init__(self, window_size=2**15, lookahead_buffer_size=2**8):
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

        self.LL_symbols_map = init_values(length_code_ranges)
        for i in range(0, 257):
            self.LL_symbols_map[i] = Code_Data(symbol=i, offset_bits=None)

        self.D_symbols_map = init_values(distance_code_ranges)
        self.bs = None

    def write_value(self, value, bitstream):
        symbol, offset = value.symbol, value.extra_bits
        res = bin(symbol) + offset if offset is not None else bin(symbol)
        bitstream.append(res)

    def find_longest_match(self, window, lookahead):
        longest_match_length = None
        longest_match_distance = None
        window = bytes(window)
        lookahead = bytes(lookahead)

        while (index := window.rfind(lookahead)) == -1:
            lookahead = lookahead[:-1]

        if index != -1:
            longest_match_length = len(lookahead)
            longest_match_distance = len(window) - index

        return longest_match_length, longest_match_distance

    def compress(self, data):
        self.bs = BitStream(data)
        compressed_data = BitStream()
        window = bytearray()
        lookahead = bytearray()

        for _ in range(self.lookahead_buffer_size):
            if self.bs.pos < len(data) * 8:
                lookahead += self.bs.read("bytes:1")

        while self.bs.pos < len(data) * 8 or len(lookahead) > 0:
            match_length, match_distance = self.find_longest_match(window, lookahead)

            if match_length is not None and match_length > 2:
                compressed_data.append("0b1")  # Match bit '1'
                # print(match_length, match_distance)

                self.LL_symbols_map[str(match_length)].freq += 1
                self.D_symbols_map[str(match_distance)].freq += 1
                length_val = self.LL_symbols_map[str(match_length)]
                distance_val = self.D_symbols_map[str(match_distance)]

                self.write_value(length_val, compressed_data)
                self.write_value(distance_val, compressed_data)
                # print(length_val, distance_val)

                for _ in range(match_length):
                    if len(lookahead) == 0:
                        break
                    window += lookahead[:1]
                    window = window[-self.window_size :]
                    lookahead = lookahead[1:]
                    if self.bs.pos < len(data) * 8:
                        lookahead += self.bs.read("bytes:1")
            else:
                compressed_data.append("0b0")  # Literal bit '0'

                literal = lookahead[:1]
                self.LL_symbols_map[literal[0]].freq += 1
                compressed_data.append(BitStream(bytes=literal))

                window += literal
                window = window[-self.window_size :]
                lookahead = lookahead[1:]
                if self.bs.pos < len(data) * 8:
                    lookahead += self.bs.read("bytes:1")

        self.LL_symbols_map = {
            k: v
            for k, v in sorted(
                self.LL_symbols_map.items(), key=lambda item: item[1].freq, reverse=True
            )
        }
        self.D_symbols_map = {
            k: v
            for k, v in sorted(
                self.D_symbols_map.items(), key=lambda item: item[1].freq, reverse=True
            )
        }

        # WRITING EOB
        compressed_data.append("0b1")
        self.LL_symbols_map[256].freq += 1
        compressed_data.append(BitStream(bin(256)))

        return compressed_data


class Code_Data:
    def __init__(self, symbol: int, offset_bits: str, freq=0):
        self.symbol = symbol
        self.extra_bits = offset_bits
        self.freq = freq

    def __str__(self):
        return f"{{symbol: {self.symbol}, extra_bits: {self.extra_bits}, freq: {self.freq}}}"

    def __repr__(self):
        return self.__str__()


class Node:
    def __init__(
        self,
        symbol=None,
        freq=0,
        left=None,
        right=None,
        is_package=False,
    ):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self.is_package = is_package

    def __lt__(self, other):
        return self.freq < other.freq

    def __repr__(self):
        return f"Node(symbol={self.symbol}, freq={self.freq})"


class HuffmanTree:
    def __init__(self, symbols_map, max_length=15):
        self.symbol_map = symbols_map
        self.root = self._package_merge(symbols_map, max_length)
        self.codes = self._generate_huffman_codes(self.root)
        self._add_extra_bits()

    def _package_merge(self, frequencies, max_length=15):
        leaves = [Node(symbol, freq=data.freq) for symbol, data in frequencies.items()]
        heapq.heapify(leaves)

        levels = [leaves]

        for _ in range(max_length - 1):
            new_level = []
            prev_level = levels[-1] + levels[-1]
            prev_level.sort(key=lambda node: (node.freq, self._get_symbol_order(node)))

            for i in range(0, len(prev_level), 2):
                if i + 1 < len(prev_level):
                    new_node = Node(
                        freq=prev_level[i].freq + prev_level[i + 1].freq,
                        left=prev_level[i],
                        right=prev_level[i + 1],
                        is_package=True,
                    )
                    new_level.append(new_node)

            new_level = sorted(new_level, key=lambda node: node.freq)[
                : len(frequencies)
            ]
            heapq.heapify(new_level)
            levels.append(new_level)

        final_nodes = []
        for level in levels:
            for node in level:
                if not node.is_package:
                    final_nodes.append(node)

        while len(final_nodes) > 1:
            first = heapq.heappop(final_nodes)
            second = heapq.heappop(final_nodes)
            merged = Node(freq=first.freq + second.freq, left=first, right=second)
            heapq.heappush(final_nodes, merged)

        return final_nodes[0] if final_nodes else None

    def _get_symbol_order(self, node):
        if node.symbol is not None:
            return self.symbol_map[node.symbol].symbol
        return float("inf")

    def _generate_huffman_codes(self, node, prefix="", code_map=None):
        if code_map is None:
            code_map = {}

        if node is not None:
            if node.symbol is not None:
                code_map[node.symbol] = {
                    "code": prefix,
                }
            self._generate_huffman_codes(node.left, prefix + "0", code_map)
            self._generate_huffman_codes(node.right, prefix + "1", code_map)

        # Sorting the codes by length and then lexicographically
        sorted_codes = sorted(
            code_map.items(), key=lambda item: (len(item[1]["code"]), item[1]["code"])
        )
        return dict(sorted_codes)

    def _add_extra_bits(self):
        for symbol in list(self.codes.keys()):
            self.codes[symbol]["extra_bits"] = self.symbol_map[symbol].extra_bits
            if type(symbol) == str:
                self.codes[self.symbol_map[symbol].symbol] = self.codes.pop(symbol)


class HuffmanEncoder:
    def __init__(self, LL_symbols_map, D_symbols_map, max_length=15):
        self.LL_map = LL_symbols_map
        self.D_map = D_symbols_map

        non_zero_LL = {
            k: v
            for k, v in sorted(
                (
                    (symbol, details)
                    for symbol, details in LL_symbols_map.items()
                    if details.freq > 0
                ),
                key=lambda item: item[1].freq,
                reverse=True,
            )
        }
        non_zero_D = {
            k: v
            for k, v in sorted(
                (
                    (symbol, details)
                    for symbol, details in D_symbols_map.items()
                    if details.freq > 0
                ),
                key=lambda item: item[1].freq,
                reverse=True,
            )
        }

        self.LL_tree = HuffmanTree(non_zero_LL, max_length)
        self.D_tree = HuffmanTree(non_zero_D, max_length)

        self.LL_codes = self.LL_tree.codes
        self.D_codes = self.D_tree.codes
        self.CL_codes = self._generate_CL_codes_table()

        self.CL_lens_header = self.generate_CL_lens_header()
        self.LL_D_header = self.generate_LL_D_header()

    def generate_CL_lens_header(self):
        # remember to return and push these LSB padded 3
        enc_ord = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
        CL_lens = []
        for val in enc_ord:
            if val in self.CL_codes:
                CL_lens.append(len(self.CL_codes[val]["code"]))
            else:
                CL_lens.append(0)
        return CL_lens

    def generate_CL_stream(self, combined_lengths):
        """
        Code Length Encoding
        0-15: Represent code lengths of 0-15
        16: Copy the previous code length 3-6 times.
            (2 bits of length)
            (0 = 3, ... , 3 = 6)
        17: Repeat a code length of 0 for 3-10 times.
            (3 bits of length)
        18: Repeat a code length of 0 for 11-138 times
            (7 bits of length)
        """
        CL_codes = []

        prev_length = -1
        count = 0

        for length in combined_lengths + [-1]:  # Add sentinel at end
            if length == prev_length:
                count += 1
                if length != 0 and count >= 6:
                    CL_codes.append(
                        (16, min(count - 3, 3))
                    )  # Repeat previous 3-6 times
                    count -= min(count, 6)
                elif length == 0 and count >= 138:
                    CL_codes.append(
                        (18, min(count - 11, 127))
                    )  # Repeat zero 11-138 times
                    count -= min(count, 138)
            else:
                # Output for previous length
                if prev_length != -1:
                    if prev_length == 0:
                        if count <= 2:
                            CL_codes.extend([(0, 0)] * count)  # Output zeros directly
                        elif count <= 10:
                            CL_codes.append((17, count - 3))  # Repeat zero 3-10 times
                        else:
                            CL_codes.append(
                                (18, count - 11)
                            )  # Repeat zero 11-138 times
                    else:
                        CL_codes.extend(
                            [(prev_length, 0)] * min(count, 2)
                        )  # Output non-zero length directly
                        if count > 2:
                            CL_codes.append((16, count - 3))  # Repeat previous length

                prev_length = length
                count = 1
        return CL_codes

    def _generate_CL_codes_table(self):
        CL_codes = {}

        LL_code_lengths_table = {}

        for symbol in range(286):
            if symbol in self.LL_codes:
                LL_code_lengths_table[symbol] = len(self.LL_codes[symbol]["code"])
            else:
                LL_code_lengths_table[symbol] = 0

        D_code_lengths_table = {}
        for symbol in range(30):
            if symbol in self.D_codes:
                D_code_lengths_table[symbol] = len(self.D_codes[symbol]["code"])
            else:
                D_code_lengths_table[symbol] = 0

        combined_lengths = list(LL_code_lengths_table.values()) + list(
            D_code_lengths_table.values()
        )

        # sanity check
        # combined_lengths =[0] * 32+ [4]+ [0] * 32+ [4]+ [3]+ [3, 3, 3]+ [0] * 138+ [0] * 48+ [3]+ [0]+ [3]+ [0, 0, 0, 0]+ [3]
        # print(combined_lengths)

        CL_codes = self.generate_CL_stream(combined_lengths)

        CL_codes_array = [0] * 19
        for symbol, length in CL_codes:
            CL_codes_array[symbol] += 1

        CL_codes_table = defaultdict(int)
        for i, freq in enumerate(CL_codes_array):
            if freq > 0:
                CL_codes_table[i] = Code_Data(symbol=i, offset_bits=None, freq=freq)
        self.LL_D_table = CL_codes.copy()

        CL_tree = HuffmanTree(CL_codes_table, max_length=7)
        return CL_tree.codes

    def generate_LL_D_header(self):
        # remember to return and push these to the bitstream (code, extra_bits)
        ll_d_lens = []
        for symbol, offset in self.LL_D_table:
            CL_symbol = symbol
            extra_bits = offset if offset > 0 else None
            if extra_bits:
                match symbol:
                    case 16:
                        fill = 2
                    case 17:
                        fill = 3
                    case 18:
                        fill = 7
                extra_bits = bin(extra_bits)[2:].zfill(fill)

            res = (self.CL_codes[CL_symbol]["code"], extra_bits)
            ll_d_lens.append(res)
        return ll_d_lens


if __name__ == "__main__":
    test_string = BitStream(b"ABCDEABCD ABCDEABCD")
    test_lz77 = LZ77Encoder()
    test_lz77.compress(test_string.tobytes())
    LL_symbols_map = test_lz77.LL_symbols_map
    D_symbols_map = test_lz77.D_symbols_map

    test_huff = HuffmanEncoder(LL_symbols_map, D_symbols_map)
    print(test_huff.LL_codes)
    print(test_huff.D_codes)
