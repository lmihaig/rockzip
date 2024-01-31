# <!-- nu citit codul acum e oribil promit ca il reformatez cand am timp liber (niciodata) -->
from collections import defaultdict
from bitstring import BitArray, BitStream, ReadError
import heapq
from utils.bad_idea import This_is_a_bad_idea


# Compact representation of the length code value (257-285), length range and number
# of extra bits to use in LZ77 compression (See Section 3.2.5 of RFC 1951)
#   [code, num_bits_extra, lower_bound, upper_bound]
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


def static_LL_to_length(symbol):
    if 0 <= symbol < 144:
        return 8
    elif 144 <= symbol < 256:
        return 9
    elif 256 <= symbol < 280:
        return 7
    elif 280 <= symbol < 288:
        return 8
    else:
        raise ValueError("Symbol out of range")


def static_D_to_length(symbol):
    return 5


def initialize_static_huffman_codes():
    static_LL_huffman_codes = {}
    for i in range(288):
        code_length = static_LL_to_length(i)
        static_LL_huffman_codes[i] = {
            "code": format(i, "0" + str(code_length) + "b"),
            "extra_bits": None,
        }

    static_D_huffman_codes = {}
    for i in range(32):
        code_length = static_D_to_length(i)
        static_D_huffman_codes[i] = {"code": format(i, "05b"), "extra_bits": None}

    static_LL_symbols_map = init_values(length_code_ranges)
    static_D_symbols_map = init_values(distance_code_ranges)

    return (
        static_LL_huffman_codes,
        static_D_huffman_codes,
        static_LL_symbols_map,
        static_D_symbols_map,
    )


class Code_Data:
    def __init__(self, symbol: int, offset_bits: str, freq=0):
        self.symbol = symbol
        self.extra_bits = offset_bits
        self.freq = freq

    def __str__(self):
        return f"{{symbol: {self.symbol}, extra_bits: {self.extra_bits}, freq: {self.freq}}}"

    def __repr__(self):
        return self.__str__()


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


class DeflateEncoder:
    def __init__(self, input):
        self.input_buffer = input
        self.output_buffer = BitStream()

        self.lz77 = None
        self.huffman = None

        self.bit_stack = []

    def push_bits(self, bits, lsb_first=False, pad_amount=None):
        # bits has to be a binary string "0b1001" or "1001"
        if bits.startswith("0b"):
            bits = bits[2:]

        if pad_amount:
            bits = bits.zfill(pad_amount)

        if len(bits) == 16:
            bits = bits[8:] + bits[:8]

        if lsb_first:
            bits = "".join([bits[x : x + 8][::-1] for x in range(0, len(bits), 8)])

        bits = list(map(int, bits))
        # this is not clean but it works
        for bit in bits:
            self.bit_stack.append(bit)
            if len(self.bit_stack) == 8:
                # print(self.bit_stack[::-1])
                self.output_buffer.append(self.bit_stack[::-1])
                self.bit_stack = []

    def pad_body(self):
        while len(self.bit_stack) > 0:
            self.push_bits("0")

    def write_block(self, block_data, is_last=0, block_type="0b10"):
        if block_type == "0b00" and len(self.input_buffer) >= 2**16:
            raise ValueError("NU SE POATE ASA CEVA")

        self.push_bits(bin(is_last))
        # they call it block_type 10 in the RFC but since it's data/number it's LSB first
        self.push_bits(block_type, lsb_first=True, pad_amount=2)

        block_types = {
            "0b00": 1,
            "0b01": 0,
            "0b10": 1,
        }

        if block_type not in block_types:
            raise ValueError("Invalid block type")

        # fixed huffman codes
        if block_type == "0b01":
            (
                static_LL_huffman_codes,
                static_D_huffman_codes,
                static_LL_symbols_map,
                static_D_symbols_map,
            ) = initialize_static_huffman_codes()

            # print(static_LL_huffman_codes)
            dumb = This_is_a_bad_idea(
                block_data.tobytes(),
                static_LL_huffman_codes,
                static_D_huffman_codes,
                LL_symbols_map=static_LL_symbols_map,
                D_symbols_map=static_D_symbols_map,
            )
            compressed_data = dumb.run_LZSS_with_PrefixCodes()
            self.push_bits(compressed_data.bin)

        # dynamic huffman
        elif block_type == "0b10":
            self.push_bits(
                bin(286 - 257), lsb_first=True, pad_amount=5
            )  # HLIT , using all of them icba
            self.push_bits(bin(30 - 1), lsb_first=True, pad_amount=5)  # HDIST
            self.push_bits(bin(19 - 4), lsb_first=True, pad_amount=4)  # HCLEN
            self.lz77 = LZ77Encoder()
            self.lz77.compress(block_data.tobytes())
            self.huffman = HuffmanEncoder(
                self.lz77.LL_symbols_map, self.lz77.D_symbols_map
            )
            # print("EOB sanity check (huffman codes): ", self.huffman.LL_codes[256])
            # lord forgive me for what I must do but there is no time
            # instead of doing the RIGHT thing and parsing the lz77 output
            # i will just RERUN the lzSS algorithm but now with huffman codes
            # this doubles the time complexity but i can't anymore
            CL_headers = self.huffman.CL_lens_header
            LL_D_headers = self.huffman.LL_D_header

            # print("\nCL headers: ", CL_headers)
            # print("\nLL_D headers: ", LL_D_headers)
            # print("\nCL codes: ", self.huffman.CL_codes)
            for symbol in CL_headers:
                self.push_bits(bin(symbol), lsb_first=True, pad_amount=3)

            # print("\nLL CODES", self.huffman.LL_codes)
            # print("\nD CODES", self.huffman.D_codes)
            # print("\nCL CODES ", self.huffman.CL_codes)
            # print("\nCL HEADER:", CL_headers)
            # print("\nLLD HEADER:", LL_D_headers)

            for symbol, offset in LL_D_headers:
                # print(symbol, offset)
                self.push_bits(symbol)
                if offset:
                    self.push_bits(offset, lsb_first=True)

            dumb = This_is_a_bad_idea(
                block_data.tobytes(),
                self.huffman.LL_codes,
                self.huffman.D_codes,
                LL_symbols_map=self.lz77.LL_symbols_map,
                D_symbols_map=self.lz77.D_symbols_map,
            )
            compressed_data = dumb.run_LZSS_with_PrefixCodes()
            self.push_bits(compressed_data.bin)
        else:
            # block type 0b00
            block_len = bin(len(block_data.bytes))
            negated_block_len = bin(len(block_data.bytes) ^ 0xFFFF)
            # print(block_len, negated_block_len)
            self.push_bits("0b00000")
            self.push_bits(block_len, lsb_first=True, pad_amount=16)
            self.push_bits(negated_block_len, lsb_first=True, pad_amount=16)
            self.output_buffer.append(block_data)

    def deflate(self):
        if len(self.input_buffer) <= 128:
            self.write_block(self.input_buffer, is_last=1, block_type="0b00")
        else:
            self.write_block(self.input_buffer, is_last=1, block_type="0b10")
        if 2 > 3:
            self.write_block(self.input_buffer, is_last=1, block_type="0b01")

        self.pad_body()
        return self.output_buffer


# yeah you can use any prefix codes but we're using huffman,
# ok actually its fixed max length huffman with package-merge
class Node:
    def __init__(
        self,
        symbol=None,
        freq=0,
        left=None,
        right=None,
        real_symbol=None,
    ):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self.real_symbol = real_symbol

    def __lt__(self, other):
        if self.freq < other.freq:
            return True
        elif self.freq == other.freq:
            return self.real_symbol < other.real_symbol
        return False

    def __repr__(self):
        return f"Node(symbol={self.symbol}, freq={self.freq})"


class HuffmanTree:
    def __init__(self, symbols_map, max_length=15):
        # print("\nSYMBOLS: ", symbols_map)
        self.symbol_map = symbols_map
        self.root = self._package_merge(symbols_map, max_length)
        self.codes = self._generate_huffman_codes(self.root)
        self._standard_algo()
        self._add_extra_bits()

    def _standard_algo(self):
        self.codes = {
            k: v
            for k, v in sorted(
                self.codes.items(),
                key=lambda item: (
                    len(item[1]["code"]),
                    self.symbol_map[item[0]].symbol,
                ),
            )
        }
        # print(self.codes)
        code_lengths = [elem["code"] for elem in self.codes.values()]
        code_lengths = [len(i) for i in code_lengths]

        max_length = max(code_lengths)
        length_counts = [0] * (max_length + 1)
        for length in code_lengths:
            length_counts[length] += 1

        code = 0
        length_counts[0] = 0
        next_code = [0] * (max_length + 1)
        for bits in range(1, max_length + 1):
            code = (code + length_counts[bits - 1]) << 1
            next_code[bits] = code

        code_table = [(0, 0)] * len(code_lengths)
        for n in range(len(code_lengths)):
            length = code_lengths[n]
            if length != 0:
                code_table[n] = (length, next_code[length])
                next_code[length] += 1

        # print(code_table)
        for i, symbol in enumerate(self.codes.keys()):
            self.codes[symbol]["code"] = bin(code_table[i][1])[2:].zfill(
                code_table[i][0]
            )

        # print(self.codes)
        # print("\n")
        return code_table

    def _package_merge(self, frequencies, max_length=15):
        leaves = [
            Node(symbol, freq=data.freq, real_symbol=data.symbol)
            for symbol, data in frequencies.items()
        ]
        heapq.heapify(leaves)

        while len(leaves) > 1:
            left = heapq.heappop(leaves)
            right = heapq.heappop(leaves)
            new_node = Node(
                symbol=None,
                freq=left.freq + right.freq,
                real_symbol=left.real_symbol + right.real_symbol,
                left=left,
                right=right,
            )
            heapq.heappush(leaves, new_node)

        root = leaves[0]
        root.left, root.right = root.right, root.left
        return root

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

        return code_map

    def _add_extra_bits(self):
        values = []
        for symbol in self.codes.keys():
            self.codes[symbol]["extra_bits"] = self.symbol_map[symbol].extra_bits
            if type(symbol) == str:
                values.append(symbol)
        # i carried these fucking values all the way along just to convert them here god dammit
        # i am a moron
        for value in values:
            self.codes[self.symbol_map[value].symbol] = self.codes.pop(value)


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

        prev_symbol = -1
        count = 0

        for symbol in combined_lengths + [-1]:  # Add sentinel at end
            if symbol == prev_symbol:
                count += 1
            else:
                if prev_symbol != -1:
                    if prev_symbol == 0:
                        while count > 0:
                            if count <= 2:
                                CL_codes.append((prev_symbol, -1))  # Emit as is
                                count -= 1
                            elif count <= 10:
                                CL_codes.append((17, count - 3))  # Use code 17
                                count = 0
                            else:
                                CL_codes.append(
                                    (18, min(count, 138) - 11)
                                )  # Use code 18
                                count -= min(count, 138)
                    else:
                        # Emit the previous symbol as many times as it appeared
                        for _ in range(count):
                            CL_codes.append((prev_symbol, -1))
                count = 1
                prev_symbol = symbol

        # print("CL SEQ: ", CL_codes)
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

        CL_codes_table = {}
        for i, freq in enumerate(CL_codes_array):
            if freq > 0:
                CL_codes_table[i] = Code_Data(symbol=i, offset_bits=None, freq=freq)
        self.LL_D_table = CL_codes.copy()

        CL_tree = HuffmanTree(CL_codes_table, max_length=7)
        return CL_tree.codes

    def generate_LL_D_header(self):
        # remember to return and push these to the bitstream (code, extra_bits)
        ll_d_lens = []
        # print("LL_D_table", self.LL_D_table)
        # print("CL CODES", self.CL_codes)
        for symbol, offset in self.LL_D_table:
            CL_symbol = symbol
            extra_bits = offset if offset > -1 else None
            if extra_bits is not None:
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


# yeah it's actually LZSS but the standard keeps calling it LZ77,
# cannot keep back references larger than 32768 bytes because of the standard
# so we keep a window of 32768 bytes and a lookahead buffer of 258 bytes,
# because i didn't implement block splitting and everything is
# one big block that is only aware of the data in it's block
# and can't reference data from previous blocks
# but that's good because they don't exist
class LZ77Encoder:
    def __init__(
        self,
        window_size=2**15,
        lookahead_buffer_size=2**8,
    ):
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
                # print(f"len: {match_length}, dist: {match_distance}")
                # print(window, lookahead)

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


if __name__ == "__main__":
    # BILLBIRD
    test_string = BitStream(b"ABCDEABCD ABCDEABCD")
    test = DeflateEncoder(test_string)
    test.deflate()

    # test_lzss = LZ77Encoder()
    # test_lzss.compress(test_string)
