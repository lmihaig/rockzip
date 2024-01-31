from collections import defaultdict
from bitstring import BitStream, ReadError
import cProfile

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

from line_profiler import LineProfiler

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


class Code_Data:
    def __init__(self, symbol: int, offset_bits: str):
        self.symbol = symbol
        self.extra_bits = offset_bits
        self.freq = 0

    def __str__(self):
        return (
            f"[symbol: {self.symbol}, extra_bits: {self.extra_bits}, freq: {self.freq}]"
        )

    def __repr__(self):
        return self.__str__()


def init_values(code_ranges):
    # example for length but similar for distances
    #        value : symbol, num_bits_extra, extra_bits
    #           20 : 269,     2,             '01'
    codes = defaultdict(Code_Data)
    for symbol, num_bits_extra, lower_bound, upper_bound in code_ranges:
        for i in range(lower_bound, upper_bound + 1):
            # code of symbol, number of extra bits, value of extra bits
            codes[str(i)] = Code_Data(
                symbol=symbol,
                offset_bits=bin(i - lower_bound)[2:].zfill(num_bits_extra)
                if num_bits_extra > 0
                else None,
            )

    return codes


def find_longest_match(window, lookahead):
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


def write_value(value):
    value.freq += 1
    symbol, offset = value.symbol, value.extra_bits
    res = bin(symbol) + offset if offset is not None else bin(symbol)
    return res


def main():
    LL_symbols_map = init_values(length_code_ranges)
    # adding the literals
    for i in range(0, 257):
        LL_symbols_map[i] = Code_Data(symbol=i, offset_bits=None)
    D_symbols_map = init_values(distance_code_ranges)

    window_size = 2**15
    lookahead_buffer_size = 2**8

    # LZSS compression
    compressed_data = BitStream()
    window = bytearray()
    lookahead = bytearray()
    input_data = b"i have seen sense and sensibility"
    # with open("../corpus/caragiale.txt", "rb") as f:
    # input_data = f.read()
    bs = BitStream(input_data)

    for _ in range(lookahead_buffer_size):
        if bs.pos < len(input_data) * 8:
            lookahead += bs.read("bytes:1")

    while bs.pos < len(input_data) * 8 or len(lookahead) > 0:
        match_length, match_distance = find_longest_match(window, lookahead)

        if match_length is not None and match_length > 2:
            compressed_data.append("0b1")  # Match bit '1'
            # print(match_length, match_distance)

            length_val = LL_symbols_map[str(match_length)]
            distance_val = D_symbols_map[str(match_distance)]

            write_value(length_val)
            write_value(distance_val)

            for _ in range(match_length):
                if len(lookahead) == 0:
                    break
                window += lookahead[:1]
                window = window[-window_size:]
                lookahead = lookahead[1:]
                if bs.pos < len(input_data) * 8:
                    lookahead += bs.read("bytes:1")
        else:
            compressed_data.append("0b0")  # Literal bit '0'

            literal = lookahead[:1]
            LL_symbols_map[literal[0]].freq += 1
            compressed_data.append(BitStream(bytes=literal))

            window += literal
            window = window[-window_size:]
            lookahead = lookahead[1:]
            if bs.pos < len(input_data) * 8:
                lookahead += bs.read("bytes:1")

    LL_symbols_map = {
        k: v
        for k, v in sorted(
            LL_symbols_map.items(), key=lambda item: item[1].freq, reverse=True
        )
    }
    D_symbols_map = {
        k: v
        for k, v in sorted(
            D_symbols_map.items(), key=lambda item: item[1].freq, reverse=True
        )
    }

    # WRITING EOB
    compressed_data.append("0b1")
    LL_symbols_map[256].freq += 1
    compressed_data.append(BitStream(bin(256)))

    print(len(compressed_data.bin))
    print(input_data)


if __name__ == "__main__":
    # profiler = LineProfiler()
    # profiler.add_function(find_longest_match)  # Add the function to be profiled
    # profiler.enable()  # Start profiling
    main()
    # profiler.disable()  # Stop profiling
    # profiler.print_stats()  # Print the stats
