import heapq
from bitstring import BitStream


class This_is_a_bad_idea:
    def __init__(
        self,
        block_input,
        LL_codes,
        D_codes,
        window_size=2**15,
        lookahead_buffer_size=2**8,
        LL_symbols_map=None,
        D_symbols_map=None,
    ):
        self.window_size = window_size
        self.lookahead_buffer_size = lookahead_buffer_size

        self.LL_codes = LL_codes
        self.D_codes = D_codes

        self.output_bitstream = BitStream()

        self.LL_symbols_map = LL_symbols_map
        self.D_symbols_map = D_symbols_map

        self.block_input = BitStream(block_input)
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
                self.output_bitstream.append(self.bit_stack)
                self.bit_stack = []

    def write_value(self, value):
        self.push_bits(value["code"])
        if value["extra_bits"]:
            self.push_bits(value["extra_bits"], lsb_first=True)

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

    def run_LZSS_with_PrefixCodes(self):
        window = bytearray()
        lookahead = bytearray()

        data_len = len(self.block_input.bin)

        for _ in range(self.lookahead_buffer_size):
            if self.block_input.pos < data_len:
                lookahead += self.block_input.read("bytes:1")

        while self.block_input.pos < data_len or len(lookahead) > 0:
            match_length, match_distance = self.find_longest_match(window, lookahead)

            if match_length is not None and match_length > 2:
                # print(match_length, match_distance)
                # print(window, lookahead)

                match_length_symbol = self.LL_symbols_map[str(match_length)].symbol
                match_distance_symbol = self.D_symbols_map[str(match_distance)].symbol

                length_val = self.LL_codes[match_length_symbol]
                distance_val = self.D_codes[match_distance_symbol]

                self.write_value(length_val)
                self.write_value(distance_val)

                for _ in range(match_length):
                    if len(lookahead) == 0:
                        break
                    window += lookahead[:1]
                    window = window[-self.window_size :]
                    lookahead = lookahead[1:]
                    if self.block_input.pos < data_len:
                        lookahead += self.block_input.read("bytes:1")
            else:
                literal = lookahead[:1]

                val = self.LL_codes[ord(literal)]
                self.write_value(val)

                window += literal
                window = window[-self.window_size :]
                lookahead = lookahead[1:]
                if self.block_input.pos < data_len:
                    lookahead += self.block_input.read("bytes:1")

        # WRITING EOB
        self.write_value(self.LL_codes[256])
        self.output_bitstream.append(self.bit_stack)
        return self.output_bitstream


class StaticNode:
    def __init__(
        self,
        symbol=None,
        freq=0,
        left=None,
        right=None,
        real_symbol=None,
        key_func=None,
    ):
        self.key_func = key_func
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self.real_symbol = real_symbol

    def __lt__(self, other):
        if self.symbol is None:
            return False
        elif other.symbol is None:
            return True
        if self.key_func(self.real_symbol) < self.key_func(other.real_symbol):
            return True
        elif self.key_func(self.real_symbol) == self.key_func(other.real_symbol):
            return self.real_symbol < other.real_symbol
        return False

    def __repr__(self):
        return f"Node(symbol={self.symbol}, freq={self.freq})"
