import os
import struct
import time
from bitstring import BitStream
from utils.utils import DeflateEncoder
import binascii
from timeit import default_timer as timer


class Compressor:
    def __init__(self, raw_data, output_file):
        self.data = BitStream(raw_data)
        self.bitstream = BitStream()
        self.output_file = output_file
        # TODO implement crc32 calculation from scratch
        # print(raw_data)
        self.crc32 = binascii.crc32(raw_data)

    def write_bytes(self, bits_to_write):
        self.bitstream.append(bits_to_write)

    def build_header(self):
        timestamp = int(time.time())

        # magic num     compression method      flags
        # 1F 8B         08                      00
        # gzip file     deflate                 no flags
        header = bytes([0x1F, 0x8B, 0x08, 0x00])

        # number so LSB first, 4 bytes
        header += struct.pack("<I", timestamp)

        # XFL flags         Atari OS identifier lol
        # 00                05
        header += bytes([0x00, 0x05])
        self.write_bytes(header)

    def build_body(self):
        # yeah you can use other compression algorithms but we're going with deflate
        compressor = DeflateEncoder(self.data)
        deflate = compressor.deflate()
        self.write_bytes(deflate)

    def build_footer(self):
        # these are numbers so LSB first
        footer = struct.pack("<I", self.crc32)
        footer += struct.pack("<I", len(self.data.tobytes()) % (1 << 32))
        # print(bin(self.crc32))
        self.write_bytes(footer)

    def compress(self):
        self.build_header()
        self.build_body()
        self.build_footer()

        gzipped_data = self.bitstream.tobytes()

        with open(self.output_file, "wb") as file:
            file.write(gzipped_data)

        return len(gzipped_data)


def compress(inputfile, outputfile):
    with open(inputfile, "rb") as f:
        data = f.read()

    start = timer()
    compressor = Compressor(data, outputfile)
    file_size_after_compression = compressor.compress()
    finish = timer()

    print(f"File size before compression: {len(data)}")
    print(f"File size after compression: {file_size_after_compression}")
    print(f"Time: {finish - start}")
    return file_size_after_compression


if __name__ == "__main__":
    inputfile = "../tests/mojicule.txt"
    outputfile = "../tests/mojicule_rockzip.gz"
    # $ gzip --k -n -c ../tests/mojicule.txt > mojicule_gzip.gz

    compress(inputfile, outputfile)
    os.system(f"gzip -c -k -n -1 {inputfile} > ../tests/mojicule_gzip.gz")
    os.system(f"file {outputfile}")
    # os.system(f"xxd {outputfile}")
    os.system(
        f"python ../tests/gzstat.py --print-block-codes --decode-blocks < {outputfile}"
    )
