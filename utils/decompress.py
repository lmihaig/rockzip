import gzip
from pathlib import Path
import shutil


"""
do
    read block header from input stream.
    if stored with no compression
        skip any remaining bits in current partially
            processed byte
        read LEN and NLEN (see next section)
        copy LEN bytes of data to output
    otherwise
        if compressed with dynamic Huffman codes
            read representation of code trees (see
            subsection below)
        loop (until end of block code recognized)
            decode literal/length value from input stream
            if value < 256
            copy value (literal byte) to output stream
            otherwise
            if value = end of block (256)
                break from loop
            otherwise (value = 257..285)
                decode distance from input stream

                move backwards distance bytes in the output
                stream, and copy length bytes from this
                position to the output stream.
        end loop
while not last block
"""
# TODO implement decompression from scratch


def decompress(input_file: Path, output_file: Path, keep):
    with gzip.open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        f_out.write(f_in.read())

    new_file_size = output_file.stat().st_size

    if not keep:
        input_file.unlink()

    return new_file_size
