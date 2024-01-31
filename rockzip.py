import typer
from typing_extensions import Annotated
from pathlib import Path
from utils.compress import compress as gzip_compression
from utils.decompress import decompress as gzip_decompression


def print_stats(before, after):
    print("before\tafter\tratio")
    print(f"{before}\t{after}\t{before/after:.0%}")


def compress_file(filename: Path, output_file: Path, keep: bool):
    if output_file is None:
        output_file = filename.with_suffix(".gz")

    print(f"Compressing {filename} to {output_file}, keep={keep}")

    new_fs = gzip_compression(filename, output_file)
    return new_fs


def decompress_file(filename: Path, output_file: Path, keep: bool):
    if filename.suffix != ".gz":
        raise typer.BadParameter("Only .gz files can be decompressed")

    if output_file is None:
        output_file = filename.with_suffix("")

    print(f"Decompressing {filename} to {output_file}, keep={keep}")

    new_fs = gzip_decompression(input_file=filename, output_file=output_file, keep=keep)
    return new_fs


def main(
    filename: Annotated[
        Path,
        typer.Argument(
            help="file to de/compress",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    decompress: Annotated[bool, typer.Option("-d", help="decompress mode")] = False,
    compress: Annotated[bool, typer.Option("-c", help="compress mode")] = False,
    output_file: Annotated[
        Path, typer.Option("-o", help="file to output result to")
    ] = None,
    keep: Annotated[
        bool, typer.Option("-k", help="keep input file [default False]")
    ] = False,
):
    """
    Compress or decompress FILENAME.
    By default FILENAME.gz is created; use -o to specify output filename.
    """

    initial_size = filename.stat().st_size

    if initial_size == 0:
        raise typer.BadParameter("File is empty")

    if decompress:
        res_size = decompress_file(filename, output_file, keep)
    else:
        res_size = compress_file(filename, output_file, keep)

    print_stats(initial_size, res_size)


if __name__ == "__main__":
    typer.run(main)
