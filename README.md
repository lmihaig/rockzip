# rockzip

## primitive implementation of **gun**zip

<img src='rockzip.png' width='128'>

**rockzip** is a python CLI tool to compress and decompress files in the gzip format.

First use:
`pip install -r requirements.txt`

Usage:

```
$ python rockzip.py --help
Usage: rockzip.py [OPTIONS] FILENAME

  Compress or decompress FILENAME. By default FILENAME.gz is created; use -o
  to specify output filename.

Arguments:
  FILENAME  file to de/compress  [required]

Options:
  -d       decompress mode
  -c       compress mode
  -o TEXT  file to output result to
  -k       keep input file [default False]
  --help   Show this message and exit.
```

only implemented DEFLATE algorithm
only implemented block type 2 compression (dynamic huffman codes) and only in one big block (and block type 0 which is uncompressed)

corpus:

- caragiale.txt : Nuvele volume of Ion Luca Caragiale as provided by Project Gutenberg Org
- shakespare.txt : The Complete Works of William Shakespeare as provided by Project Gutenberg Org
- kennedy.xls : Excel spreadsheet
- samba.tar : Tarred source code of Samba 2-2.3

<!-- //
- enwik9.xml : the first 10^9 bytes of a specific version of English Wikipedia.
- pi.txt : 10^9 digits of pi in decimal as provided by archive.org
// -->

algorithms to test:

- gzip
- brotli
- zopfli
- snappy
- lz4
- lzma
- bzip2
- zstd
- lzfse sau lzopt doar una dintre ele
- xz

bibliography:

- https://datatracker.ietf.org/doc/html/rfc1952
- https://datatracker.ietf.org/doc/html/rfc1951
- https://www.youtube.com/watch?v=oi2lMBBjQ8s
- https://youtu.be/SJPvNi4HrWQ
- http://www.codersnotes.com/notes/elegance-of-deflate/

note:
this is a very bad implementation from a performance point of view, it should serve as purely educational
