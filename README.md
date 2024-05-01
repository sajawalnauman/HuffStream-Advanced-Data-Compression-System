# Huffman Advanced Data Compression System

## Description
This project implements Huffman coding for file compression and decompression. It provides a Python implementation that includes a compression utility, a Huffman coding module, and a set of unit tests to ensure the functionality of the Huffman properties.

## Features
- **File Compression**: Compress files using Huffman coding to reduce file size efficiently.
- **File Decompression**: Decompress previously compressed files to their original state.
- **Testing**: Unit tests to verify the correctness and efficiency of the Huffman compression algorithm.

## Modules
- `compress.py`: Handles the compression and decompression processes.
- `huffman.py`: Implements the Huffman coding algorithm.
- `utils.py`: Provides utility functions that support compression operations.
- `test_huffman_properties_basic.py`: Contains basic unit tests for validating the Huffman coding properties.

## Installation
To run this project, clone this repository and ensure that Python 3 is installed on your system.

```bash
git clone https://github.com/yourusername/huffman-compression.git
cd huffman-compression
```

## Usage
To compress a file, run:
```bash
python compress.py --compress yourfile.txt
```

To decompress a file, run:
```bash
python compress.py --decompress yourfile.huffman
```

## Testing
To run the unit tests, execute:
```bash
python test_huffman_properties_basic.py
```
