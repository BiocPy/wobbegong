#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstring>

#include <zlib.h>
#include <lz4.h> 
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace py = pybind11;

class MMapFile {
public:
    const char* data = nullptr;
    size_t size = 0;
    int fd = -1;

    MMapFile(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Could not open file: " + path);
        
        struct stat sb;
        if (fstat(fd, &sb) == -1) throw std::runtime_error("fstat failed");
        size = sb.st_size;
        
        if (size > 0) {
            data = (const char*)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) throw std::runtime_error("mmap failed");
        }
    }

    ~MMapFile() {
        if (data && size > 0) munmap((void*)data, size);
        if (fd != -1) close(fd);
    }
};

size_t decompress_block(const char* src, size_t src_size, std::vector<char>& dst, const std::string& algo) {
    if (src_size == 0) return 0;

    if (dst.empty()) dst.resize(std::max((size_t)1024, src_size * 4));

    while (true) {
        if (algo == "lz4") {
            int res = LZ4_decompress_safe(src, dst.data(), static_cast<int>(src_size), static_cast<int>(dst.size()));
            if (res >= 0) return static_cast<size_t>(res);
            dst.resize(dst.size() * 2);
        } else {
            uLongf dLen = dst.size();
            int res = uncompress(reinterpret_cast<Bytef*>(dst.data()), &dLen, reinterpret_cast<const Bytef*>(src), src_size);
            
            if (res == Z_OK) return dLen;
            if (res == Z_BUF_ERROR) {
                dst.resize(dst.size() * 2);
            } else {
                throw std::runtime_error("Zlib decompression failed with error code: " + std::to_string(res));
            }
        }
    }
}

py::bytes decompress_wrapper(py::bytes src_bytes, const std::string& compression) {
    std::string src_str = src_bytes;
    std::vector<char> dst;

    size_t out_size;

    {
        py::gil_scoped_release release;
        out_size = decompress_block(
            src_str.data(),
            src_str.size(),
            dst,
            compression
        );
    }

    return py::bytes(dst.data(), out_size);
}

PYBIND11_MODULE(lib_wobbegong, m) {
    m.doc() = "Wobbegong high-performance C++ reader/writer bindings";

    m.def(
        "decompress",
        &decompress_wrapper,
        py::arg("data"),
        py::arg("compression") = "zlib",
        R"pbdoc(
            Decompress a compressed byte buffer.

            Args:
                data:
                    Compressed data buffer.

                compression:
                    Compression algorithm ("zlib" or "lz4").

            Returns:
                Decompressed data.
        )pbdoc"
    );
}