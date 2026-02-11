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