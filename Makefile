## Following: https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually

## Compiler and flags
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -fopenmp 

LIB = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	  ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
	  ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp \
	  -lpthread -lm -ldl

# Source files
SRC = cpp/starm.cpp 
# Output shared library
TARGET = pystarm$(shell python3-config --extension-suffix)

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(shell python3 -m pybind11 --includes) -I${MKLROOT}/include $< -o $@ $(LIB)

# Clean target
clean:
	rm -f $(TARGET)
	rm -f *.so
	rm -rf build

.PHONY: all clean
