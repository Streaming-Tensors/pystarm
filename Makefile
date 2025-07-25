## Following: https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually

## Compiler and flags
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -fopenmp 

LIB = -m64 -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

# Source files
SRC = cpp/starm.cpp 
# Output shared library
TARGET = pystarm$(shell python3-config --extension-suffix)

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(shell python3 -m pybind11 --includes) -m64 -I${MKLROOT}/include $< -o $@ $(LIB)

# Clean target
clean:
	rm -f $(TARGET)
	rm -f *.so
	rm -rf build

.PHONY: all clean
