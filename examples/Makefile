all: demo

INCLUDES = -I ../include
CXXARGS = -std=c++11 -O3

demo: demo.cu
	nvcc $(CXXARGS) $(INCLUDES) demo.cu -o demo

clean:
	rm -f *.o demo
