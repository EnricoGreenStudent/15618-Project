
CFLAGS := -Wall -m64
TARGET_BELLMAN := bellman
TARGET_DIJKSTRA := dijkstra
TARGETBIN := tester

OBJDIR=objs
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
CUDAOBJS=$(OBJDIR)/delta-step-cuda.o

.phony: all

all: $(TARGET_BELLMAN) $(TARGET_DIJKSTRA) $(TARGETBIN)

$(TARGET_BELLMAN):
	g++ $(CFLAGS) src/bellman.cpp -o $(TARGET_BELLMAN)

$(TARGET_DIJKSTRA):
	g++ $(CFLAGS) src/dijkstra.cpp -o $(TARGET_DIJKSTRA)

$(TARGETBIN):
	g++ $(CFLAGS) src/main.cpp -o $(TARGETBIN)

# Objects for CUDA compilation
$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

cuda: $(CUDAOBJS)


clean:
	rm -f $(TARGET_BELLMAN) $(TARGET_DIJKSTRA) $(TARGETBIN)
