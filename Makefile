
CFLAGS := -Wall
TARGET_BELLMAN := bellman
TARGET_DIJKSTRA := dijkstra
TARGETBIN := tester

.phony: all

all: $(TARGET_BELLMAN) $(TARGET_DIJKSTRA) $(TARGETBIN)

$(TARGET_BELLMAN):
	g++ $(CFLAGS) src/bellman.cpp -o $(TARGET_BELLMAN)

$(TARGET_DIJKSTRA):
	g++ $(CFLAGS) src/dijkstra.cpp -o $(TARGET_DIJKSTRA)

$(TARGETBIN):
	g++ $(CFLAGS) src/main.cpp -o $(TARGETBIN)

clean:
	rm -f $(TARGET_BELLMAN) $(TARGET_DIJKSTRA) $(TARGETBIN)
