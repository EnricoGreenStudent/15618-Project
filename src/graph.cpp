#include <vector>

struct edge {
    int dest;
    int weight;
};

struct graph {
    int numVertices;
    std::vector<std::vector<edge>> vertices;
};