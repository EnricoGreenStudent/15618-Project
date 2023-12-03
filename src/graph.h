#include <vector>

struct edge {
    int dest;
    float weight;
};

struct graph {
    int numVertices;
    std::vector<std::vector<edge>> vertices;
};