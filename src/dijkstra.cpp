#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <math.h>
#ifndef SOLVER_HEADER
#define SOLVER_HEADER
#include "solver.h"
#endif

class Dijkstra : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;
  std::vector<std::vector<edge>> incomingEdges;

public:
  void dijkstra(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    
    distance[source] = 0;
    std::priority_queue<std::pair<float, int>> pq;
    pq.push(std::make_pair(0.0f, source));
    while (!pq.empty()) {
        // float dist = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        // iterate through all edges
        for (edge &e : edges[u]) {
            int v = e.dest;
            int w = e.weight;
            if (distance[u] + w < distance[v]) {
                distance[v] = distance[u] + w;
                predecessor[v] = u;
                pq.push(std::make_pair(distance[v], v));
            }
        }
    }
  }

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    dijkstra(source, edges, distance, predecessor);
  }
};

/*int main() {
  std::vector<std::vector<edge>> edges;
  std::vector<std::vector<edge>> incomingEdges;
  int numVertices, numEdges, source;
  std::cin >> numVertices >> numEdges >> source;
  edges.resize(numVertices);
  incomingEdges.resize(numVertices);
  for (int i = 0; i < numEdges; i++) {
    int u, v;
    float w;
    std::cin >> u >> v >> w;
    edges[u].push_back(edge{v, w});
    incomingEdges[v].push_back(edge{u, w});
  }
  std::vector<float> distance(numVertices, INFINITY);
  std::vector<int> predecessor(numVertices, -1);
  Dijkstra solver;
  solver.dijkstra(source, edges, distance, predecessor);
  // print results
  for (int u = 0; u < numVertices; u++) {
    std::cout << "vert " << u << " dist " << distance[u] << " pred " << predecessor[u] << "\n";
  }
}*/
