#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>
#include "solver.h"

typedef std::pair<int, float> request;

enum EdgeType {
  LIGHT = 0,
  HEAVY = 1,
};

class ParallelDeltaStepping : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  float delta;
  std::vector<std::vector<edge>> edges;
  std::vector<std::vector<edge>> lightEdges;
  std::vector<std::vector<edge>> heavyEdges;

  std::vector<request> findRequests(std::vector<int> bucket, std::vector<float> &distance, EdgeType type) {
    std::vector<request> requests;
    for (int u : bucket) {
      std::vector<std::vector<edge>> &searchEdges = (type == EdgeType::LIGHT) ? lightEdges : heavyEdges;
      for (edge &e : searchEdges[u]) {
        int v = e.dest;
        float w = e.weight;
        requests.push_back(std::make_pair(v, w));
      }
    }
  }

public:
  void deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    // TODO: choose a good delta somehow?
    this->delta = 1;
    
    // separate into light and heavy edges
    lightEdges.resize(numVertices);
    heavyEdges.resize(numVertices);
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        int v = e.dest;
        float w = e.weight;
        if (w <= delta) {
          this->lightEdges[v].push_back(e);
        } else {
          this->heavyEdges[v].push_back(e);
        }
      }
    }
    // TODO: take the lowest non-empty bucket
    distance[source] = 0;
  }

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    deltaStep(source, edges, distance, predecessor);
  }
};

int main() {
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
  ParallelDeltaStepping solver;
  solver.solve(source, edges, distance, predecessor);
  // print results
  for (int u = 0; u < numVertices; u++) {
    std::cout << "vert " << u << " dist " << distance[u] << " pred " << predecessor[u] << "\n";
  }
}
