#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>
#include <mutex>
#ifndef SOLVER_HEADER
#define SOLVER_HEADER
#include "solver.h"
#endif

class ParallelBellmanFordForward : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;
  std::vector<std::vector<edge>> incomingEdges;

  void bellmanFordRound(std::vector<float> &distance, std::vector<int> &predecessor, std::mutex *mutexes) {
    // order of updates doesn't matter... issues with cache evictions?
    // parallelize by destination node, thread only writes to one memory location
    #pragma omp parallel for
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        int v = e.dest;
        float w = e.weight;
        mutexes[v].lock();
        if (distance[u] + w < distance[v]) {
          distance[v] = distance[u] + w;
          predecessor[v] = u;
        }
        mutexes[v].unlock();
      }
    }
  }
    /*
    int numParticles = particles.size();
    #pragma omp parallel for if (numLeaves <= 2 && numParticles >= 8*QuadTreeLeafSize)
    for (int idx=0; idx < 4-numLeaves; idx++) {
        int i = nonleaf[idx];
        Vec2 cmin, cmax;
        cmin.x = (i & 1) ? pivot.x : bmin.x;
        cmin.y = (i & 2) ? pivot.y : bmin.y;
        cmax.x = (i & 1) ? bmax.x : pivot.x;
        cmax.y = (i & 2) ? bmax.y : pivot.y;
        quadTreeNode->children[i] = buildQuadTree(childParticles[i], cmin, cmax);
    }
    */
    // #pragma omp parallel for reduction(min:bminx,bminy) reduction(max:bmaxx,bmaxy) schedule(static)
    // #pragma omp parallel for schedule(dynamic, 16) private(nearby)

public:
  void bellmanFord(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    std::mutex mutexes [numVertices];
    
    // preprocess incoming edges (TODO: have input be incoming?);
    incomingEdges.resize(numVertices);
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        int v = e.dest;
        float w = e.weight;
        this->incomingEdges[v].push_back(edge{u, w});
      }
    }
    // run Bellman-Ford round n-1 iterations
    // std::vector<float> distance(numVertices, INFINITY);
    // std::vector<int> predecessor(numVertices, -1);
    distance[source] = 0;
    for (int round = 1; round < numVertices; round++) {
      bellmanFordRound(distance, predecessor, mutexes);
    }
  }

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    bellmanFord(source, edges, distance, predecessor);
  }
};
/*
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
  ParallelBellmanFordForward solver;
  solver.bellmanFord(source, edges, distance, predecessor);
  // print results
  for (int u = 0; u < numVertices; u++) {
    std::cout << "vert " << u << " dist " << distance[u] << " pred " << predecessor[u] << "\n";
  }
}
*/