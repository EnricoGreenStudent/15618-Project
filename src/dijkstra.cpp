#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <math.h>
#ifndef SOLVER_HEADER
#define SOLVER_HEADER
#include "solver.h"
#include "timing.h"
#endif

class Dijkstra : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;

public:
  void dijkstra(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    std::set<int> visited;
    distance[source] = 0;
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> pq;
    pq.push(std::make_pair(0.0f, source));
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        if(visited.find(u) != visited.end()) {
          continue;
        }
        visited.insert(u);
        // iterate through all edges
        for (edge &e : edges[u]) {
            int v = e.dest;
            if(visited.count(v) != 0) {
              continue;
            }
            int w = e.weight;
            if (distance[u] + w < distance[v]) {
                distance[v] = distance[u] + w;
                predecessor[v] = u;
                pq.push(std::make_pair(distance[v], v));
            }
        }
    }
  }

  void init(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {}

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    dijkstra(source, edges, distance, predecessor);
  }
};