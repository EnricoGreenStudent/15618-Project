#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>
#include <set>
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
  std::vector<std::vector<edge>> edges;
  // Preprocessed data
  float delta;
  int numBuckets;
  std::vector<std::vector<edge>> lightEdges;
  std::vector<std::vector<edge>> heavyEdges;
  // Updating fields
  std::vector<float> distance;
  std::vector<std::set<int>> buckets; // can use unordered_set for O(1) randomized?
  std::vector<std::mutex> bucketLocks; // TODO: this probably runs into issues might need to do the same thing
  std::vector<std::mutex> vertexLocks;

  /**
   * Get the new bucket number for the given distance
  */
  int getBucketNum(float distance) {
    return (int) (distance / delta) % (numBuckets * delta);
  }

  /**
   * @param[in] bucketnum bucket index
   * @param[in] type light or heavy
  */
  std::vector<request> findRequests(int bucketNum, EdgeType type) {
    std::vector<request> requests;
    std::set<int> &bucket = buckets[bucketNum];
    for (int u : bucket) {
      std::vector<std::vector<edge>> &searchEdges = (type == EdgeType::LIGHT) ? lightEdges : heavyEdges;
      for (edge &e : searchEdges[u]) {
        int v = e.dest;
        float w = e.weight;
        requests.push_back(std::make_pair(v, w));
      }
    }
  }

  void relaxRequests(std::vector<request> &requests) {
    for (request &req : requests) {
      int v = req.first;
      float dist = req.second;
      // find new min and add to bucket
      // lock vertex
      vertexLocks[v].lock();
      if (dist < distance[v]) {
        int oldBucketNum = getBucketNum(distance[v]);
        int newBucketNum = getBucketNum(dist);
        distance[v] = dist;
        // lock buckets and move v
        bucketLocks[oldBucketNum].lock();
        buckets[oldBucketNum].erase(v);
        bucketLocks[oldBucketNum].unlock();
        bucketLocks[newBucketNum].lock();
        buckets[newBucketNum].insert(v);
        bucketLocks[newBucketNum].unlock();
      }
      vertexLocks[v].unlock();
    }
  }

public:
  void deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    // TODO: choose a good delta somehow?
    this->delta = 1;
    float heaviestEdgeWeight = 0;
    
    // separate into light and heavy edges
    lightEdges.resize(numVertices);
    heavyEdges.resize(numVertices);
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        int v = e.dest;
        float w = e.weight;
        if (w > heaviestEdgeWeight) {
          heaviestEdgeWeight = w;
        }
        if (w <= delta) {
          this->lightEdges[v].push_back(e);
        } else {
          this->heavyEdges[v].push_back(e);
        }
      }
    }
    this->numBuckets = (int) std::ceil(heaviestEdgeWeight / this->delta) + 1;
    for(int i = 0; i < numBuckets; i++) {
      std::set<int> bucket;
      this->buckets.push_back(bucket);
    }
    // TODO: Relax source vertex
    int lastEmptiedBucket = this->numBuckets - 1;
    int currentBucket = 0;
    while(currentBucket != lastEmptiedBucket) {
      if (!this->buckets[currentBucket].empty()) {
        std::map<int> deletedNodes;
        // Inner loop
        std::vector<request> requests = findRequests(deletedNodes, HEAVY);
        relaxRequests(requests);
        lastEmptiedBucket = currentBucket;
      }
      currentBucket = (currentBucket + 1) % this->numBuckets;
    }
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
