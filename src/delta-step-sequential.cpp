#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>
#include <set>
#include <math.h>
#ifndef SOLVER_HEADER
#define SOLVER_HEADER
#include "solver.h"
#include "timing.h"
#endif
#ifndef DELTA_CMN_HEADER
#define DELTA_CMN_HEADER
#include "delta-step-common.h"
#endif

typedef std::pair<int, float> request;

class SequentialDeltaStepping : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;
  // Preprocessed data
  float delta;
  int numBuckets;
  float heaviestEdgeWeight;
  std::vector<std::vector<edge>> lightEdges;
  std::vector<std::vector<edge>> heavyEdges;
  // Updating fields
  std::vector<std::set<int>> buckets; // can use unordered_set for O(1) randomized?

  /**
   * Get the new bucket number for the given distance
  */
  int getBucketNum(float distance) {
    return (int) (distance / delta) % numBuckets;
  }

  /**
   * @param[in] nodes vertices to process
   * @param[in] type light or heavy
   * @return list of requests
  */
  std::vector<request> findRequests(std::set<int> &nodes, EdgeType type, std::vector<float> &distance) {
    std::vector<request> requests;
    std::mutex requestsLock;
    // Idea: increase granularity of tasks by passing multiple nodes at once in order to reduce contention on requests vector
    for (int u : nodes) {
      findOneRequest(u, type, distance, requests, requestsLock);
    }
    return requests;
  }

  /**
  * Helper function for findRequests -- finds all the requests for a single node
  */
  void findOneRequest(int u, EdgeType type, std::vector<float> &distance, std::vector<request> &requests, std::mutex &requestsLock) {
    std::vector<request> req;
    std::vector<std::vector<edge>> &searchEdges = (type == EdgeType::LIGHT) ? lightEdges : heavyEdges;
    for (edge &e : searchEdges[u]) {
      int v = e.dest;
      float w = e.weight;
      req.push_back(std::make_pair(v, distance[u] + w));
    }
    requestsLock.lock();
    requests.insert(requests.end(), req.begin(), req.end());
    requestsLock.unlock();
  }

  /**
   * Like findRequests but directly finds the lowest distance among the requests
   * @param[out] updated set of destination nodes that have 
   * @return lowest distance found among requests to each vertex
  */
  std::map<int, float> findMinNewDists(std::set<int> &nodes, EdgeType type, std::vector<float> &distance) {
    std::map<int, float> minDists;
    // to parallelize, split the nodes among processes (one split at the start?)
    // maybe keep track of local minDists and reduction to get min for each index
    for (int u : nodes) {
      std::vector<std::vector<edge>> &searchEdges = (type == EdgeType::LIGHT) ? lightEdges : heavyEdges;
      for (edge &e : searchEdges[u]) {
        int v = e.dest;
        float w = e.weight;
        float newDist = distance[u] + w;
        if (newDist < minDists[v]) {
          minDists[v] = newDist;
        }
      }
    }
    return minDists;
  }

  void relaxRequests(std::vector<request> &requests, std::mutex *bucketLocks, std::mutex *vertexLocks, std::vector<float> &distance) {
    for (request &req : requests) {
      int v = req.first;
      float dist = req.second;
      // find new min and add to bucket
      // lock vertex
      vertexLocks[v].lock();
      if (dist < distance[v]) {
        int oldBucketNum = getBucketNum(distance[v]);
        int newBucketNum = getBucketNum(dist);
        // lock buckets and move v
        if (distance[v] != INFINITY) {
          bucketLocks[oldBucketNum].lock();
          buckets[oldBucketNum].erase(v);
          bucketLocks[oldBucketNum].unlock();
        }
        distance[v] = dist;
        bucketLocks[newBucketNum].lock();
        buckets[newBucketNum].insert(v);
        bucketLocks[newBucketNum].unlock();
      }
      vertexLocks[v].unlock();
    }
  }

public:
  void deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->numBuckets = (int) std::ceil(heaviestEdgeWeight / this->delta) + 1;
    std::mutex bucketLocks[numBuckets];
    std::mutex vertexLocks[numVertices];
    buckets.resize(numBuckets);
    distance[source] = 0;
    buckets[0].insert(0);
    int lastEmptiedBucket = numBuckets - 1;
    int currentBucket = 0;
    int counter = 0;
    while(currentBucket != lastEmptiedBucket) {
      counter += 1;
      if (!buckets[currentBucket].empty()) {
        std::vector<request> requests;
        std::set<int> deletedNodes;
        // Inner loop
        while (!buckets[currentBucket].empty()) {
          requests = findRequests(buckets[currentBucket], LIGHT, distance);
          deletedNodes.insert(buckets[currentBucket].begin(), buckets[currentBucket].end());
          buckets[currentBucket].clear();
          relaxRequests(requests, bucketLocks, vertexLocks, distance);
        }
        requests = findRequests(deletedNodes, HEAVY, distance);
        relaxRequests(requests, bucketLocks, vertexLocks, distance);
        lastEmptiedBucket = currentBucket;
      }
      currentBucket = (currentBucket + 1) % this->numBuckets;
    }
  }

  void init(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    this->heaviestEdgeWeight = 0;
    this->buckets.clear();
    
    // separate into light and heavy edges
    lightEdges.resize(numVertices);
    heavyEdges.resize(numVertices);
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        float w = e.weight;
        if(w > heaviestEdgeWeight) {
          heaviestEdgeWeight = w;
        }
      }
    }
    this->delta = heaviestEdgeWeight / DELTA_FACTOR;
    for (int u = 0; u < numVertices; u++) {
      lightEdges[u].clear();
      heavyEdges[u].clear();
      for (edge &e : edges[u]) {
        float w = e.weight;
        if (w <= delta) {
          lightEdges[u].push_back(e);
        } else {
          heavyEdges[u].push_back(e);
        }
      }
    }
  }

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    deltaStep(source, edges, distance, predecessor);
  }
};
