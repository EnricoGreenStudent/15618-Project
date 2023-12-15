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

class ParallelDeltaStepping : public SSSPSolver {
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
   * Helper function for adding requests to local request list
  */
  void findRequestsForVertex(int u, EdgeType type, std::vector<float> &distance, std::vector<request> &localReqs) {
    std::vector<std::vector<edge>> &searchEdges = (type == EdgeType::LIGHT) ? lightEdges : heavyEdges;
    for (edge &e : searchEdges[u]) {
      int v = e.dest;
      float w = e.weight;
      localReqs.push_back(std::make_pair(v, distance[u] + w));
    }
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
    Timer t;
    double setupTime = 0;
    double findTime = 0;
    double relaxTime = 0;
    t.reset();

    this->numBuckets = (int) std::ceil(heaviestEdgeWeight / this->delta) + 1;
    std::mutex bucketLocks[numBuckets];
    std::mutex vertexLocks[numVertices];
    buckets.resize(numBuckets);
    distance[source] = 0;
    buckets[0].insert(0);
    int lastEmptiedBucket = numBuckets - 1;
    int currentBucket = 0;
    setupTime = t.elapsed();
    while(currentBucket != lastEmptiedBucket) {
      if (!buckets[currentBucket].empty()) {
        std::vector<request> requests;
        std::set<int> deletedNodes;
        std::vector<int> nodes;
        // Inner loop
        while (!buckets[currentBucket].empty()) {
          t.reset();
          nodes.assign(buckets[currentBucket].begin(), buckets[currentBucket].end());
          #pragma omp parallel
          {
            std::vector<request> localReqs;
            #pragma omp for
            for (int u : nodes) {
              findRequestsForVertex(u, LIGHT, distance, localReqs);
            }
            #pragma omp barrier
            #pragma omp single
            {
              findTime += t.elapsed();
              deletedNodes.insert(nodes.begin(), nodes.end());
              buckets[currentBucket].clear();
              t.reset();
            }
            #pragma omp barrier
            relaxRequests(localReqs, bucketLocks, vertexLocks, distance);
            #pragma omp single
            relaxTime += t.elapsed();
          }
        }
        t.reset();
        nodes.assign(deletedNodes.begin(), deletedNodes.end());
        #pragma omp parallel
        {
          std::vector<request> localReqs;
          #pragma omp for
          for (int u : nodes) {
            findRequestsForVertex(u, HEAVY, distance, localReqs);
          }
          #pragma omp single
          {
            findTime += t.elapsed();
            t.reset();
          }
          #pragma omp barrier
          relaxRequests(localReqs, bucketLocks, vertexLocks, distance);
          #pragma omp single
          relaxTime += t.elapsed();
        }
        lastEmptiedBucket = currentBucket;
      }
      currentBucket = (currentBucket + 1) % this->numBuckets;
    }
    printf("OpenMP Profiling:\n\tSetup: %f\n\tFind: %f\n\tRelax: %f\n", setupTime, findTime, relaxTime);
  }

  void init(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    // Setup specific to loading the graph
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    this->heaviestEdgeWeight = 0;
    this->buckets.clear();
    
    // separate into light and heavy edges
    lightEdges.resize(numVertices);
    heavyEdges.resize(numVertices);
    #pragma omp parallel for
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        float w = e.weight;
        if(w > heaviestEdgeWeight) {
          heaviestEdgeWeight = w;
        }
      }
    }
    this->delta = heaviestEdgeWeight / DELTA_FACTOR;
    #pragma omp parallel for
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
