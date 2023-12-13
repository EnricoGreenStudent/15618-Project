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

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#define THREADS_PER_BLK 256

/**
 * Type definitions
*/

// typedef std::pair<int, float> request;
typedef struct request_st {
  int first;
  float second;
  bool operator<(const request_st& other) const {
    return this->first < other.first || (this->first == other.first && this->second < other.second);
  }
} request;

enum EdgeType {
  LIGHT = 0,
  HEAVY = 1,
};

/**
 * Global constants
*/

struct DeltaSteppingConstants {
  // Constant values, in shared memory
  int source;
  int numVertices;
  // Preprocessed data
  float delta;
  int numBuckets;
  edge* lightEdges;
  int* vLightOffsets; // start of edges for each vertex in `lightEdges`
  edge* heavyEdges;
  int* vHeavyOffsets; // start of edges for each vertex in `heavyEdge`
  float* distance;
};

struct DeltaSteppingRelaxParams {
  int numRequests;
  int numRelaxVertices;
  request* relaxRequests;
  int* vRelaxOffsets; // start of relax requests for each vertex in `relaxRequests`
  int* relaxVertices; // vertex numbers for each vertex in `vRelaxOffsets`
  int* isMin;
  request* output;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ DeltaSteppingConstants constParams;
__constant__ DeltaSteppingRelaxParams constRelaxParams;  // Updates outside of device

/**
 * Returns the vertex index to relax for the given thread.
 * 
 * The thread is given the vertices and a cumulative count of
 * the number of relax requests for each vertex in the bucket.
 * 
 * We use binary search to find the vertex this thread should
 * handle, then find which edge of that vertex thread should
 * process for the relaxation.
*/
__device__ int search_edge_index(int* searchOffsets) {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;

  int left = 0;
  int right = constRelaxParams.numRelaxVertices;
  while (right > left) {
    int mid = (left + right) / 2;
    if (constRelaxParams.vRelaxOffsets[mid] <= nodeID) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return constRelaxParams.relaxVertices[left - 1];
}

/**
 * Each thread calculates the new distance for its own edge
 * We can combine these results by sorting and taking the minimum
 * distance for each destination vertex
*/
__global__ void findRequests(EdgeType type) {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constParams.numVertices) {
    return;
  }

  edge* searchEdges = (type == EdgeType::LIGHT) ? constParams.lightEdges : constParams.heavyEdges;
  int* searchOffsets = (type == EdgeType::LIGHT) ? constParams.vLightOffsets : constParams.vHeavyOffsets;
  int u = search_edge_index(searchOffsets);
  // offset + start of edges for the vertex
  int edgeNum = (nodeID - constRelaxParams.vRelaxOffsets[left - 1]) + constParams.searchOffsets[u];
  edge &e = searchEdges[edgeNum];
  int v = e.dest;
  float w = e.weight;
  request req;
  req.first = v;
  req.second = constParams.distance[u] + w;
  constRelaxParams.relaxRequests[nodeID] = req;
}

/**
 * Sets value of isMin to find which request outputs correspond
 * to minimal distances for that destination vertex.
 * 
 * Pass results into a prefix sum, then use that to fill output.
*/
__global__ void getIsMinDistance() {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constRelaxParams.numRequests) {
    return;
  }

  constRelaxParams.isMin[nodeID] = 0;
  int v = constRelaxParams.relaxRequests[nodeID].first;
  if (nodeID == 0 || constRelaxParams.relaxRequests[nodeID - 1].first != v) {
    constRelaxParams.isMin[nodeID] = 1;
  }
}

/**
 * Returns the number of output requests.
 * The thread is given prefix summed values in isMin to use as indices
*/
__global__ void collectDistanceUpdates() {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constRelaxParams.numRequests) {
    return;
  }

  int outputIndex = constRelaxParams.isMin[nodeID];
  if (nodeID == 0 || constRelaxParams.isMin[nodeID - 1] != outputIndex) {
    constRelaxParams.output[outputIndex] = constRelaxParams.relaxRequests[nodeID];
  }
}

class ParallelDeltaStepping : public SSSPSolver {
  // Constant values, in shared memory
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;
  // Preprocessed data
  float delta;
  int numBuckets;
  std::vector<std::set<int>> buckets; // can use unordered_set for O(1) randomized?
  // Constant values, in shared memory
  std::vector<edge> lightEdges;
  std::vector<int> vLightOffsets; // start of edges for each vertex in `lightEdges`
  std::vector<edge> heavyEdges;
  std::vector<int> vHeavyOffsets; // start of edges for each vertex in `heavyEdge`

  // DeltaSteppingConstants values
  edge* cuLightEdges;
  int* cuVLightOffsets;
  edge* cuHeavyEdges;
  int* cuVHeavyOffsets;
  float* cuDistance;
  request* cuRelaxRequests;
  int* cuVRelaxOffsets;
  int* cuRelaxVertices;
  int* cuRelaxIsMin;
  request* cuRelaxOutput;

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
  std::vector<request> findRequests(std::set<int> &nodes, EdgeType type, std::vector<float> &distance) {
    std::vector<request> requests;
    std::mutex requestsLock;
    #pragma omp parallel
    #pragma omp single nowait
    // Idea: increase granularity of tasks by passing multiple nodes at once in order to reduce contention on requests vector
    for (int u : nodes) {
      #pragma omp task
      findOneRequest(u, type, distance, requests, requestsLock);
    }
    #pragma omp taskwait
    return requests;
  }
  */

  /**
  * Helper function for findRequests -- finds all the requests for a single node
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
  */

  /**
   * Like findRequests but directly finds the lowest distance among the requests
   * @param[out] updated set of destination nodes that have 
   * @return lowest distance found among requests to each vertex
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
  */

  void relaxRequests(std::vector<request> &requests, std::mutex *bucketLocks, std::mutex *vertexLocks, std::vector<float> &distance) {
    #pragma omp parallel for
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

  void cudaFindRequests(std::set<int> &nodes, std::vector<request> &relaxUpdates, EdgeType type) {
    // Copy bucket vectors, pass the data to device memory
    int bucketSize = nodes.size();
    std::vector<int> curBucket;
    std::vector<int> bucketOffsets;
    curBucket.reserve(bucketSize);
    bucketOffsets.reserve(bucketSize);
    int curBucketOffset = 0;
    for (int u : nodes) {
      curBucket.push_back(u);
      bucketOffsets.push_back(curBucketOffset);
      curBucketOffset += vLightOffsets[u+1] - vLightOffsets[u];
    }
    // Total number of requests is curBucketOffset
    bucketOffsets.push_back(curBucketOffset);
    cudaMalloc(&cuRelaxRequests, curBucketOffset * sizeof(request));
    cudaMalloc(&cuVRelaxOffsets, (bucketSize+1) * sizeof(int));
    cudaMalloc(&cuRelaxVertices, bucketSize * sizeof(int));
    cudaMalloc(&cuRelaxIsMin, curBucketOffset * sizeof(int));
    cudaMalloc(&cuRelaxOutput, curBucketOffset * sizeof(request));
    // Create offsets for all of the vertices in the current bucket
    cudaMemcpy(cuVRelaxOffsets, bucketOffsets.data(), (bucketSize+1) * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(cuRelaxVertices, curBucket.data(), bucketSize * sizeof(int),
                cudaMemcpyHostToDevice);

    DeltaSteppingRelaxParams relaxParams;
    relaxParams.numRequests = curBucketOffset;
    relaxParams.rumRelaxVertices = bucketSize;
    relaxParams.vRelaxOffsets = cuVRelaxOffsets;
    relaxParams.relaxVertices = cuRelaxVertices;
    relaxParams.relaxRequests = cuRelaxRequests;
    relaxParams.isMin = cuRelaxIsMin;
    relaxParams.output = cuRelaxOutput;
    cudaMemcpyToSymbol(constRelaxParams, &relaxParams, sizeof(DeltaSteppingRelaxParams));

    // Find requests using CUDA
    int numVertexBlocks = (bucketSize + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
    findRequests<<<numVertexBlocks, THREADS_PER_BLK>>>(type);
    // Sort relaxations using thrust
    thrust::sort(
      relaxParams.relaxRequests,
      relaxParams.relaxRequests + relaxParams.numRequests,
      thrust::less<request>());
    // Find minimal relaxations
    int numReqBlocks = (curBucketOffset + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
    getIsMinDistance<<<numReqBlocks, THREADS_PER_BLK>>>();
    // Get prefix sums using thrust
    thrust::exclusive_scan(relaxParams.isMin, relaxParams.isMin + relaxParams.numRequests, relaxParams.isMin);
    // Collect updates
    collectDistanceUpdates<<<numReqBlocks, THREADS_PER_BLK>>>();
    // Get size of output requests
    int numRelaxUpdates;
    cudaMemcpy(&numRelaxUpdates, &relaxParams.relaxRequests[curBucketOffset-1], sizeof(int), cudaMemcpyDeviceToHost);
    // Copy output to host
    relaxUpdates.resize(numRelaxUpdates);
    cudaMemcpy(relaxUpdates.data(), &relaxParams.output, numRelaxUpdates * sizeof(request), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(cuRelaxRequests);
    cudaFree(cuVRelaxOffsets);
    cudaFree(cuRelaxVertices);
    cudaFree(cuRelaxIsMin);
    cudaFree(cuRelaxOutput);
  }

public:
  void deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    float heaviestEdgeWeight = 0;
    
    // separate into light and heavy edges
    #pragma omp parallel for
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        float w = e.weight;
        if(w > heaviestEdgeWeight) {
          heaviestEdgeWeight = w;
        }
      }
    }
    this->delta = heaviestEdgeWeight / 10;
    int lightIndex = 0;
    int heavyIndex = 0;
    for (int u = 0; u < numVertices; u++) {
      vLightOffsets.push_back(lightIndex);
      vHeavyOffsets.push_back(heavyIndex);
      for (edge &e : edges[u]) {
        // int v = e.dest;
        float w = e.weight;
        if (w <= delta) {
          lightEdges.push_back(e);
          lightIndex++;
        } else {
          heavyEdges.push_back(e);
          heavyIndex++;
        }
      }
    }
    vLightOffsets.push_back(lightIndex);
    vHeavyOffsets.push_back(heavyIndex);

    // Initialize CUDA memory for constant params
    cudaMalloc(&cuLightEdges, lightEdges.size() * sizeof(edge));
    cudaMalloc(&cuVLightOffsets, (numVertices+1) * sizeof(int));
    cudaMalloc(&cuHeavyEdges, heavyEdges.size() * sizeof(edge));
    cudaMalloc(&cuVHeavyOffsets, (numVertices+1) * sizeof(int));
    cudaMalloc(&cuDistance, numVertices * sizeof(float));
    cudaMemcpy(cuLightEdges, lightEdges.data(), lightEdges.size() * sizeof(edge),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuVLightOffsets, vLightOffsets.data(), (numVertices+1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuHeavyEdges, heavyEdges.data(), lightEdges.size() * sizeof(edge),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuVHeavyOffsets, vHeavyOffsets.data(), (numVertices+1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuDistance, distance.data(), numVertices * sizeof(float),
               cudaMemcpyHostToDevice);

    DeltaSteppingConstants params;
    params.source = source;
    params.numVertices = numVertices;
    params.delta = delta;
    params.numBuckets = numBuckets;
    params.lightEdges = cuLightEdges;
    params.vLightOffsets = cuVLightOffsets;
    params.heavyEdges = cuHeavyEdges;
    params.vHeavyOffsets = cuVHeavyOffsets;
    params.distance = cuDistance;
    cudaMemcpyToSymbol(constParams, &params, sizeof(DeltaSteppingConstants));

    // Run algorithm
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
      // Repeat light edge relaxations until no vertices placed back in bucket
      if (!buckets[currentBucket].empty()) {
        std::vector<request> requests;
        std::set<int> deletedNodes;
        // Inner loop
        std::vector<request> relaxUpdates;
        while (!buckets[currentBucket].empty()) {
          cudaFindRequests(buckets[currentBucket], relaxUpdates, LIGHT);

          // Empty current bucket, move all node to new bucket
          deletedNodes.insert(buckets[currentBucket].begin(), buckets[currentBucket].end());
          buckets[currentBucket].clear();
          relaxRequests(relaxUpdates, bucketLocks, vertexLocks, distance);
          relaxUpdates.clear();
        }
        // requests = findRequests(deletedNodes, HEAVY, distance);
        cudaFindRequests(deletedNodes, relaxUpdates, HEAVY);
        relaxRequests(requests, bucketLocks, vertexLocks, distance);
        lastEmptiedBucket = currentBucket;
      }
      currentBucket = (currentBucket + 1) % this->numBuckets;
    }

    // Free CUDA device memory
    cudaFree(cuLightEdges);
    cudaFree(cuVLightOffsets);
    cudaFree(cuHeavyEdges);
    cudaFree(cuVHeavyOffsets);
    cudaFree(cuDistance);
  }

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) override {
    deltaStep(source, edges, distance, predecessor);
  }
};
