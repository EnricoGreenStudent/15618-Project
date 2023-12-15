#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>
#include <set>
#include <math.h>

#include "delta-step-cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#define THREADS_PER_BLK 256


#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(errCode), file, line);
        if (abort) exit(errCode);
    }
}

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
  cuRequest* relaxRequests;
  int* vRelaxOffsets; // start of relax requests for each vertex in `relaxRequests`
  int* relaxVertices; // vertex numbers for each vertex in `vRelaxOffsets`
  int* isMin;
  cuRequest* output;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ DeltaSteppingConstants constParams;
__constant__ DeltaSteppingRelaxParams constRelaxParams;  // Updates outside of device

struct ReqCmp {
  __device__
  bool operator()(const cuRequest& o1, const cuRequest& o2) {
      return o1.first < o2.first || (o1.first == o2.first && o1.second < o2.second);
  }
};

/**
 * Returns the index among the relaxed vertices for thi thread.
 * 
 * The thread is given the vertices and a cumulative count of
 * the number of relax requests for each vertex in the bucket.
 * 
 * We use binary search to find the vertex this thread should
 * handle, then find which edge of that vertex thread should
 * process for the relaxation.
*/
__device__ __inline__ int search_edge_index() {
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
  return left - 1;
}

/**
 * Each thread calculates the new distance for its own edge
 * We can combine these results by sorting and taking the minimum
 * distance for each destination vertex
*/
__global__ void findRequests(EdgeType type) {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constRelaxParams.numRequests) {
    return;
  }

  edge* searchEdges = (type == EdgeType::LIGHT) ? constParams.lightEdges : constParams.heavyEdges;
  int* searchOffsets = (type == EdgeType::LIGHT) ? constParams.vLightOffsets : constParams.vHeavyOffsets;
  int searchIndex = search_edge_index();
  int u = constRelaxParams.relaxVertices[searchIndex];
  // offset + start of edges for the vertex
  int edgeNum = (nodeID - constRelaxParams.vRelaxOffsets[searchIndex]) + searchOffsets[u];
  edge &e = searchEdges[edgeNum];
  int v = e.dest;
  float w = e.weight;
  cuRequest req;
  req.first = v;
  req.second = constParams.distance[u] + w;
  constRelaxParams.relaxRequests[nodeID] = req;
}

/**
 * Sets value of isMin to find which request outputs correspond
 * to minimal distances for that destination vertex.
 * 
 * Pass results into a prefix sum, then use that to fill output.
 * Also updates the distance values on GPU device memory
*/
__global__ void getIsMinDistance() {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constRelaxParams.numRequests) {
    return;
  }

  constRelaxParams.isMin[nodeID] = 0;
  int v = constRelaxParams.relaxRequests[nodeID].first;
  float dist = constRelaxParams.relaxRequests[nodeID].second;
  if (nodeID == 0 || constRelaxParams.relaxRequests[nodeID - 1].first != v) {
    if (dist < constParams.distance[v]) {
      constParams.distance[v] = dist;
      constRelaxParams.isMin[nodeID] = 1;
    }
  }
}

/**
 * The thread is given prefix summed values in isMin to use as indices
*/
__global__ void collectDistanceUpdates() {
  int nodeID = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeID >= constRelaxParams.numRequests) {
    return;
  }

  int outputIndex = constRelaxParams.isMin[nodeID];
  if (constRelaxParams.isMin[nodeID + 1] != outputIndex) {
    constRelaxParams.output[outputIndex] = constRelaxParams.relaxRequests[nodeID];
  }
}

  /**
   * Get the new bucket number for the given distance
  */
  int ParallelCUDADeltaStepping::getBucketNum(float distance) {
    return (int) (distance / delta) % numBuckets;
  }

  void ParallelCUDADeltaStepping::relaxRequests(std::vector<cuRequest> &requests, std::mutex *bucketLocks, std::mutex *vertexLocks, std::vector<float> &distance) {
    // #pragma omp parallel for
    for (cuRequest &req : requests) {
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

  void ParallelCUDADeltaStepping::cudaFindRequests(std::set<int> &nodes, std::vector<cuRequest> &relaxUpdates, EdgeType type) {
    // Copy bucket vectors, pass the data to device memory
    int bucketSize = nodes.size();
    std::vector<int> &searchOffsets = (type == EdgeType::LIGHT) ? vLightOffsets : vHeavyOffsets;
    std::vector<int> curBucket;
    std::vector<int> bucketOffsets;
    curBucket.reserve(bucketSize);
    bucketOffsets.reserve(bucketSize);
    int curBucketOffset = 0;
    for (int u : nodes) {
      curBucket.push_back(u);
      bucketOffsets.push_back(curBucketOffset);
      curBucketOffset += searchOffsets[u+1] - searchOffsets[u];
    }
    // Total number of requests is curBucketOffset
    bucketOffsets.push_back(curBucketOffset);
    if (curBucketOffset == 0) {
      return;
    }

    // Allocate device memory for finding requests
    cudaMalloc(&cuRelaxRequests, curBucketOffset * sizeof(cuRequest));
    cudaMalloc(&cuVRelaxOffsets, (bucketSize+1) * sizeof(int));
    cudaMalloc(&cuRelaxVertices, bucketSize * sizeof(int));
    cudaMalloc(&cuRelaxIsMin, (curBucketOffset+1) * sizeof(int));
    cudaMalloc(&cuRelaxOutput, curBucketOffset * sizeof(cuRequest));
    // Create offsets for all of the vertices in the current bucket
    cudaMemcpy(cuVRelaxOffsets, bucketOffsets.data(), (bucketSize+1) * sizeof(int),
                cudaMemcpyHostToDevice);
    cudaMemcpy(cuRelaxVertices, curBucket.data(), bucketSize * sizeof(int),
                cudaMemcpyHostToDevice);


    DeltaSteppingRelaxParams relaxParams;
    relaxParams.numRequests = curBucketOffset;
    relaxParams.numRelaxVertices = bucketSize;
    relaxParams.vRelaxOffsets = cuVRelaxOffsets;
    relaxParams.relaxVertices = cuRelaxVertices;
    relaxParams.relaxRequests = cuRelaxRequests;
    relaxParams.isMin = cuRelaxIsMin;
    relaxParams.output = cuRelaxOutput;
    cudaMemcpyToSymbol(constRelaxParams, &relaxParams, sizeof(DeltaSteppingRelaxParams));

    // Find requests using CUDA
    int numReqBlocks = (curBucketOffset + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
    findRequests<<<numReqBlocks, THREADS_PER_BLK>>>(type);
    // cudaCheckError(cudaDeviceSynchronize());
    // Sort relaxations using thrust
/* Note: This uses an execution policy to tell thrust the pointers are in device memory
    thrust::sort(
      thrust::device,
      relaxParams.relaxRequests,
      relaxParams.relaxRequests + relaxParams.numRequests,
      ReqCmp());
*/
    thrust::device_ptr<cuRequest> ptrRelaxRequests(relaxParams.relaxRequests);
    thrust::sort(
      ptrRelaxRequests,
      ptrRelaxRequests + relaxParams.numRequests,
      ReqCmp());
    // cudaCheckError(cudaDeviceSynchronize());
    // Find minimal relaxations
    getIsMinDistance<<<numReqBlocks, THREADS_PER_BLK>>>();
    // cudaCheckError(cudaDeviceSynchronize());
    // Get prefix sums using thrust
    thrust::device_ptr<int> ptrIsMin(relaxParams.isMin);
    thrust::exclusive_scan(ptrIsMin, ptrIsMin + relaxParams.numRequests + 1, ptrIsMin);
    // cudaCheckError(cudaDeviceSynchronize());
    // Collect updates
    collectDistanceUpdates<<<numReqBlocks, THREADS_PER_BLK>>>();
    // cudaCheckError(cudaDeviceSynchronize());
    // Get size of output requests
    int numRelaxUpdates;
    cudaMemcpy(&numRelaxUpdates, &relaxParams.isMin[curBucketOffset], sizeof(int), cudaMemcpyDeviceToHost);
    // Copy output to host
    relaxUpdates.resize(numRelaxUpdates);
    cudaMemcpy(relaxUpdates.data(), relaxParams.output, numRelaxUpdates * sizeof(cuRequest), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(cuRelaxRequests);
    cudaFree(cuVRelaxOffsets);
    cudaFree(cuRelaxVertices);
    cudaFree(cuRelaxIsMin);
    cudaFree(cuRelaxOutput);
  }

  
  void ParallelCUDADeltaStepping::deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    Timer t;
    double setupTime = 0;
    double findTime = 0;
    double relaxTime = 0;
    double dataMovementTime = 0;
    t.reset();
    
    // Algorithm-specific setup
    distance[source] = 0;
    this->numBuckets = (int) std::ceil(heaviestEdgeWeight / this->delta) + 1;
    std::mutex bucketLocks[numBuckets];
    std::mutex vertexLocks[numVertices];
    buckets.resize(numBuckets);
    buckets[0].insert(0);
    int lastEmptiedBucket = numBuckets - 1;
    int currentBucket = 0;
    setupTime = t.elapsed();
    t.reset();

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
    cudaMemcpy(cuHeavyEdges, heavyEdges.data(), heavyEdges.size() * sizeof(edge),
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
    dataMovementTime = t.elapsed();
    // Run algorithm
    while(currentBucket != lastEmptiedBucket) {
      // Repeat light edge relaxations until no vertices placed back in bucket
      if (!buckets[currentBucket].empty()) {
        std::vector<cuRequest> requests;
        std::set<int> deletedNodes;
        // Inner loop
        std::vector<cuRequest> relaxUpdates;
        while (!buckets[currentBucket].empty()) {
          t.reset();
          cudaFindRequests(buckets[currentBucket], relaxUpdates, LIGHT);
          findTime += t.elapsed();
          // Empty current bucket, move all node to new bucket
          deletedNodes.insert(buckets[currentBucket].begin(), buckets[currentBucket].end());
          buckets[currentBucket].clear();
          t.reset();
          relaxRequests(relaxUpdates, bucketLocks, vertexLocks, distance);
          relaxTime += t.elapsed();
        }
        t.reset();
        cudaFindRequests(deletedNodes, relaxUpdates, HEAVY);
        findTime += t.elapsed();
        t.reset();
        relaxRequests(relaxUpdates, bucketLocks, vertexLocks, distance);
        relaxTime += t.elapsed();
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
    printf("CUDA Profiling:\n\tSetup: %f\n\tData movement: %f\n\tFind: %f\n\tRelax: %f\n", setupTime, dataMovementTime, findTime, relaxTime);
  }

  
  void ParallelCUDADeltaStepping::init(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    // Setup specific to loading the graph
    this->source = source;
    this->numVertices = edges.size();
    this->edges = edges;
    this->heaviestEdgeWeight = 0;
    buckets.clear();
    lightEdges.clear();
    heavyEdges.clear();
    vLightOffsets.clear();
    vHeavyOffsets.clear();
    
    // separate into light and heavy edges
    for (int u = 0; u < numVertices; u++) {
      for (edge &e : edges[u]) {
        float w = e.weight;
        if(w > heaviestEdgeWeight) {
          heaviestEdgeWeight = w;
        }
      }
    }
    this->delta = heaviestEdgeWeight / DELTA_FACTOR;
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
  }

  void ParallelCUDADeltaStepping::solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {
    deltaStep(source, edges, distance, predecessor);
  }

