#include <vector>
#include <mutex>
#include <set>

#ifndef SOLVER_HEADER
#define SOLVER_HEADER
#include "solver.h"
#include "timing.h"
#endif
#ifndef DELTA_CMN_HEADER
#define DELTA_CMN_HEADER
#include "delta-step-common.h"
#endif

/**
 * Type definitions
*/

typedef struct cuRequest_st {
  int first;
  float second;
  bool operator<(const cuRequest_st& other) const {
    return this->first < other.first || (this->first == other.first && this->second < other.second);
  }
} cuRequest;

/**
 * Class definition
*/
class ParallelCUDADeltaStepping : public SSSPSolver {
  // Constant input values
  int source;
  int numVertices;
  std::vector<std::vector<edge>> edges;
  // Delta-stepping parameters
  float delta;
  int numBuckets;
  float heaviestEdgeWeight;
  std::vector<std::set<int>> buckets; // can use unordered_set for O(1) randomized?
  // Preprocessed edges
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
  cuRequest* cuRelaxRequests;
  int* cuVRelaxOffsets;
  int* cuRelaxVertices;
  int* cuRelaxIsMin;
  cuRequest* cuRelaxOutput;
  

  int getBucketNum(float distance);
  void relaxRequests(std::vector<cuRequest> &requests, std::mutex *bucketLocks, std::mutex *vertexLocks, std::vector<float> &distance);
  void cudaFindRequests(std::set<int> &nodes, std::vector<cuRequest> &relaxUpdates, EdgeType type);

public:
  void deltaStep(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor);

  void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor);
};
