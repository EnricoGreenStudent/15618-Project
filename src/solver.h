#include <vector>
#include "graph.h"

class SSSPSolver {
public:
  virtual void solve(int source, std::vector<std::vector<edge>> &edges, std::vector<float> &distance, std::vector<int> &predecessor) {};
};
