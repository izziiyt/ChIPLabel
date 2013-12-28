#ifndef nprodHHMM_H
#define nprodHHMM_H

#include <vector>
#include <cstdint>
#include <memory>
#include <Eigen/Dense>
#include <map>
#include "baseHHMM.hpp"
#include "prodHHMM.hpp"

using namespace std;
using namespace Eigen;

namespace hhmm{

  template <typename T>
  using up = unique_ptr<T>;

  class TestHHMM;

  class nprodHHMM final:public baseHHMM{
    friend TestHHMM;
  protected:
    MatrixXd transMat;
  public:
    vector<up<baseHHMM>> children;
    map<uint64_t,uint32_t> convert;

    nprodHHMM(uint32_t,uint32_t,uint32_t);
    nprodHHMM(uint32_t,uint32_t,uint32_t,uint32_t,baseHHMM*);
    ~nprodHHMM() = default;
    double trans(up<baseHHMM> const&);
    double trans(baseHHMM*);
    double trans(const up<baseHHMM>&,const up<baseHHMM>&);
    double trans(baseHHMM*,const up<baseHHMM>&);
    double trans(const up<baseHHMM>&,baseHHMM*);
    void cpyTransMat(MatrixXd const&);
  };

}

#endif
