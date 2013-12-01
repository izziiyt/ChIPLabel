#ifndef prodHHMM_H
#define prodHHMM_H

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <cmath>
#include "baseHHMM.hpp"

using namespace std;
using namespace Eigen;

namespace hhmm{

  class TestHHMM;
  
  using DM = DiagonalMatrix<double,Dynamic>;

  class prodHHMM final:public baseHHMM{
    friend TestHHMM;
  protected:
    VectorXd mean;
    vector<double> testemit;
    DM var;
  public:
    prodHHMM(uint32_t,uint32_t,baseHHMM*);
    ~prodHHMM() = default;
    void setEmit(vector<double> const& x);
    double emit(VectorXd const&) const;
    double emit(uint32_t i) const;
    VectorXd getMean() const;
    MatrixXd getVar() const;
    void cpyMean(VectorXd const&);
    void cpyVar(DM const&);
    void swpMean(VectorXd&);
    void swpVar(DM&);
  };

}

#endif
