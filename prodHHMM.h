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
    //    vector<double> testemit;
    DM var;
    //MatrixXd var;
  public:
    double emitParent;
    prodHHMM(uint32_t,uint32_t,baseHHMM*);
    ~prodHHMM() = default;
    //void setEmit(vector<double> const& x);
    //    vector<double>& setEmit(){return testemit;}
    double emit(VectorXd const&) const;
    //    double emit(uint32_t i) const;
    VectorXd const& getMean() const;
    VectorXd& setMean();
    DM const& getVariance() const;
    DM& setVariance();
    // MatrixXd const& getVariance() const;
    // MatrixXd& setVariance();
    void cpyMean(VectorXd const&);
    void cpyVar(DM const&);
    void swpMean(VectorXd&);
    void swpVar(DM&);
    void clearParam();
  };

}

#endif
