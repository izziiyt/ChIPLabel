#ifndef prodHHMM_H
#define prodHHMM_H

#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include "baseHHMM.hpp"

using namespace std;
using namespace Eigen;

using MatrixXld = Matrix<long double,Dynamic,Dynamic>;
using VectorXld = Matrix<long double,Dynamic,1>;
using DM = DiagonalMatrix<long double,Dynamic>;

namespace hhmm{

  class TestHHMM;

  class prodHHMM final:public baseHHMM{
    friend TestHHMM;
  protected:
    VectorXld mean;
    //    vector<long double> testemit;
    DM var;
  public:
    long double emitParent;
    prodHHMM(uint32_t,uint32_t,baseHHMM*);
    ~prodHHMM() noexcept {};

    //void setEmit(vector<long double> const& x);
    //    vector<long double>& setEmit(){return testemit;}
    long double emit(VectorXld const&) const;
    //    long double emit(uint32_t i) const;
    VectorXld const& getMean() const;
    VectorXld& setMean();
    DM const& getVariance() const;
    DM& setVariance();
    void cpyMean(VectorXld const&);
    void cpyVar(DM const&);
    void swpMean(VectorXld&);
    void swpVar(DM&);
    void clearParam();
    void check();
    void initParam(vector<long double> const&);
  };

}

#endif
