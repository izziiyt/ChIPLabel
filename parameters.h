#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include "upperTriangle.hpp"
#include "nprodHHMM.hpp"
#include "prodHHMM.hpp"

using namespace std;
using namespace Eigen;

template<typename T>
using up = unique_ptr<T>;

template<typename T>
using diVector = vector<vector<T>>;

using MatrixXld = Matrix<long double,Dynamic,Dynamic>;
using VectorXld = Matrix<long double,Dynamic,1>;

namespace hhmm{

  class TestHHMM;
  class baseHHMM;

  class parameters{
    friend TestHHMM;
  private:
    MatrixXld xiContent;
    MatrixXld tmpTransContent;//auxiliary variables for parameter's alteration
  public:
    upperTriangle<long double> alpha;
    upperTriangle<long double> beta;
    vector<long double> etaIn;
    vector<long double> etaOut;
    vector<long double> chi;
    vector<long double> gammaIn;
    vector<long double> gammaOut;
    parameters* parent;
    vector<up<parameters>> children;
    
    long double tmpPi;//auxiliary variables for parameter's alteration
    long double tmpEmitParent;//auxiliary variables for parameter's alteration
    VectorXld tmpMean;//auxiliary variables for parameter's alteration
    DM tmpVariance;
    //vector<long double> tmpEmit;
    
    upperTriangle<long double> delta;
    upperTriangle<uint32_t> phi;
    upperTriangle<uint32_t> tau;
    
    long double& xi(uint32_t x,baseHHMM* y,nprodHHMM* z);
    long double& tmpTrans(baseHHMM* x,baseHHMM* y,nprodHHMM* z);
    MatrixXld& tmpTrans();

    parameters(vector<uint32_t> const&,uint32_t,uint32_t,uint32_t,parameters*);
    parameters(uint32_t,uint32_t,uint32_t,parameters*);
    parameters(vector<uint32_t> const&,uint32_t,uint32_t);
    void transform();
    ~parameters() = default;
   };

}

#endif





