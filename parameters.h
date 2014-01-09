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

namespace hhmm{

  class TestHHMM;
  class baseHHMM;

  class parameters{
    friend TestHHMM;
  private:
    MatrixXd xiContent;
    MatrixXd tmpTransContent;//auxiliary variables for parameter's alteration
  public:
    upperTriangle<double> alpha;//forward variables
    upperTriangle<double> beta;//backward ward variables
    vector<double> etaIn;//auxiliary variables
    vector<double> etaOut;
    vector<double> chi;
    vector<double> gammaIn;
    vector<double> gammaOut;
    parameters* parent;
    vector<up<parameters>> children;
    
    double tmpPi;//auxiliary variables for parameter's alteration
    double tmpEmitParent;//auxiliary variables for parameter's alteration
    VectorXd tmpMean;//auxiliary variables for parameter's alteration
    DM tmpVariance;
    //vector<double> tmpEmit;
    
    double& xi(uint32_t x,baseHHMM* y,nprodHHMM* z);
    double& tmpTrans(baseHHMM* x,baseHHMM* y,nprodHHMM* z);
    MatrixXd& tmpTrans();

    parameters(uint32_t,uint32_t,uint32_t,uint32_t,parameters*);
    parameters(uint32_t,uint32_t,uint32_t,parameters*);
    parameters(uint32_t,uint32_t,uint32_t,uint32_t);
    ~parameters() = default;
   };

}

#endif





