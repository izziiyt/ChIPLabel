#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "parameters.hpp"

using namespace std;
using namespace Eigen;

namespace hhmm{

  class HHMM;
  class TestHHMM;

  class Sequence{
    friend HHMM;
    friend TestHHMM;
  protected:
    uint32_t len;
    vector<VectorXd> V;//Observed sequence.
    vector<uint32_t> testV;//Fix Me.
    parameters param;
  public:

    Sequence(vector<VectorXd> const&,uint32_t,uint32_t,uint32_t);
    Sequence(vector<uint32_t> const&,uint32_t,uint32_t,uint32_t);
    virtual ~Sequence() = default;
    //    uint32_t obs(uint32_t i) const{return testV[i];}
    VectorXd obs(uint32_t i) const{return V[i];}
    uint32_t size() const{return len;}
    
  };
  
}

#endif
