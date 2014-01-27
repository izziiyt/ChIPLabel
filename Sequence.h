#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "parameters.hpp"
#include <string>
#include <fstream>

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
    vector<VectorXld> V;//Observed sequence.
    vector<uint32_t> state;
    vector<uint32_t> testV;//Fix Me.

  public:
    parameters param;
    Sequence(vector<VectorXld> const&,vector<uint32_t> const&,uint32_t);
    //    Sequence(vector<uint32_t> const&,uint32_t,uint32_t,uint32_t);
    virtual ~Sequence() = default;
    //    uint32_t obs(uint32_t i) const{return testV[i];}
    VectorXld obs(uint32_t i) const{return V[i];}
    uint32_t size() const{return len;}
    void print(string const&) const;
  };
  
}

#endif
