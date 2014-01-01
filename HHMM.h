#ifndef HHMM_H
#define HHMM_H

#include <vector>
#include <cstdint>
#include <memory>
#include <Eigen/Dense>
#include <functional>
#include "Sequence.h"
#include "baseHHMM.hpp"
#include "prodHHMM.hpp"
#include "nprodHHMM.hpp"

using namespace std;
using namespace Eigen;

namespace hhmm{

  template<typename T>
  using myit = typename vector<up<T>>::iterator;

  class TestHHMM;

  class HHMM{
    friend TestHHMM;
  private:
    nprodHHMM root;
    uint32_t dim;
    uint32_t stateNum;
    uint32_t depth;
    vector<up<Sequence>> seq;
    
    template <typename T>
    void setIterator(myit<T>&,myit<T>&,myit<baseHHMM>&,myit<baseHHMM>&,T*,baseHHMM*);
    void forward(Sequence&);
    void forward(Sequence&,baseHHMM*,parameters*);
    void backward(Sequence&);
    void backward(Sequence&,baseHHMM*,parameters*);
    void auxIn(Sequence&);
    void auxIn(Sequence&,baseHHMM*,parameters*);
    void auxOut(Sequence&);
    void auxOut(Sequence&,baseHHMM*,parameters*);
    void horizon(Sequence&);
    void horizon(Sequence&,baseHHMM*,parameters*);
    void vertical(Sequence&);
    void vertical(Sequence&,baseHHMM*,parameters*);
    void calcGamma(Sequence&);
    void calcGamma(Sequence&,baseHHMM*,parameters*);
    double likelihood(Sequence&);
    void EM();
    void paramAssemble(Sequence&,baseHHMM*,parameters*);
    void pramStandardize();
    void clearParam();
    void calcTmpPi(Sequence&);
    void calcTmpPi(Sequence&,baseHHMM*,parameters*);
    void calcTmptrans(Sequence&);
    void calcTmpTrans(Sequence&,baseHHMM*,parameters*);
    void calcTmpEmit(Sequence&);
    void calcTmpEmit(Sequence&,baseHHMM*,parameters*);
  public:
    HHMM(uint32_t,uint32_t,uint32_t);
  };
}

#endif









