#ifndef HHMM_H
#define HHMM_H

#include <vector>
#include <cstdint>
#include <memory>
#include <Eigen/Dense>
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
    vector<Sequence> seq;
    template <typename T>
    void setIterator(myit<T>&,myit<T>&,myit<baseHHMM>&,myit<baseHHMM>&,T*,baseHHMM*);
    void forward(Sequence&,baseHHMM*,tree<parameters>*);
    void backward(Sequence&,baseHHMM*,tree<parameters>*);
    void auxIn(Sequence&,baseHHMM*,tree<parameters>*);
    void auxOut(Sequence&,baseHHMM*,tree<parameters>*);

  public:
    HHMM(uint32_t,uint32_t,uint32_t);

  };
}

#endif









