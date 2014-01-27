#ifndef HHMM_H
#define HHMM_H

#include <vector>
#include <cstdint>
#include <memory>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <omp.h>
#include "Sequence.hpp"
#include "parameters.hpp"
#include "baseHHMM.hpp"
#include "prodHHMM.hpp"
#include "nprodHHMM.hpp"

using namespace std;
using namespace Eigen;

namespace hhmm{

  template<typename T>
  using myit = typename vector<up<T>>::iterator;

  auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
  auto cast_prod = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};

  class TestHHMM;

  class HHMM{
    friend TestHHMM;
  private:
    nprodHHMM root;
    uint32_t dim;
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
    long double likelihood(Sequence&);

    void paramAssemble(Sequence&,baseHHMM*,parameters*);
    void paramAssemble(Sequence&);
    void varianceAssemble(Sequence&,baseHHMM*,parameters*);
    void varianceAssemble(Sequence&);
    void paramStandardize(baseHHMM*);
    void paramStandardize();
    void varianceStandardize(baseHHMM*);
    void varianceStandardize();
    void clearParam();

    void calcTmpPi(Sequence&);
    void calcTmpPi(Sequence&,baseHHMM*,parameters*);
    void calcTmpTrans(Sequence&);
    void calcTmpTrans(Sequence&,baseHHMM*,parameters*);
    // void calcTmpEmit(Sequence&);
    // void calcTmpEmit(Sequence&,baseHHMM*,parameters*);
    void calcTmpMean(Sequence&);
    void calcTmpMean(Sequence&,baseHHMM*,parameters*);
    void calcTmpVariance(Sequence&);
    void calcTmpVariance(Sequence&,baseHHMM*,parameters*);
    void innerBacktrack(Sequence&,parameters const&,uint32_t,uint32_t,uint32_t,uint32_t);
    void backtrack(Sequence&,parameters const&,uint32_t,uint32_t,uint32_t);
    void viterbi(Sequence&,baseHHMM*,parameters*);

  public:
    HHMM(vector<uint32_t> const&,uint32_t);
    void initParam(vector<long double> const&);
    void EM(uint32_t,string const&);
    void viterbi(Sequence&);
    vector<up<Sequence>>& setSeq(){return seq;};
    void check();
  };
}

#endif









