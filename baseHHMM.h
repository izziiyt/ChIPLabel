#ifndef BASEHHMM_H
#define BASEHHMM_H

#include <cstdint>
#include <vector>

using namespace std;

namespace hhmm{

  class TestHHMM;

  class baseHHMM{
    friend TestHHMM;
  protected:
    uint32_t level;
    //transition probability from this->parent to this
    long double pi;
    //this parameter is used to standardize the pies of child-HHMMs
    //in M-step of EM algorithm.
    long double stdPi;
  public:
    baseHHMM* parent;
    
    baseHHMM(uint32_t,baseHHMM*);
    virtual ~baseHHMM() = default;
    uint32_t getLevel() const;
    long double getPi() const;
    long double& setPi();
    void setPi(long double x);
    virtual void clearParam();
    virtual void check(){}
    virtual void initParam(vector<long double> const&);
    virtual void log(uint32_t,uint32_t);
  };
}

#endif
