#ifndef BASEHHMM_H
#define BASEHHMM_H

#include <cstdint>

using namespace std;

namespace hhmm{

  class TestHHMM;

  class baseHHMM{
    friend TestHHMM;
  protected:
    uint32_t level;
    //transition probability from this->parent to this
    double pi;
  public:
    baseHHMM* parent;
    
    baseHHMM(uint32_t,baseHHMM*);
    virtual ~baseHHMM() = default;
    uint32_t getLevel() const;
    double getPi() const;
    double& setPi();
    void setPi(double x);
  };
  
}

#endif
