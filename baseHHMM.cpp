#include "baseHHMM.h"

namespace hhmm{

  baseHHMM::baseHHMM(uint32_t _level,baseHHMM* _parent)
    :level(_level),
     parent(_parent)
  {}
  
  uint32_t baseHHMM::getLevel() const
  {
    return level;
  }

  long double baseHHMM::getPi() const
  {
    return pi;
  }
  
  long double& baseHHMM::setPi()
  {
    return pi;
  }

  void baseHHMM::setPi(long double x)

  {
    pi = x;
  }

  void baseHHMM::clearParam()
  {
    pi = 0.0;
  }
}
