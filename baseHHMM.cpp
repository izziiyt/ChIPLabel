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

  double baseHHMM::getPi() const
  {
    return pi;
  }

  void baseHHMM::setPi(double x)

  {
    pi = x;
  }

}