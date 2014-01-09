#ifndef UPPERTRIANGLE_H
#define UPPERTRIANGLE_H

#include <vector>
#include <iostream>

using namespace std;

template <typename T>
using diVector = vector<vector<T>>;

namespace hhmm{

  template<typename T>
  class upperTriangle{
  protected:
    diVector<T> val;
  public:
    upperTriangle() = default;
    upperTriangle(uint32_t len,T x);
    T& operator()(uint32_t x,uint32_t y);
    void print() const;
    void clear();
  };

}

#endif
