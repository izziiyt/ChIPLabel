#ifndef UPPERTRIANGLE_CPP
#define UPPERTRIANGLE_CPP

#include "upperTriangle.h"

namespace hhmm{

  template<typename T>
  void upperTriangle<T>::clear(){
    for(auto& i:val){
      for(auto& j:i){
        j = 0.0;
      }
    }
  }

  template<typename T>
  upperTriangle<T>::upperTriangle(uint32_t len,T x)
  {
    for(uint32_t i=0;i<len;++i){val.push_back(vector<T>(len-i,x));}
    for_each(begin(val),end(val),[](vector<T>& y){y.shrink_to_fit();});
    val.shrink_to_fit();
  }

  template<typename T>
  T& upperTriangle<T>::operator()(uint32_t x,uint32_t y){
    return val[x][y-x];
  }

  template<typename T>
  void upperTriangle<T>::print() const{
    for(uint32_t i=0;i<val.size();++i){
      for(uint32_t j=0;j<val.size();++j){
        if(j < i){cout << 0 << " ";}
        else{cout << val[i][j-i] << " ";}
      }
      cout << endl;
    }
  }

}

#endif
