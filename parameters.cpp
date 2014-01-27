#ifndef PARAMETERS_CPP
#define PARAMETERS_CPP

#include "parameters.h"

namespace hhmm{

  //constructor for neither the top nor the deepest
  parameters::parameters(vector<uint32_t> const& _stateNum,uint32_t _brosNum,uint32_t _length,uint32_t _dim,parameters* _parent)
    :xiContent(_length,_brosNum+1),
     tmpTransContent(_stateNum[0],_stateNum[0]+1),
     alpha(_length,0.0),
     beta(_length,0.0),
     etaIn(_length,0.0),
     etaOut(_length,0.0),
     chi(_length,0.0),
     gammaIn(_length,0.0),
     gammaOut(_length,0.0),
     parent(_parent)
  {
    if(_stateNum.size() > 1){
      vector<uint32_t> nextNum(++begin(_stateNum),end(_stateNum));
      for(uint32_t t=0;t<_stateNum[0];++t){
        children.push_back(up<parameters>(new parameters(nextNum,_stateNum[0],_length,_dim,this)));
      }
      children.shrink_to_fit();
    }
    else if(_stateNum.size() == 1){
      for(uint32_t t=0;t<_stateNum[0];++t){
        children.push_back(up<parameters>(new parameters(_stateNum[0],_length,_dim,this)));
      }
      children.shrink_to_fit();
    }
  }
  
  //constructor for the deepest
  parameters::parameters(uint32_t _brosNum,uint32_t _length,uint32_t _dim,parameters* _parent)
    :xiContent(_length,_brosNum+1),    
     alpha(_length,0.0),
     beta(_length,0.0),
     etaIn(_length,0.0),
     etaOut(_length,0.0),
     chi(_length,0.0),
     gammaIn(_length,0.0),
     //gammaOut(_length,0.0),
     parent(_parent),
     tmpMean(_dim),
     tmpVariance(_dim)
  {}

  //constructor for the top
  parameters::parameters(vector<uint32_t> const& _stateNum,uint32_t _length,uint32_t _dim)
    :tmpTransContent(_stateNum[0],_stateNum[0]+1),
     parent(nullptr)
  {
    if(_stateNum.size() < 2){exit(1);}
    
    vector<uint32_t> nextNum(++begin(_stateNum),end(_stateNum));
    for(uint32_t t=0;t<_stateNum[0];++t){
      children.push_back(up<parameters>(new parameters(nextNum,_stateNum[0],_length,_dim,this)));
    }
    children.shrink_to_fit();
  }

  void parameters::transform()
  {
    delta = move(alpha);
    beta.clear();
    etaIn.clear();
    etaOut.clear();
    chi.clear();
    gammaIn.clear();
    gammaOut.clear();
    phi.resize(delta.size());
    tau.resize(delta.size());
    if(not children.empty()){
      for(auto& c:children){
        c->transform();
      }
    }
  }

  long double& parameters::xi(uint32_t x,baseHHMM* y,nprodHHMM* z)
  {
    return xiContent(x,z->convert[reinterpret_cast<uint64_t>(y)]);
  }

  long double& parameters::tmpTrans(baseHHMM* x,baseHHMM* y,nprodHHMM* z)
  {
    return tmpTransContent(z->convert[reinterpret_cast<uint64_t>(x)],   \
                           z->convert[reinterpret_cast<uint64_t>(y)]);
  }

  MatrixXld& parameters::tmpTrans()
  {
    return tmpTransContent;
  }
}

#endif
