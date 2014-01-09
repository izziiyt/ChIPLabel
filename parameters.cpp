#ifndef PARAMETERS_CPP
#define PARAMETERS_CPP

#include "parameters.h"

namespace hhmm{

  //constructor for neither the top nor the deepest
  parameters::parameters(uint32_t _depth,uint32_t _childNum,uint32_t _length,uint32_t _dim,parameters* _parent)
    :xiContent(_length,_childNum+1),
     tmpTransContent(_childNum,_childNum+1),
     alpha(_length,0.0),
     beta(_length,0.0),
     etaIn(_length,0.0),
     etaOut(_length,0.0),
     chi(_length,0.0),
     gammaIn(_length,0.0),
     gammaOut(_length,0.0),
     parent(_parent)
  {
    if(_depth > 2){
      for(uint32_t t=0;t<_childNum;++t){
        children.push_back(up<parameters>
                           (new parameters(_depth-1,_childNum,_length,_dim,this)));
      }
      children.shrink_to_fit();
    }
    else if(_depth == 2){
      for(uint32_t t=0;t<_childNum;++t){
        children.push_back(up<parameters>(new parameters(_length,_childNum,_dim,this)));
      }
      children.shrink_to_fit();
    }
  }
  
  //constructor for the deepest
  parameters::parameters(uint32_t _length,uint32_t _childNum,uint32_t _dim,parameters* _parent)
    :xiContent(_length,_childNum+1),    
     alpha(_length,0.0),
     beta(_length,0.0),
     etaIn(_length,0.0),
     etaOut(_length,0.0),
     chi(_length,0.0),
     gammaIn(_length,0.0),
     gammaOut(_length,0.0),
     parent(_parent),
     //tmpEmit(_dim)
     tmpMean(_dim),
     tmpVariance(_dim)
  {}

  //constructor for the top
  parameters::parameters(uint32_t _depth,uint32_t _childNum,uint32_t _length,uint32_t _dim)
    :xiContent(_length,_childNum+1),
     tmpTransContent(_childNum,_childNum+1),
     parent(nullptr)
  {
    if(_depth < 2){
      cerr << "Depth is not enough." << endl;
      exit(1);
    }
    for(uint32_t t=0;t<_childNum;++t){
      children.push_back(up<parameters>
                         (new parameters(_depth-1,_childNum,_length,_dim,this)));
    }
    children.shrink_to_fit();
  }

  double& parameters::xi(uint32_t x,baseHHMM* y,nprodHHMM* z)
  {
    return xiContent(x,z->convert[reinterpret_cast<uint64_t>(y)]);
  }
  double& parameters::tmpTrans(baseHHMM* x,baseHHMM* y,nprodHHMM* z)
  {
    return tmpTransContent(z->convert[reinterpret_cast<uint64_t>(x)],   \
                           z->convert[reinterpret_cast<uint64_t>(y)]);
  }
  MatrixXd& parameters::tmpTrans()
  {
    return tmpTransContent;
  }
}

#endif
