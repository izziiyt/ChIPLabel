#include "nprodHHMM.h"

namespace hhmm{

  //constructor for root Node.
  nprodHHMM::nprodHHMM(uint32_t _depth,uint32_t _stateNum,uint32_t _dim)

    :baseHHMM(0,nullptr),
     transMat(_stateNum,_stateNum+1)
  {
    if(_depth < 2){exit(1);}
    else if(_depth == 2){
      for(uint32_t t=0;t<_stateNum;++t){
        children.push_back(up<baseHHMM>(new prodHHMM(1,_dim,this)));
      }
    }
    else{
      for(uint32_t t=0;t<_stateNum;++t){
        children.push_back(up<baseHHMM>(new nprodHHMM(_depth,_stateNum,_dim,1,this)));
      }
    }
    uint32_t i;
    for(i=0;i<children.size();++i){
      convert.insert(pair<uint64_t,uint32_t>\
                     (reinterpret_cast<uint64_t>(children[i].get()),i));
    }
    convert.insert(pair<uint64_t,uint32_t>(0,i));
    pi = 1.0;
  }

  //constructor for not-root Node
  nprodHHMM::nprodHHMM(uint32_t _depth,uint32_t _stateNum,              \
                       uint32_t _dim,uint32_t _level,baseHHMM* _parent)
    :baseHHMM(_level,_parent),
     transMat(_stateNum,_stateNum+1)
  {
    if(level < _depth-2){
      for(uint32_t t=0;t<_stateNum;++t){
        children.push_back(up<baseHHMM>                       \
                           (new nprodHHMM(_depth,_stateNum,_dim,_level+1,this)));
      }
    }
    else if(level == _depth-2){
      for(uint32_t t=0;t<_stateNum;++t){
        children.push_back(up<baseHHMM>(new prodHHMM(_level+1,_dim,this)));
      }
    }
    uint32_t i;
    for(i=0;i<children.size();++i){
      convert.insert(pair<uint64_t,uint32_t>                            \
                     (reinterpret_cast<uint64_t>(children[i].get()),i));
    }
    convert.insert(pair<uint64_t,uint32_t>(0,i));
  }
  
  double nprodHHMM::trans(baseHHMM* a)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a)],convert[0]);
  }

  double nprodHHMM::trans(up<baseHHMM> const& a)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],convert[0]);
  }
  
  double nprodHHMM::trans(const up<baseHHMM>& a,const up<baseHHMM>& b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],
                    convert[reinterpret_cast<uint64_t>(b.get())]);
  }

  double nprodHHMM::trans(baseHHMM* a,const up<baseHHMM>& b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a)],
                    convert[reinterpret_cast<uint64_t>(b.get())]);
  }

  double nprodHHMM::trans(const up<baseHHMM>& a,baseHHMM* b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],
                    convert[reinterpret_cast<uint64_t>(b)]);
  }

  void nprodHHMM::cpyTransMat(MatrixXd const& x)
  {
    transMat = x;
  }

}
