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
  
  MatrixXld& nprodHHMM::trans()
  {
    return transMat;
  }
  
  long double nprodHHMM::trans(baseHHMM* a)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a)],convert[0]);
  }

  long double nprodHHMM::trans(up<baseHHMM> const& a)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],convert[0]);
  }
  
  long double nprodHHMM::trans(const up<baseHHMM>& a,const up<baseHHMM>& b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],
                    convert[reinterpret_cast<uint64_t>(b.get())]);
  }

  long double nprodHHMM::trans(baseHHMM* a,const up<baseHHMM>& b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a)],
                    convert[reinterpret_cast<uint64_t>(b.get())]);
  }

  long double nprodHHMM::trans(const up<baseHHMM>& a,baseHHMM* b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a.get())],
                    convert[reinterpret_cast<uint64_t>(b)]);
  }

  void nprodHHMM::cpyTransMat(MatrixXld const& x)
  {
    transMat = x;
  }

  void nprodHHMM::clearParam()
  {
    pi = 0.0;
    transMat.setZero();
    for(auto& c:children){c->clearParam();}
  }

  void nprodHHMM::initParam(vector<long double> const& xs)
  {
    mt19937 gen(children.size());
    //initiation of the vertical transition probability.
    vector<uint32_t> alpha(children.size(),1);
    ytd::dirichlet_distribution<long double> dd0(alpha);
    auto x = dd0(gen);
    for(uint32_t i=0;i<children.size();++i){
      children[i]->setPi() = x[i];
    }
    //initiation of the horizontal transition probability.
    alpha.push_back(1);
    ytd::dirichlet_distribution<long double> dd1(alpha);
    for(uint32_t i=0;i<transMat.rows();++i){
      auto y = dd1(gen);
      transMat.row(i) = Map<VectorXld>(&y[0],y.size());
    }
    for(auto& c:children){c->initParam(xs);}
  }
  
  long double& nprodHHMM::setTrans(baseHHMM* a,const up<baseHHMM>& b)
  {
    return transMat(convert[reinterpret_cast<uint64_t>(a)],
                    convert[reinterpret_cast<uint64_t>(b.get())]);
  }

  void nprodHHMM::check()
  {
    for(uint32_t i=0;i<transMat.rows();++i){
      cout << "Is this 1 ? : " << transMat.row(i).sum() << endl;
    }
    long double tmp = 0.0;
    for(auto& c:children){tmp += c->getPi();}
    cout << "Is this 1 ? : " << tmp << endl;
    
    for(auto& c:children){c->check();}
  }
}


















