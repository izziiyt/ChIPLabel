#include "prodHHMM.h"
#include <cmath>
#include <iostream>

namespace hhmm{

  prodHHMM::prodHHMM(uint32_t _level,uint32_t _dim,baseHHMM* _parent)
    :baseHHMM(_level,_parent),
     mean(_dim),
     var(_dim)
  {}

  long double prodHHMM::emit(VectorXld const& O) const
  {
    long double mom = sqrt(pow(2 * M_PI,mean.size()) * var.diagonal().prod());
    long double son = exp(-0.5 * (O - mean).transpose() * var.inverse() * (O - mean));
    //    if(isnan(son)){
    //      return 1;
    //    }
    //    if(isnan(son/mom)){
    //      cout << "emit problem" << endl;
    //      cout << mom << endl;
    //      cout << son << endl;
    //      cout << "----------------------------" << endl;
    //      cout << mean << endl;
    //      cout << "----------------------------" << endl;
    //      cout << var.diagonal() << endl;
    //      cout << "----------------------------" << endl;
    //      cout << O << endl;
    //      exit(1);
    //    }
    return son/mom;
  }

  
  VectorXld const& prodHHMM::getMean() const
  {
    return mean;
  }
  
  VectorXld& prodHHMM::setMean()
  {
    return mean;
  }

  DM const& prodHHMM::getVariance() const
  {
    return var;
  }

  DM& prodHHMM::setVariance()
  {
    return var;
  }

  void prodHHMM::cpyMean(VectorXld const& args)
  {
    mean = args;
  }
  
  void prodHHMM::cpyVar(DM const& args)
  {
    var = args;
  }

  void prodHHMM::swpMean(VectorXld& args)
  {
    mean.swap(args);
  }

  void prodHHMM::swpVar(DM& args)
  {
    var.diagonal().swap(args.diagonal());
  }

  // void prodHHMM::setEmit(vector<long double> const& x)
  // {
  //   testemit = x;
  // }
  
  // long double prodHHMM::emit(uint32_t i) const
  // {
  //   return testemit[i];
  // }

  void prodHHMM::clearParam()
  {
    pi = 0.0;
    emitParent = 0.0;
    mean.setZero();
    //for(uint32_t i=0;i<testemit.size();++i){testemit[i] = 0.0;}
    var.setZero();
  }

  void prodHHMM::check()
  {
    cout << "mean" << endl;
    cout << mean.transpose() << endl;
    cout << "variance" << endl;
    cout << var.diagonal().transpose() << endl;
  }

  void prodHHMM::initParam(vector<long double> const& xs)
  {
    mt19937 gen((uint64_t)this);
    for(uint32_t i=0;i<xs.size();++i){
      normal_distribution<long double> nd(xs[i],1.0);
      mean(i) = nd(gen);
    }
    for(uint32_t i=0;i<var.diagonal().size();++i){
      var.diagonal()(i) = 1.0;
    }
  }

  void prodHHMM::log(uint32_t loop,uint32_t ID,string const& toDir)
  {
    ofstream ofs(toDir + to_string(ID),ios::out | ios::app);
    ofs << "loop " << loop << endl;
    ofs << "emitParent " << emitParent << endl;
    ofs << "mean " << mean.transpose() << endl;
    ofs << "variance " << var.diagonal().transpose() << endl;
    ofs.close();
  }
}
