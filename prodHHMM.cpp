#include "prodHHMM.h"
#include <cmath>
#include <iostream>

namespace hhmm{

  prodHHMM::prodHHMM(uint32_t _level,uint32_t _dim,baseHHMM* _parent)
    :baseHHMM(_level,_parent),
     mean(_dim),
     //var(_dim,_dim)
     var(_dim)
  {}

  long double prodHHMM::emit(VectorXld const& O) const
  {
    long double mom = sqrt(pow(2 * M_PI,mean.size()) * var.diagonal().prod());
    long double son = exp(-0.5 * (O - mean).transpose() * var.inverse()*(O - mean));
    if(std::isnan(son/mom)){std::cout << "yes" << std::endl;
    cout << O << endl;
    cout << var.diagonal() << endl;
    cout << mean << endl;
    cout << mom << " " << son << endl;
    cout << "------------ " << endl;
    cout << (O - mean).transpose() << endl;
    cout << var.inverse().diagonal() << endl;
    cout << (O - mean) << endl; 
    exit(1);
    }
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

}
