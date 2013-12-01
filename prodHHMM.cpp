#include "prodHHMM.h"

namespace hhmm{

  prodHHMM::prodHHMM(uint32_t _level,uint32_t _dim,baseHHMM* _parent)
    :baseHHMM(_level,_parent),
     mean(_dim),
     var(_dim)
  {}

  double prodHHMM::emit(VectorXd const& O) const
  {
    double mom = sqrt(pow(2 * M_PI,mean.size()) * var.diagonal().prod());
    double son = exp(-0.5 * (O - mean).transpose() * var.inverse()*(O - mean));
    return son/mom;
  }

  VectorXd prodHHMM::getMean() const
  {
    return mean;
  }
  
  MatrixXd prodHHMM::getVar() const
  {
    return var;
  }

  void prodHHMM::cpyMean(VectorXd const& args)
  {
    mean = args;
  }
  
  void prodHHMM::cpyVar(DM const& args)
  {
    var = args;
  }

  void prodHHMM::swpMean(VectorXd& args)
  {
    mean.swap(args);
  }

  void prodHHMM::swpVar(DM& args)
  {
    var.diagonal().swap(args.diagonal());
  }

  void prodHHMM::setEmit(vector<double> const& x)
  {
    testemit = x;
  }
  
  double prodHHMM::emit(uint32_t i) const
  {
    return testemit[i];
  }

}
