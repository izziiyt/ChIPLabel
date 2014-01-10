#include "HHMM.h"
#include <iostream>

namespace hhmm{

  HHMM::HHMM(uint32_t _dim,uint32_t _stateNum,uint32_t _depth)
    :root(_depth,_stateNum,_dim),
     dim(_dim),
     stateNum(_stateNum),
     depth(_depth){}

  void HHMM::forward(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Clear the alpha_values.
    if(root->getLevel() != 0){param->alpha.clear();}

    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<parameters> pit,pend,bpit,bpend,cpit,cpend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,forward(children).
    setIterator<parameters>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      forward(seq,rit->get(),pit->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          (*pit)->alpha(i,i) = (*rit)->getPi() * cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<parameters>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            setIterator<parameters>(bpit,bpend,brit,brend,param,root);
            for(;bpit != bpend && brit != brend;++bpit,++brit){
              (*pit)->alpha(i,j) += (*bpit)->alpha(i,j-1) * \
                cast_nprod(root)->trans(*brit,*rit);
            }
            (*pit)->alpha(i,j) *= cast_prod(rit->get())->emit(seq.obs(j));
          }
        }
      }
    }
    //In the level that is neither the deepest nor the second deepest.  
    else{
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
          for(;cpit != cpend && crit != crend;++cpit,++crit){
            (*pit)->alpha(i,i) += (*cpit)->alpha(i,i) * \
              cast_nprod(rit->get())->trans(*crit);
          }
          (*pit)->alpha(i,i) *= (*rit)->getPi();


        }
        long double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<parameters>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<parameters>(bpit,bpend,brit,brend,param,root);
              for(tmp0 = 0.0;bpit != bpend && brit != brend;++bpit,++brit){
                tmp0 += (*bpit)->alpha(i,k) * cast_nprod(root)->trans(*brit,*rit);
              }
              setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
              for(tmp1 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
                tmp1 += (*cpit)->alpha(k+1,j) * cast_nprod(rit->get())->trans(*crit);
              }
              (*pit)->alpha(i,j) += tmp0 * tmp1;
            }
            setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
            for(tmp2 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
              tmp2 += (*cpit)->alpha(i,j) * cast_nprod(rit->get())->trans(*crit);
            }
            (*pit)->alpha(i,j) += tmp2 * (*rit)->getPi();
          }
        }
      }
    }
  }

  void HHMM::backward(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Clear the beta_values.
    if(root->getLevel() != 0){param->beta.clear();}

    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<parameters> pit,pend,bpit,bpend,cpit,cpend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,backward(children).
    setIterator<parameters>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      backward(seq,rit->get(),pit->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          (*pit)->beta(i,i) = cast_nprod(root)->trans(*rit) * \
            cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<parameters>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            setIterator<parameters>(bpit,bpend,brit,brend,param,root);
            for(;bpit != bpend && brit != brend;++bpit,++brit){
              (*pit)->beta(i,j) += (*bpit)->beta(i+1,j) * \
                cast_nprod(root)->trans(*rit,*brit);
            }
            (*pit)->beta(i,j) *= cast_prod(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
          for(;cpit != cpend && crit != crend;++cpit,++crit){
            (*pit)->beta(i,i) += (*cpit)->beta(i,i) * (*crit)->getPi();
          }
          (*pit)->beta(i,i) *= cast_nprod(root)->trans(*rit);
        }
        long double tmp0,tmp1,tmp2;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<parameters>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<parameters>(bpit,bpend,brit,brend,param,root);
              for(tmp0 = 0.0;bpit != bpend && brit != brend;++bpit,++brit){
                tmp0 += (*bpit)->beta(k+1,j) * cast_nprod(root)->trans(*rit,*brit);
              }
              setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
              for(tmp1 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
                tmp1 += (*cpit)->beta(i,k) * (*crit)->getPi();
              }
              (*pit)->beta(i,j) += tmp0 * tmp1;
            }
            setIterator<parameters>(cpit,cpend,crit,crend,pit->get(),rit->get());
            for(tmp2 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
              tmp2 += (*cpit)->beta(i,j) * (*crit)->getPi();
            }
            (*pit)->beta(i,j) += tmp2 * cast_nprod(root)->trans(*rit);
          }
        }
      }
    }
  }

  void HHMM::auxIn(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;
    //In the second top level.
    if(root->getLevel() == 1){
      //Clear the etaIn_values.
      for(uint32_t i=0;i<seq.size();++i){param->etaIn[i] = 0.0;}
      //i=0
      param->etaIn[0] = root->getPi();
      for(int64_t i=1;i<seq.size();++i){
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          param->etaIn[i] += (*pit)->alpha(0,i-1) *     \
            cast_nprod(root->parent)->trans(*rit,root);
        }
      }
    }
    //In not the second top level.
    else if(root->getLevel() > 1){
      //Clear the etaIn_values.
      for(uint32_t i=0;i<seq.size();++i){param->etaIn[i] = 0.0;}
      //i=0
      param->etaIn[0] = param->parent->etaIn[0] * root->getPi();
      for(int64_t i=1;i<seq.size();++i){
        for(uint32_t j=0;j<i;++j){
          long double tmp;
          setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
          for(tmp = 0.0;pit != pend && rit != rend;++pit,++rit){
            tmp += (*pit)->alpha(j,i-1) * cast_nprod(root->parent)->trans(*rit,root);
          }
          param->etaIn[i] += tmp * param->parent->etaIn[j];
        }
        param->etaIn[i] += param->parent->etaIn[i] * root->getPi(); 
      }
    }
    if(root->getLevel() < depth-1){
      //At last,etaIn(children).
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        auxIn(seq,rit->get(),pit->get());
      }
    }
  }

  void HHMM::auxOut(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;
    //In the second top level.
    if(root->getLevel() == 1){
      //Clear the etaOut_values.
      for(uint32_t i=0;i<seq.size();++i){param->etaOut[i] = 0.0;}
      for(int64_t i=0;i<seq.size()-1;++i){
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          param->etaOut[i] += (*pit)->beta(i+1,seq.size()-1) *
            cast_nprod(root->parent)->trans(root,*rit);
        }
      }
      param->etaOut[seq.size()-1] = cast_nprod(root->parent)->trans(root);
    }
    //In the level that is not the deepest nor the second deepest.  
    else if(root->getLevel() > 1){
      //Clear the etaOut_values.
      for(uint32_t i=0;i<seq.size();++i){param->etaOut[i] = 0.0;}
      for(int64_t i=0;i<seq.size()-1;++i){
        for(uint32_t j=i+1;j<seq.size();++j){
          long double tmp;
          setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
          for(tmp = 0.0;pit != pend && rit != rend;++pit,++rit){
            tmp += (*pit)->beta(i+1,j) * cast_nprod(root->parent)->trans(root,*rit);
          }
          param->etaOut[i] += tmp * param->parent->etaOut[j];
        }
        param->etaOut[i] += param->parent->etaOut[i] * \
          cast_nprod(root->parent)->trans(root); 
      }
      param->etaOut[seq.size()-1] = param->parent->etaOut[seq.size()-1] * \
        cast_nprod(root->parent)->trans(root);
    }
    if(root->getLevel() < depth-1){
      //At last,etaOut(children).
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        auxOut(seq,rit->get(),pit->get());
      }
    }
  }

  void HHMM::horizon(Sequence& seq,baseHHMM* root,parameters* param)
  {
    using namespace std::placeholders;
    //xi is param->xi(_1,_2,cast_nprod(root->parent))
    auto xi = bind(&parameters::xi,param,_1,_2,cast_nprod(root->parent));
    auto endXi = bind(&parameters::xi,param,_1,nullptr,cast_nprod(root->parent));

    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;

    // In the second top level.
    if(root->getLevel() == 1){
      //Clear the xi_values.
      for(uint32_t i=0;i<seq.size()-1;++i){
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          xi(i,rit->get()) = param->alpha(0,i) * (*pit)->beta(i+1,seq.size()-1) * \
            cast_nprod(root->parent)->trans(root,*rit) / likelihood(seq);
        }
        endXi(i) = 0.0;
      }
      // i == seq.size()-1
      setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
      for(;pit != pend && rit != rend;++pit,++rit){
        xi(seq.size()-1,(*rit).get()) = 0.0;
      }
      endXi(seq.size()-1) = param->alpha(0,seq.size()-1) *        \
        cast_nprod(root->parent)->trans(root) / likelihood(seq);
    }
    //In the level that is not the top nor the second top.
    else if(root->getLevel() > 1){
      //Clear the xi_values.
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){xi(i,(*rit).get()) = 0.0;}
        endXi(i) = 0.0;
      }
      for(uint32_t i=0;i<seq.size();++i){
        long double tmp0 = 0.0;
        for(uint32_t j=0;j<=i;++j){
          tmp0 += param->parent->etaIn[j] * param->alpha(j,i);
        }
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          if(i < seq.size()-1){
            long double tmp1 = 0.0;
            for(uint32_t k=i+1;k<seq.size();++k){
              tmp1 += param->parent->etaOut[k] * (*pit)->beta(i+1,k);
            }
            xi(i,(*rit).get()) = tmp0 * tmp1 *                          \
              cast_nprod(root->parent)->trans(root,*rit)  / likelihood(seq);
          }
          else{
            xi(i,(*rit).get()) = 0.0;
          }
        }
        endXi(i) = tmp0 * cast_nprod(root->parent)->trans(root) *  \
          param->parent->etaOut[i] / likelihood(seq);
      }
    }
    if(root->getLevel() < depth-1){
      //At last,horizon(children).
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        horizon(seq,rit->get(),pit->get());
      }
    }
  }
  
  void HHMM::vertical(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //In the second top level.
    if(root->getLevel() == 1){
      param->chi[0] = root->getPi() * param->beta(0,seq.size()-1) / likelihood(seq);
    }
    //In neither the top nor the second top level.
    else if(root->getLevel() > 1){
      for(uint32_t i=0;i<seq.size();++i){
        param->chi[i] = 0.0;
        for(uint32_t j=i;j<seq.size();++j){
          param->chi[i] += param->beta(i,j) * param->parent->etaOut[j];
        }
        param->chi[i] *= param->parent->etaIn[i] * root->getPi() / likelihood(seq);
      }
    }
    //In not the deepest level.
    if(root->getLevel() < depth-1){
      //Declear iterators.
      myit<parameters> pit,pend;
      myit<baseHHMM> rit,rend;
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        vertical(seq,rit->get(),pit->get());
      }
    }
  }
  
  //GammaOut may be not used.
  void HHMM::calcGamma(Sequence& seq,baseHHMM* root,parameters* param)
  {
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;

    //In not the top level.
    if(root->getLevel() != 0){
      //i == 0
      param->gammaIn[0] = 0.0;
      // setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
      // for(;pit != pend && rit != rend;++pit,++rit){
      //   param->gammaOut[0] += param->xi(0,rit->get(),cast_nprod(root->parent));
      // }
      // param->gammaOut[0] += param->xi(0,nullptr,cast_nprod(root->parent));
      //i == 1 to T-1
      for(uint32_t i=1;i<seq.size();++i){
        param->gammaIn[i] = 0.0;
        //param->gammaOut[i] = 0.0;        
        setIterator<parameters>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          param->gammaIn[i] += (*pit)->xi(i-1,root,cast_nprod(root->parent));
          //param->gammaOut[i] += param->xi(i,rit->get(),cast_nprod(root->parent));
        }
        //param->gammaOut[i] += param->xi(i,nullptr,cast_nprod(root->parent));
      }
    }
    //In not the deepest level.
    if(root->getLevel() < depth-1){
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        calcGamma(seq,rit->get(),pit->get());
      }
    }
  }

  void HHMM::calcTmpTrans(Sequence& seq,baseHHMM* root,parameters* param)
  {
    myit<parameters> pit,pend,bpit,bpend;
    myit<baseHHMM> rit,rend,brit,brend;

    //In not the deepest level.
    if(root->getLevel() < depth-1){
      param->tmpTrans().setZero();
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          setIterator<parameters>(bpit,bpend,brit,brend,param,root);
          for(;bpit != bpend && brit != brend;++bpit,++brit){
            param->tmpTrans(rit->get(),brit->get(),cast_nprod(root)) += \
              (*pit)->xi(i,brit->get(),cast_nprod(root));
          }
          param->tmpTrans(rit->get(),nullptr,cast_nprod(root)) += \
            (*pit)->xi(i,nullptr,cast_nprod(root));
        }
      }
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        calcTmpTrans(seq,rit->get(),pit->get());
      }
    }
  }

  // void HHMM::calcTmpEmit(Sequence& seq,baseHHMM* root,parameters* param)
  // {
  //   myit<parameters> pit,pend,bpit,bpend;
  //   myit<baseHHMM> rit,rend,brit,brend;

  //   //In not the deepest level.
  //   if(root->getLevel() < depth-1){
  //     setIterator<parameters>(pit,pend,rit,rend,param,root);
  //     for(;pit != pend && rit != rend;++pit,++rit){
  //       calcTmpEmit(seq,rit->get(),pit->get());
  //     }
  //   }
  //   //In the deepest level.
  //   if(root->getLevel() == depth-1){
  //     for(auto& tE:param->tmpEmit){tE = 0.0;}
  //     param->tmpEmit[seq.obs(0)] += param->chi[0];
  //     for(uint32_t i=0;i<seq.size();++i){
  //       param->tmpEmit[seq.obs(i)] += param->chi[i] + param->gammaIn[i];
  //     }
  //   }
  // }

  //not debagged
  void HHMM::calcTmpMean(Sequence& seq,baseHHMM* root,parameters* param)
  {
    myit<parameters> pit,pend,bpit,bpend;
    myit<baseHHMM> rit,rend,brit,brend;
    //In the deepest level
    if(root->getLevel() == depth-1){
      //when t == 0
      param->tmpMean = seq.obs(0) * param->chi[0];

      param->tmpEmitParent = param->chi[0];
      //when t == 0 to T-1
      for(uint32_t i=1;i<seq.size();++i){
        param->tmpMean += (param->chi[i] + param->gammaIn[i]) * seq.obs(i);
        param->tmpEmitParent += param->chi[i] + param->gammaIn[i];
      }
      //cout << param->tmpMean << endl;
    }
    else{
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        calcTmpMean(seq,rit->get(),pit->get());
      }
    }
  }

  void HHMM::calcTmpVariance(Sequence& seq,baseHHMM* root,parameters* param)
  {
    myit<parameters> pit,pend,bpit,bpend;
    myit<baseHHMM> rit,rend,brit,brend;
    if(root->getLevel() == depth-1){
      //when t == 0
      VectorXld tmp = seq.obs(0) - cast_prod(root)->getMean();
      param->tmpVariance.diagonal() = tmp.asDiagonal() * tmp * param->chi[0];
      //when t == 0 to T-1
      for(uint32_t i=1;i<seq.size();++i){
        tmp = seq.obs(i) - cast_prod(root)->getMean();
        param->tmpVariance.diagonal() += tmp.asDiagonal() * tmp * \
          (param->chi[i] + param->gammaIn[i]);
      }
    }
    else{
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        calcTmpVariance(seq,rit->get(),pit->get());
      }
    }
  }

  void HHMM::calcTmpPi(Sequence& seq,baseHHMM* root,parameters* param)
  {
    if(root->getLevel() > 0){
      //In the second top level.
      if(root->getLevel() == 1){
        param->tmpPi = param->chi[0];
      }
      //In neither the top nor the second top level.
      else{
        param->tmpPi = 0.0;
        for(uint32_t i=0;i<seq.size();++i){
          param->tmpPi += param->chi[i];
        }
      }
    }
    //In not the deepest level.
    if(root->getLevel() < depth-1){
      myit<parameters> pit,pend;
      myit<baseHHMM> rit,rend;
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        calcTmpPi(seq,rit->get(),pit->get());
      }
    }
  }
 
  void HHMM::clearParam()
  {
    root.clearParam();
  }
  
  //assemble sequences' tmpValues, excluding the variance variables
  void HHMM::varianceAssemble(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;

    //In not the deepest level.
    if(root->getLevel() != depth-1){
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        varianceAssemble(seq,rit->get(),pit->get());
      }
    }
    //In the deepest level.
    else{
      cast_prod(root)->setVariance().diagonal() += param->tmpVariance.diagonal();
    }
  }


  void HHMM::paramAssemble(Sequence& seq,baseHHMM* root,parameters* param)
  {
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;

    //In not the deepest level.
    if(root->getLevel() != depth-1){
      cast_nprod(root)->trans() += param->tmpTrans();
      setIterator<parameters>(pit,pend,rit,rend,param,root);
      for(;pit != pend && rit != rend;++pit,++rit){
        paramAssemble(seq,rit->get(),pit->get());
      }
    }
    //In not the top level.
    if(root->getLevel() > 0){
      root->setPi() += param->tmpPi;
      //In the deepest level.
      if(root->getLevel() == depth-1){
        cast_prod(root)->setMean() += param->tmpMean;
        cast_prod(root)->emitParent += param->tmpEmitParent;
        // for(uint32_t i=0;i<param->tmpEmit.size();++i){
        //   cast_prod(root)->setEmit()[i] += param->tmpEmit[i];
        // }
      }
    }
  }

  //Excluding variance variables
  void HHMM::paramStandardize(baseHHMM* root)
  {
    long double tmp = 0.0;
    //In not the deepest level.
    if(root->getLevel() < depth-1){
      myit<baseHHMM> b = begin(cast_nprod(root)->children);
      myit<baseHHMM> e = end(cast_nprod(root)->children);

      for(auto it = b;it != e;++it){
        tmp += (*it)->getPi();
      }
      for(auto it = b;it != e;++it){
        (*it)->setPi() /= tmp;
      }
      for(uint32_t i=0;i<cast_nprod(root)->trans().rows();++i){
        tmp = 0.0;
        for(uint32_t j=0;j<cast_nprod(root)->trans().cols();++j){
          tmp += cast_nprod(root)->trans()(i,j);
        }
        for(uint32_t j=0;j<cast_nprod(root)->trans().cols();++j){
          cast_nprod(root)->trans()(i,j) /= tmp;
        }
      }
      for(auto it = b;it != e;++it){
        paramStandardize(it->get());
      }
    }
    //In the deepest level.
    if(root->getLevel() == depth-1){
      cast_prod(root)->setMean() /= cast_prod(root)->emitParent;
    //   tmp = accumulate(begin(cast_prod(root)->setEmit()),     \
    //                    end(cast_prod(root)->setEmit()),0.0);
    //   for(auto& e:cast_prod(root)->setEmit()){e /= tmp;}
    }
  }

  void HHMM::varianceStandardize(baseHHMM* root)
  {
    //In not the deepest level.
    if(root->getLevel() < depth-1){
      myit<baseHHMM> b = begin(cast_nprod(root)->children);
      myit<baseHHMM> e = end(cast_nprod(root)->children);
      for(auto it = b;it != e;++it){
        varianceStandardize(it->get());
      }
    }
    //In the deepest level.
    else{
      cast_prod(root)->setVariance().diagonal() /= cast_prod(root)->emitParent;
    }
  }

  template<typename T>
  void HHMM::setIterator(myit<T>& b0,myit<T>& e0,myit<baseHHMM>& b1,myit<baseHHMM>& e1,T* x,baseHHMM* y)
  {
    b0 = begin(x->children);
    e0 = end(x->children);
    b1 = begin(cast_nprod(y)->children);
    e1 = end(cast_nprod(y)->children);
  }

  long double HHMM::likelihood(Sequence& seq)
  {
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;
    long double result = 0.0;
    setIterator<parameters>(pit,pend,rit,rend,&(seq.param),&root);
    for(;pit != pend && rit != rend;++pit,++rit){
      result += (*pit)->alpha(0,seq.size()-1) * cast_nprod(&root)->trans(*rit);
    }
    return result;
  }

  void HHMM::forward(Sequence& s){
    forward(s,&root,&(s.param));
  }
  void HHMM::backward(Sequence& s){
    backward(s,&root,&(s.param));
  }
  void HHMM::auxIn(Sequence& s){
    auxIn(s,&root,&(s.param));
  }
  void HHMM::auxOut(Sequence& s){
    auxOut(s,&root,&(s.param));
  }
  void HHMM::horizon(Sequence& s){
    horizon(s,&root,&(s.param));
  }
  void HHMM::vertical(Sequence& s){
    vertical(s,&root,&(s.param));
  }
  void HHMM::calcGamma(Sequence& s){
    calcGamma(s,&root,&(s.param));
  }
  void HHMM::calcTmpPi(Sequence& s){
    calcTmpPi(s,&root,&(s.param));
  }
  void HHMM::calcTmpTrans(Sequence& s){
    calcTmpTrans(s,&root,&(s.param));
  }
  void HHMM::calcTmpMean(Sequence& s){
    calcTmpMean(s,&root,&(s.param));
  }
  void HHMM::calcTmpVariance(Sequence& s){
    calcTmpVariance(s,&root,&(s.param));
  }
  // void HHMM::calcTmpEmit(Sequence& s){
  //   calcTmpEmit(s,&root,&(s.param));
  // }
  void HHMM::paramAssemble(Sequence & s){
    paramAssemble(s,&root,&(s.param));
  }
  void HHMM::paramStandardize(){
    paramStandardize(&root);
  }
  void HHMM::varianceAssemble(Sequence & s){
    varianceAssemble(s,&root,&(s.param));
  }
  void HHMM::varianceStandardize(){
    varianceStandardize(&root);
  }
  void HHMM::EM()
  {
    for(uint32_t i=0;i<1;++i){

      //E-step by multi-threading
      for(auto& s:seq){
        forward(*s);
        backward(*s);
        auxIn(*s);
        auxOut(*s);
        horizon(*s);
        vertical(*s);
        calcGamma(*s);
        calcTmpPi(*s);
        calcTmpTrans(*s);
        //calcTmpEmit(*s);
        calcTmpMean(*s);
      }
      cout << "likelihood: " << likelihood(*seq[0]) << endl;
      //M-step by single-threading
      clearParam();
      for(auto& s:seq){paramAssemble(*s);}
      paramStandardize();

      
      //E-step for the variance variables by multi-threading
      for(auto& s:seq){
        calcTmpVariance(*s);
        varianceAssemble(*s);
      }

      //M-step for the variance variables by single-threading
      varianceStandardize();
      // cout << cast_prod(cast_nprod(root.children[0].get())->children[0].get())->getMean() << endl;
      // cout << cast_prod(cast_nprod(root.children[0].get())->children[1].get())->getMean() << endl;
      // cout << cast_prod(cast_nprod(root.children[1].get())->children[0].get())->getMean() << endl;
      // cout << cast_prod(cast_nprod(root.children[1].get())->children[1].get())->getMean() << endl;
      // cout << "------------------" << endl;
      // // cout << root.trans() << endl;
      // // cout << cast_nprod(root.children[0].get())->trans() << endl;
      // // cout << cast_nprod(root.children[1].get())->trans() << endl;
      // // cout << root.children[0]->getPi() << endl;
      // // cout << root.children[1]->getPi() << endl;
      // // cout << cast_nprod(root.children[0].get())->children[0]->getPi() << endl;
      // // cout << cast_nprod(root.children[0].get())->children[1]->getPi() << endl;
      // // cout << cast_nprod(root.children[1].get())->children[0]->getPi() << endl;
      // // cout << cast_nprod(root.children[1].get())->children[1]->getPi() << endl;

      // cout << cast_prod(cast_nprod(root.children[0].get())->children[0].get())->getVariance().diagonal() << endl;
      // cout << cast_prod(cast_nprod(root.children[0].get())->children[1].get())->getVariance().diagonal() << endl;
      // cout << cast_prod(cast_nprod(root.children[1].get())->children[0].get())->getVariance().diagonal() << endl;
      // cout << cast_prod(cast_nprod(root.children[1].get())->children[1].get())->getVariance().diagonal() << endl;

    }
  }

}













