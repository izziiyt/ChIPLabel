#include "HHMM.h"
#include <iostream>

namespace hhmm{

  HHMM::HHMM(uint32_t _dim,uint32_t _stateNum,uint32_t _depth)
    :root(_depth,_stateNum,_dim),
     dim(_dim),
     stateNum(_stateNum),
     depth(_depth){}

  void HHMM::forward(Sequence& seq,baseHHMM* root,tree<parameters>* param)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto cast_prod = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    using mytree = tree<parameters>;

    //Clear the alpha_values.
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        param->set().alpha(i,j) = 0.0;
      }
    }
    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<mytree> pit,pend,bpit,bpend,cpit,cpend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,forward(children).
    setIterator<mytree>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      forward(seq,rit->get(),pit->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<mytree>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          (*pit)->set().alpha(i,i) = (*rit)->getPi() * cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            setIterator<mytree>(bpit,bpend,brit,brend,param,root);
            for(;bpit != bpend && brit != brend;++bpit,++brit){
              (*pit)->set().alpha(i,j) += (*bpit)->get().alpha(i,j-1) * \
                cast_nprod(root)->trans(*brit,*rit);
            }
            (*pit)->set().alpha(i,j) *= cast_prod(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<mytree>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
          for(;cpit != cpend && crit != crend;++cpit,++crit){
            (*pit)->set().alpha(i,i) += (*cpit)->get().alpha(i,i) * \
              cast_nprod(rit->get())->trans(*crit);
          }
          (*pit)->set().alpha(i,i) *= (*rit)->getPi();
        }
        double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<mytree>(bpit,bpend,brit,brend,param,root);
              for(tmp0 = 0.0;bpit != bpend && brit != brend;++bpit,++brit){
                tmp0 += (*bpit)->get().alpha(i,k) * cast_nprod(root)->trans(*brit,*rit);
              }
              setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
              for(tmp1 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
                tmp1 += (*cpit)->get().alpha(k+1,j) * cast_nprod(rit->get())->trans(*crit);
              }
              (*pit)->set().alpha(i,j) += tmp0 * tmp1;
            }
            setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
            for(tmp2 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
              tmp2 += (*cpit)->get().alpha(i,j) * cast_nprod(rit->get())->trans(*crit);
            }
            (*pit)->set().alpha(i,j) += tmp2 * (*rit)->getPi();
          }
        }
      }
    }

  }

  void HHMM::backward(Sequence& seq,baseHHMM* root,tree<parameters>* param)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto cast_prod = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    using mytree = tree<parameters>;

    //Clear the beta_values.
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        param->set().beta(i,j) = 0.0;
      }
    }
    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<mytree> pit,pend,bpit,bpend,cpit,cpend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,backward(children).
    setIterator<mytree>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
     backward(seq,rit->get(),pit->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<mytree>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          (*pit)->set().beta(i,i) = cast_nprod(root)->trans(*rit) * \
            cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            setIterator<mytree>(bpit,bpend,brit,brend,param,root);
            for(;bpit != bpend && brit != brend;++bpit,++brit){
              (*pit)->set().beta(i,j) += (*bpit)->get().beta(i+1,j) * \
                cast_nprod(root)->trans(*rit,*brit);
            }
            (*pit)->set().beta(i,j) *= cast_prod(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<mytree>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
          for(;cpit != cpend && crit != crend;++cpit,++crit){
            (*pit)->set().beta(i,i) += (*cpit)->get().beta(i,i) * (*crit)->getPi();
          }
          (*pit)->set().beta(i,i) *= cast_nprod(root)->trans(*rit);
        }
        double tmp0,tmp1,tmp2;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(pit,pend,rit,rend,param,root);
          for(;pit != pend && rit != rend;++pit,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<mytree>(bpit,bpend,brit,brend,param,root);
              for(tmp0 = 0.0;bpit != bpend && brit != brend;++bpit,++brit){
                tmp0 += (*bpit)->get().beta(k+1,j) * cast_nprod(root)->trans(*rit,*brit);
              }
              setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
              for(tmp1 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
                tmp1 += (*cpit)->get().beta(i,k) * (*crit)->getPi();
              }
              (*pit)->set().beta(i,j) += tmp0 * tmp1;
            }
            setIterator<mytree>(cpit,cpend,crit,crend,pit->get(),rit->get());
            for(tmp2 = 0.0;cpit != cpend && crit != crend;++cpit,++crit){
              tmp2 += (*cpit)->get().beta(i,j) * (*crit)->getPi();
            }
            (*pit)->set().beta(i,j) += tmp2 * cast_nprod(root)->trans(*rit);
          }
        }
      }
    }
  }

  void HHMM::auxIn(Sequence& seq,baseHHMM* root,tree<parameters>* param)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    using mytree = tree<parameters>;

    //Clear the etaIn_values.
    for(uint32_t i=0;i<seq.size();++i){param->set().etaIn[i] = 0.0;}
    //Declear iterators.
    myit<mytree> pit,pend;
    myit<baseHHMM> rit,rend;
    //In the second top level.
    if(root->getLevel() == 1){
      param->set().etaIn[0] = root->getPi();
      for(int64_t i=1;i>seq.size();++i){
        setIterator<mytree>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          param->set().etaIn[i] += (*pit)->get().alpha(0,i-1) * \
            cast_nprod(root->parent)->trans(*rit,root);
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      param->set().etaIn[0] = param->parent->get().etaIn[0] * root->getPi();
      for(int64_t i=1;i>seq.size();++i){
        for(uint32_t j=0;j<i;++j){
          double tmp;
          setIterator<mytree>(pit,pend,rit,rend,param->parent,root->parent);
          for(tmp = 0.0;pit != pend && rit != rend;++pit,++rit){
            tmp += (*pit)->get().alpha(j,i-1) *                     \
              cast_nprod(root->parent)->trans(*rit,root);
          }
          param->set().etaIn[i] += tmp * param->parent->get().etaIn[j];
        }
        param->set().etaIn[i] += param->parent->get().etaIn[i] * root->getPi(); 
      }
    }
    //At last,etaIn(children).
    setIterator<mytree>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      auxIn(seq,rit->get(),pit->get());
    }
  }

  void HHMM::auxOut(Sequence& seq,baseHHMM* root,tree<parameters>* param)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    using mytree = tree<parameters>;

    //Clear the etaOut_values.
    for(uint32_t i=0;i<seq.size();++i){param->set().etaOut[i] = 0.0;}
    //Declear iterators.
    myit<mytree> pit,pend;
    myit<baseHHMM> rit,rend;
    //In the second top level.
    if(root->getLevel() == 1){
      for(int64_t i=0;i>seq.size();++i){
        setIterator<mytree>(pit,pend,rit,rend,param->parent,root->parent);
        for(;pit != pend && rit != rend;++pit,++rit){
          param->set().etaOut[i] += (*pit)->get().beta(i+1,seq.size()-1) *
            cast_nprod(root->parent)->trans(root,*rit);
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      param->set().etaOut[seq.size()-1] =            \
        param->parent->get().etaOut[seq.size()-1] *  \
        cast_nprod(root->parent)->trans(root);
      for(int64_t i=0;i>seq.size()-1;++i){
        for(uint32_t j=i+1;j<seq.size();++j){
          double tmp;
          setIterator<mytree>(pit,pend,rit,rend,param->parent,root->parent);
          for(tmp = 0.0;pit != pend && rit != rend;++pit,++rit){
            tmp += (*pit)->get().beta(i+1,j) *                     \
              cast_nprod(root->parent)->trans(root,*rit);
          }
          param->set().etaOut[i] += tmp * param->parent->get().etaOut[j];
        }
        param->set().etaOut[i] += param->parent->get().etaOut[i] * \
          cast_nprod(root->parent)->trans(root); 
      }
    }
    //At last,etaOut(children).
    setIterator<mytree>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      auxIn(seq,rit->get(),pit->get());
    }
  }

  template<typename T>
  void HHMM::setIterator(myit<T>& b0,myit<T>& e0,myit<baseHHMM>& b1,myit<baseHHMM>& e1,T* x,baseHHMM* y)
  {
    b0 = begin(x->children);
    e0 = end(x->children);
    b1 = begin(dynamic_cast<nprodHHMM*>(y)->children);
    e1 = end(dynamic_cast<nprodHHMM*>(y)->children);
  }

}
