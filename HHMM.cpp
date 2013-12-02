#include "HHMM.h"
#include <iostream>

namespace hhmm{

  HHMM::HHMM(uint32_t _dim,uint32_t _stateNum,uint32_t _depth)
    :root(_depth,_stateNum,_dim),
     dim(_dim),
     stateNum(_stateNum),
     depth(_depth){}

  void HHMM::forward(Sequence& seq,baseHHMM* root,tree<upperTriangle<double>>* alpha)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto cast_prod = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    using mytree = tree<upperTriangle<double>>;

    //Clear the alpha_values.
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        alpha->set()(i,j) = 0.0;
      }
    }
    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<mytree> ait,aend,bait,baend,cait,caend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,forward(children).
    for(setIterator<mytree>(ait,aend,rit,rend,alpha,root);  \
        ait != aend && rit != rend;++ait,++rit){
      forward(seq,rit->get(),ait->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(uint32_t i=0;i<seq.size();++i){
        for(setIterator<mytree>(ait,aend,rit,rend,alpha,root);  \
            ait != aend && rit != rend;++ait,++rit){
          (*ait)->set()(i,i) = (*rit)->getPi() * cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          for(setIterator<mytree>(ait,aend,rit,rend,alpha,root);  \
              ait != aend && rit != rend;++ait,++rit){
            for(setIterator<mytree>(bait,baend,brit,brend,alpha,root);  \
                bait != baend && brit != brend;++bait,++brit){
              (*ait)->set()(i,j) += (*bait)->get()(i,j-1) * \
                cast_nprod(root)->trans(*brit,*rit);
            }
            (*ait)->set()(i,j) *= cast_prod(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<mytree>(ait,aend,rit,rend,alpha,root);
        for(;ait != aend && rit != rend;++ait,++rit){
          setIterator<mytree>(cait,caend,crit,crend,ait->get(),rit->get());
          for(;cait != caend && crit != crend;++cait,++crit){
            (*ait)->set()(i,i) += (*cait)->get()(i,i) * \
              cast_nprod(rit->get())->trans(*crit);
          }
          (*ait)->set()(i,i) *= (*rit)->getPi();
        }
        double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(ait,aend,rit,rend,alpha,root);
          for(;ait != aend && rit != rend;++ait,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<mytree>(bait,baend,brit,brend,alpha,root);
              for(tmp0 = 0.0;bait != baend && brit != brend;++bait,++brit){
                tmp0 += (*bait)->get()(i,k) * cast_nprod(root)->trans(*brit,*rit);
              }
              setIterator<mytree>(cait,caend,crit,crend,ait->get(),rit->get());
              for(tmp1 = 0.0;cait != caend && crit != crend;++cait,++crit){
                tmp1 += (*cait)->get()(k+1,j) * cast_nprod(rit->get())->trans(*crit);
              }
              (*ait)->set()(i,j) += tmp0 * tmp1;
            }
            setIterator<mytree>(cait,caend,crit,crend,ait->get(),rit->get());
            for(tmp2 = 0.0;cait != caend && crit != crend;++cait,++crit){
              tmp2 += (*cait)->get()(i,j) * cast_nprod(rit->get())->trans(*crit);
            }
            (*ait)->set()(i,j) += tmp2 * (*rit)->getPi();
          }
        }
      }
    }
    // //If it is the top level.
    // if(root->getLevel() == 0){
    //   for(uint32_t i=0;i<seq.size();++i){
    //     setIterator<mytree>(cait,caend,crit,crend,alpha,root);
    //     for(;cait != caend && crit != crend;++cait,++crit){
    //       alpha->set()(i,i) += (*cait)->get()(i,i) * cast_nprod(root)->trans(*crit);
    //     }
    //     alpha->set()(i,i) *= root->getPi();
    //     double tmp1 = 0.0,tmp2 = 0.0;
    //     for(uint32_t j=i+1;j<seq.size();++j){
    //       for(uint32_t k=i;k<j;++k){
    //         setIterator<mytree>(cait,caend,crit,crend,alpha,root);
    //         for(tmp1 = 0.0;cait != caend && crit != crend;++cait,++crit){
    //           tmp1 += (*cait)->get()(k+1,j) * cast_nprod(root)->trans(*crit);
    //         }
    //         alpha->set()(i,j) += alpha->get()(i,k) * tmp1;
    //       }
    //       setIterator<mytree>(cait,caend,crit,crend,alpha,root);
    //       for(tmp2 = 0.0;cait != caend && crit != crend;++cait,++crit){
    //         tmp2 += (*cait)->get()(i,j) * cast_nprod(root)->trans(*crit);
    //       }
    //       alpha->set()(i,j) += tmp2 * root->getPi();
    //     }
    //   }
    // }
  }

  void HHMM::backward(Sequence& seq,baseHHMM* root,tree<upperTriangle<double>>* beta)
  {
    auto cast_nprod = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto cast_prod = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    using mytree = tree<upperTriangle<double>>;

    //Clear the beta_values.
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        beta->set()(i,j) = 0.0;
      }
    }
    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<mytree> bit,bend,bbit,bbend,cbit,cbend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,forward(children).
    setIterator<mytree>(bit,bend,rit,rend,beta,root);
    for(;bit != bend && rit != rend;++bit,++rit){
     backward(seq,rit->get(),bit->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<mytree>(bit,bend,rit,rend,beta,root);
        for(;bit != bend && rit != rend;++bit,++rit){
          (*bit)->set()(i,i) = cast_nprod(root)->trans(*rit) * \
            cast_prod(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(bit,bend,rit,rend,beta,root);
          for(;bit != bend && rit != rend;++bit,++rit){
            setIterator<mytree>(bbit,bbend,brit,brend,beta,root);
            for(;bbit != bbend && brit != brend;++bbit,++brit){
              (*bit)->set()(i,j) += (*bbit)->get()(i+1,j) * \
                cast_nprod(root)->trans(*rit,*brit);
            }
            (*bit)->set()(i,j) *= cast_prod(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(int64_t i=seq.size()-1;i>-1;--i){
        setIterator<mytree>(bit,bend,rit,rend,beta,root);
        for(;bit != bend && rit != rend;++bit,++rit){
          setIterator<mytree>(cbit,cbend,crit,crend,bit->get(),rit->get());
          for(;cbit != cbend && crit != crend;++cbit,++crit){
            (*bit)->set()(i,i) += (*cbit)->get()(i,i) * (*crit)->getPi();
          }
          (*bit)->set()(i,i) *= cast_nprod(root)->trans(*rit);
        }
        double tmp0,tmp1,tmp2;
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<mytree>(bit,bend,rit,rend,beta,root);
          for(;bit != bend && rit != rend;++bit,++rit){
            for(uint32_t k=i;k<j;++k){
              setIterator<mytree>(bbit,bbend,brit,brend,beta,root);
              for(tmp0 = 0.0;bbit != bbend && brit != brend;++bbit,++brit){
                tmp0 += (*bbit)->get()(k+1,j) * cast_nprod(root)->trans(*rit,*brit);
              }
              setIterator<mytree>(cbit,cbend,crit,crend,bit->get(),rit->get());
              for(tmp1 = 0.0;cbit != cbend && crit != crend;++cbit,++crit){
                tmp1 += (*cbit)->get()(i,k) * (*crit)->getPi();
              }
              (*bit)->set()(i,j) += tmp0 * tmp1;
            }
            setIterator<mytree>(cbit,cbend,crit,crend,bit->get(),rit->get());
            for(tmp2 = 0.0;cbit != cbend && crit != crend;++cbit,++crit){
              tmp2 += (*cbit)->get()(i,j) * (*crit)->getPi();
            }
            (*bit)->set()(i,j) += tmp2 * cast_nprod(root)->trans(*rit);
          }
        }
      }
    }
    // //If it is the top level.
    // if(root->getLevel() == 0){
    //   for(uint32_t i=0;i<seq.size();++i){
    //     setIterator<mytree>(cbit,cbend,crit,crend,beta,root);
    //     for(;cbit != cbend && crit != crend;++bit,++rit){
    //       beta->set()(i,i) += (*cbit)->get()(i,i) * (root)->getPi();
    //     }
    //     double tmp1,tmp2;
    //     for(uint32_t j=i+1;j<seq.size();++j){
    //       for(uint32_t k=i;k<j;++k){
    //         setIterator<mytree>(cbit,cbend,crit,crend,bit->get(),rit->get());
    //         for(tmp1 = 0.0;cbit != cbend && crit != crend;++cbit,++crit){
    //           tmp1 += (*cbit)->get()(i,k) * (*crit)->getPi();
    //         }
    //         beta->set()(i,j) += beta->get()(k+1,j) * tmp1;
    //       }
    //       setIterator<mytree>(cbit,cbend,crit,crend,beta,root);
    //       for(tmp2 = 0.0;cbit != cbend && crit != crend;++cbit,++crit){
    //         tmp2 += (*cbit)->get()(i,j) * (*crit)->getPi();
    //       }
    //       beta->set()(i,j) += tmp2 * cast_nprod(root)->trans(*rit);
    //     }
    //   }
    // }
  }

  template<typename T>
  void HHMM::setIterator(myit<T>& b0,myit<T>& e0,myit<baseHHMM>& b1,myit<baseHHMM>& e1,
                         T* x,baseHHMM* y)
  {
    b0 = begin(x->children);
    e0 = end(x->children);
    b1 = begin(dynamic_cast<nprodHHMM*>(y)->children);
    e1 = end(dynamic_cast<nprodHHMM*>(y)->children);
  }

}

















