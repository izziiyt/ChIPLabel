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
    cout << "--------------- " << endl;
    //Clear the alpha_values.
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        alpha->set()(i,j) = 0.0;
      }
    }
    //If it is the deepest level,it returns;
    if(root->getLevel() == depth-1){return;}
    //Declear iterators.
    myit<tree<upperTriangle<double>>> ait,aend,bait,baend,cait,caend;
    myit<baseHHMM> rit,rend,brit,brend,crit,crend;
    //At first,forward(children).
    setIterator<tree<upperTriangle<double>>>(ait,aend,rit,rend,alpha,root);
    for(;ait != aend && rit != rend;++ait,++rit){
      forward(seq,rit->get(),ait->get());
    }
    //In the second deepest level.
    if(root->getLevel() == depth-2){
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<tree<upperTriangle<double>>>(ait,aend,rit,rend,alpha,root);
        for(;ait != aend && rit != rend;++ait,++rit){
          (*ait)->set()(i,i) = (*rit)->getPi() *                    \
            dynamic_cast<prodHHMM*>(rit->get())->emit(seq.obs(i));
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<tree<upperTriangle<double>>>(ait,aend,rit,rend,alpha,root);
          for(;ait != aend && rit != rend;++ait,++rit){
            setIterator<tree<upperTriangle<double>>>(bait,baend,brit,brend,alpha,root);
            for(;bait != baend && brit != brend;++bait,++brit){
              (*ait)->set()(i,j) += (*bait)->get()(i,j-1) *         \
                dynamic_cast<nprodHHMM*>(root)->trans(*brit,*rit);
            }
            (*ait)->set()(i,j) *= dynamic_cast<prodHHMM*>(rit->get())->emit(seq.obs(i));
          }
        }
      }
    }
    //In the level that is not the deepest nor the second deepest.  
    else{
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<tree<upperTriangle<double>>>(ait,aend,rit,rend,alpha,root);
        for(;ait != aend && rit != rend;++ait,++rit){
          setIterator<tree<upperTriangle<double>>>          \
            (cait,caend,crit,crend,ait->get(),rit->get());
          for(;cait != caend && crit != crend;++ait,++rit){
            (*ait)->set()(i,i) += (*cait)->get()(i,i) *           \
              dynamic_cast<nprodHHMM*>(rit->get())->trans(*crit);
          }
          (*ait)->set()(i,i) *= (*rit)->getPi();
        }
        for(uint32_t j=i+1;j<seq.size();++j){
          setIterator<tree<upperTriangle<double>>>(ait,aend,rit,rend,alpha,root);
          for(;ait != aend && rit != rend;++ait,++rit){
            double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
            for(uint32_t k=i;k<j;++k){
              setIterator<tree<upperTriangle<double>>>  \
                (bait,baend,brit,brend,alpha,root);
              for(;bait != baend && brit != brend;++bait,++brit){
                tmp0 += (*bait)->get()(i,k) *                         \
                  dynamic_cast<nprodHHMM*>(root)->trans(*brit,*rit);
              }
              setIterator<tree<upperTriangle<double>>>          \
                (cait,caend,crit,crend,ait->get(),rit->get());
              for(;cait != caend && crit != crend;++cait,++crit){
                tmp1 += (*cait)->get()(k+1,j) *                       \
                  dynamic_cast<nprodHHMM*>(rit->get())->trans(*crit);
              }
            }
            (*ait)->set()(i,j) += tmp0 * tmp1;
            setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
            for(;cait != caend && crit != crend;++cait,++crit){
              tmp2 += (*cait)->get()(i,j) *                       \
                dynamic_cast<nprodHHMM*>(rit->get())->trans(*crit);
            }
            (*ait)->set()(i,j) += tmp2 * (*rit)->getPi();
          }
        }
      }
    }
    //If it is the top level.
    if(root->getLevel() == 0){
      for(uint32_t i=0;i<seq.size();++i){
        setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
        for(;cait != caend && crit != crend;++cait,++crit){
          alpha->set()(i,i) += (*cait)->get()(i,i) *             \
            dynamic_cast<nprodHHMM*>(root)->trans(*crit);
        }
        alpha->set()(i,i) *= root->getPi();
        for(uint32_t j=i+1;j<seq.size();++j){
          double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
          for(uint32_t k=i;k<j;++k){
            tmp0 += alpha->get()(i,k);
            setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
            for(;cait != caend && crit != crend;++cait,++crit){
              tmp1 += (*cait)->get()(k+1,j) *                 \
                dynamic_cast<nprodHHMM*>(root)->trans(*crit);
            }
          }
          cout << tmp0 << " " << tmp1 << endl;
          alpha->set()(i,j) += tmp0 * tmp1;
          setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
          for(;cait != caend && crit != crend;++cait,++crit){
            tmp2 += (*cait)->get()(i,j) * dynamic_cast<nprodHHMM*>(root)->trans(*crit);
          }
          cout << tmp2 << endl;
          alpha->set()(i,j) += tmp2 * root->getPi();
        }
      }
    }
  }


  
  // void HHMM::forward(Sequence& seq,baseHHMM* root,tree<upperTriangle<double>>* alpha)
  // {
  //   cout << "--------------------------" << endl;
  //   //Clear the alpha_values.
  //   for(uint32_t i=0;i<seq.size();++i){
  //     for(uint32_t j=i;j<seq.size();++j){
  //       alpha->set()(i,j) = 0.0;
  //     }
  //   }
  //   //Declear the  iterators.(b,c) == (brother,child)
  //   myit<tree<upperTriangle<double>>> bait,baend,cait,caend;
  //   myit<baseHHMM> brit,brend,crit,crend;
  //   //In the deepest level.
  //   if(root->getLevel() == depth-1){
  //     for(uint32_t i=0;i<seq.size();++i){
  //       alpha->set()(i,i) = root->getPi() *\
  //         dynamic_cast<prodHHMM*>(root)->emit(seq.obs(i));
  //       for(uint32_t j=i+1;j<seq.size();++j){
  //         setIterator<tree<upperTriangle<double>>>\
  //           (bait,baend,brit,brend,alpha->parent,root->parent);
  //         uint32_t p=0;
  //         for(;bait != baend && brit != brend;++bait,++brit){
  //           cout << i << " " << j << endl;
  //           cout << (*bait)->get()(i,j-1) << endl;
  //           alpha->set()(i,j) += (*bait)->get()(i,j-1) *           \
  //             dynamic_cast<nprodHHMM*>(root->parent)->trans(*brit,root);
  //         }
  //         cout << i << " " << j << " " << alpha->get()(i,j) << endl;
  //         alpha->set()(i,j) *= dynamic_cast<prodHHMM*>(root)->emit(seq.obs(i));
  //       }
  //     }
  //   }
  //   //In the level that is not the deepest.
  //   else{
  //     //At first,forward(children).
  //     setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
  //     for(;cait != caend && crit != crend;++cait,++crit){
  //       forward(seq,crit->get(),cait->get());
  //     }
  //     for(uint32_t i=0;i<seq.size();++i){
  //       setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
  //       for(;cait != caend && crit != crend;++cait,++crit){
  //         alpha->set()(i,i) += (*cait)->get()(i,i) *           \
  //           dynamic_cast<nprodHHMM*>(root)->trans(*crit);
  //       }
  //       alpha->set()(i,i) *= root->getPi();
  //       for(uint32_t j=i;j<seq.size();++j){
  //         double tmp0 = 0.0,tmp1 = 0.0,tmp2 = 0.0;
  //         for(uint32_t k=i;k<j;++k){
  //           //If it is the top level.
  //           if(root->getLevel() == 0){
  //             tmp0 += alpha->get()(i,k);
  //           }
  //           //If it is not the top level.
  //           else{
  //             setIterator<tree<upperTriangle<double>>>            \
  //               (bait,baend,brit,brend,alpha->parent,root->parent);
  //             for(;bait != baend && brit != brend;++bait,++brit){
  //               tmp0 += (*bait)->get()(i,k) *                           \
  //                 dynamic_cast<nprodHHMM*>(root->parent)->trans(*brit,root);
  //             }
  //           }
  //           setIterator<tree<upperTriangle<double>>>            \
  //             (cait,caend,crit,crend,alpha,root);
  //           for(;cait != caend && crit != crend;++cait,++crit){
  //             tmp1 += (*cait)->get()(k+1,j) * \
  //               dynamic_cast<nprodHHMM*>(root)->trans(*crit);
  //           }
  //           alpha->set()(i,j) += tmp0 * tmp1;
  //         }
  //         setIterator<tree<upperTriangle<double>>>(cait,caend,crit,crend,alpha,root);
  //           for(;cait != caend && crit != crend;++cait,++crit){
  //             tmp2 += (*cait)->get()(i,j) * dynamic_cast<nprodHHMM*>(root)->trans(*crit);
  //           }
  //           alpha->set()(i,j) += tmp2 * root->getPi();
  //       }
  //     }
  //   }
  // }
  
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

















