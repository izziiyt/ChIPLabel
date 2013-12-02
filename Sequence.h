#ifndef SEQUENCE_CPP
#define SEQUENCE_CPP

#include<vector>
#include<Eigen/Dense>
#include<memory>
#include <iostream>

using namespace std;
using namespace Eigen;

template <typename T>
using diVector = vector<vector<T>>;

template <typename T>
using triVector = vector<diVector<T>>;

template<typename T>
using up = unique_ptr<T>;

namespace hhmm{

  class HHMM;
  class TestHHMM;

  template<typename T>
  using diVector = vector<vector<T>>;

  template<typename T>
  class upperTriangle{
  protected:
    diVector<T> val;
  public:

    upperTriangle(uint32_t len,T x)
    {
      for(uint32_t i=0;i<len;++i)
        val.push_back(vector<T>(len-i,x));
      for_each(begin(val),end(val),
               [](vector<T>& x){x.shrink_to_fit();});
      val.shrink_to_fit();
    }

    T& operator()(uint32_t x,uint32_t y){
      return val[x][y-x];
    }
    void print() const{
      for(uint32_t i=0;i<val.size();++i){
        for(uint32_t j=0;j<val.size();++j){
          if(j < i){cout << 0 << " ";}
          else{cout << val[i][j-i] << " ";}
        }
        cout << endl;
      }
    }
  };

  template<typename T> 
  class tree{
  protected:
    T val;
  public:
    tree<T>* parent;
    vector<up<tree<T>>> children;

    void print() const{val.print();}
    tree(uint32_t,uint32_t,uint32_t,tree<T>*);
    T get() const{return val;}
    T& set(){return val;}
    void cpy(T const& _val){val = _val;}
    void swp(T& _val){swap(val,_val);}
  };

  // template<typename T>
  // tree<T>::tree(uint32_t _depth,uint32_t _childNum,uint32_t _length,tree* _parent)
  //   :parent(_parent)
  // {
  //   if(_depth != 1){
  //     for(uint32_t t=0;t<_childNum;++t){
  //       children.push_back(up<tree<T>>\
  //                          (new tree<T>(_depth-1,_childNum,_length,this)));
  //     }
  //     children.shrink_to_fit();
  //   }
  // }

  template<>
  tree<upperTriangle<double>>::tree(uint32_t _depth,uint32_t _childNum,\
                                    uint32_t _length,tree* _parent)
    :val(_length,0.0),
    parent(_parent)
  {
    if(_depth != 1){
      for(uint32_t t=0;t<_childNum;++t){
        children.push_back(up<tree<upperTriangle<double>>>
                           (new tree<upperTriangle<double>>
                            (_depth-1,_childNum,_length,this)));
      }
      children.shrink_to_fit();
    }
  }


  class Sequence{
    friend HHMM;
    friend TestHHMM;
  protected:
    uint32_t len;
    vector<VectorXd> V;//Observed sequence.
    vector<uint32_t> testV;//Fix Me.
    vector<uint32_t> S;//State sequence.
    tree<upperTriangle<double>> alpha;//forward variables
    tree<upperTriangle<double>> beta;//backward// ward variables
    // // diVector<double> beta;//backward bvariables
    // // vector<double> eta;//scaling variables
    // // diVector<double> xi;//gamma[t][state[t]]
    //  triVector<double> chi;//xi[t][state[t]][state[t+1]]
  public:

    Sequence(vector<VectorXd> const&,uint32_t,uint32_t);
    Sequence(vector<uint32_t> const&,uint32_t,uint32_t);
    virtual ~Sequence() = default;
    uint32_t obs(uint32_t i) const{return testV[i];}
    uint32_t size() const{return len;}
    void clearAlpha(){};
    void showAlpha(){alpha.print();}

  };
  
  Sequence::Sequence(vector<uint32_t> const& _V,uint32_t _stateNum,uint32_t _depth)
    :len(_V.size()),
     testV(_V),
     S(_V.size()),
     alpha(_depth,_stateNum,_V.size(),nullptr),
     beta(_depth,_stateNum,_V.size(),nullptr)
  {}

  // Sequence::Sequence(vector<VectorXd> const& _V,uint32_t _stateNum,uint32_t _depth)
  //   :V(_V),//Fix Me.
  //    S(_V.size()),
  //    alpha(_depth,_stateNum,nullptr,_stateNum)
  // {}

}

#endif










