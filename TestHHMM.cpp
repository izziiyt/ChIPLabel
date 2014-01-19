#ifndef TESTHHMM_CPP
#define TESTHHMM_CPP

#include "TestHHMM.h"
#include <iostream>
#include <utility>
#include <cmath>
#include <cfloat>
#include <dirent.h>
#include <fstream>
#include <memory>
#include "readSample.cpp"

namespace hhmm{

  CPPUNIT_TEST_SUITE_REGISTRATION(TestHHMM);

  const uint32_t _STATENUM = 3;
  const uint32_t _DIM = 2;

  bool isNear(long double x,long double y)
  {
    return fabs(x-y) < x * DBL_EPSILON;
  }

  void TestHHMM::setUp()
  {
    hhmm = new HHMM(2,2,3);

    auto castn = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto castp = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    vector<VectorXld> obs0 = read("sample0.txt");
    vector<VectorXld> obs1 = read("sample1.txt");
    //    vector<uint32_t> obs1 = readSample("../sample1.txt");
    //    vector<long double> emit0 = {0.2,0.8};
    //    vector<long double> emit1 = {0.6,0.4};
    //for(auto& a:obs0){cout << a << endl;}
    VectorXld mean0(2);
    mean0 << 3.0,2.0;
    VectorXld mean1(2);
    mean1 << 4.0,1.0;
    MatrixXld variance0(2,2);
    variance0 << 1.5,0.0,
      0.0,2.5;
    MatrixXld variance1(2,2);
    variance1 << 2.3,0.0,
      0.0,1.2;
    MatrixXld trans(2,3);
    trans << 0.5,0.4,0.1,
      0.4,0.4,0.2;
    MatrixXld trans2(2,3);
    trans2 << 0.6,0.3,0.1,
      0.2,0.6,0.2;
    (hhmm->seq).push_back(up<Sequence>(new Sequence(obs0,2,3,2)));
    (hhmm->seq).push_back(up<Sequence>(new Sequence(obs1,2,3,2)));
    (hhmm->root).cpyTransMat(trans2);
    castn(hhmm->root.children[0].get())->cpyTransMat(trans);
    castn(hhmm->root.children[1].get())->cpyTransMat(trans);
    castp(castn(hhmm->root.children[0].get())->children[0].get())->setMean() = mean0;
    castp(castn(hhmm->root.children[0].get())->children[0].get())->setVariance().diagonal() = variance0.diagonal();
    castp(castn(hhmm->root.children[0].get())->children[1].get())->setMean() = mean1;
    castp(castn(hhmm->root.children[0].get())->children[1].get())->setVariance().diagonal() = variance1.diagonal();
    castp(castn(hhmm->root.children[1].get())->children[0].get())->setMean() = mean0;
    castp(castn(hhmm->root.children[1].get())->children[0].get())->setVariance().diagonal() = variance0.diagonal();
    castp(castn(hhmm->root.children[1].get())->children[1].get())->setMean() = mean1;
    castp(castn(hhmm->root.children[1].get())->children[1].get())->setVariance().diagonal() = variance1.diagonal();

    castn(hhmm->root.children[0].get())->children[0]->setPi(0.3);
    castn(hhmm->root.children[0].get())->children[1]->setPi(0.7);
    castn(hhmm->root.children[1].get())->children[0]->setPi(0.3);
    castn(hhmm->root.children[1].get())->children[1]->setPi(0.7);
    hhmm->root.children[0]->setPi(0.4);
    hhmm->root.children[1]->setPi(0.6);
  }

  void TestHHMM::tearDown()
  {
    delete hhmm;
  }

  void TestHHMM::TestEmitProb()
  {
    VectorXld test(3);
    test << 3,4,3;
    
    VectorXld mean(3);
    mean << 1,3,3;
    
    DM var(3);
    for(uint32_t i=0;i<3;++i)
      var.diagonal()[i] = 1;
    auto a = prodHHMM(0,3,nullptr);

    a.swpMean(mean);
    a.swpVar(var);

    long double estmt = pow(2*M_PI,-1.5)*exp(-2.5);
    cout << a.emit(test) << " " << estmt << endl;
    CPPUNIT_ASSERT(a.emit(test) == estmt);
  }

  void TestHHMM::TestForward()
  {
    cout << "in the Forward algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    cout << hhmm->seq[0]->param.children[0]->alpha(0,0) << endl;
  }

  void TestHHMM::TestBackward()
  {
    cout << "in the Backward algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    //debugAlphaAndBeta(*hhmm,*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  }

  void TestHHMM::debugAlphaAndBeta(HHMM& hhmm,Sequence& seq,baseHHMM* root,parameters* param)
  {
    //If it is the deepest level,it returns;
    if(root->getLevel() == hhmm.depth-1){return;}
    //Declear iterators.
    myit<parameters> pit,pend;
    myit<baseHHMM> rit,rend;
    hhmm.setIterator<parameters>(pit,pend,rit,rend,param,root);
    for(;pit != pend && rit != rend;++pit,++rit){
      debugAlphaAndBeta(hhmm,seq,rit->get(),pit->get());
    }
    cout << "level is " << root->getLevel()+1 << endl;
    for(uint32_t i=0;i<seq.size();++i){
      for(uint32_t j=i;j<seq.size();++j){
        long double BETA = 0.0,ALPHA = 0.0;
        hhmm.setIterator<parameters>(pit,pend,rit,rend,param,root);
        for(;pit != pend && rit != rend;++pit,++rit){
          BETA += (*pit)->beta(i,j) * (*rit)->getPi();
          ALPHA += (*pit)->alpha(i,j) * cast_nprod(root)->trans(rit->get());
        }
        cout << "index: " << i << " " << j << endl;
        cout << BETA << " " << ALPHA << endl;
      }
    }
  }

  void TestHHMM::TestAuxIn()
  {
    cout << "in the AuxIn algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    // auto& eI0 = hhmm->seq[0]->param.children[0]->etaIn;
    // for(auto& a:eI0){cout << a << " ";}
    // cout << endl;
    // auto& eI1 = hhmm->seq[0]->param.children[1]->etaIn;
    // for(auto& a:eI1){cout << a << " ";}
    // cout << endl;
    // auto& eI00 = hhmm->seq[0]->param.children[0]->children[0]->etaIn;
    // for(auto& a:eI00){cout << a << " ";}
    // cout << endl;
    // auto& eI11 = hhmm->seq[0]->param.children[0]->children[1]->etaIn;
    // for(auto& a:eI11){cout << a << " ";}
    // cout << endl;
  }

  void TestHHMM::TestAuxOut()
  {
    cout << "in the AuxOut algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    // auto& eO = hhmm->seq[0]->param.children[0]->etaOut;
    // for(auto& a:eO){cout << a << " ";}
    // cout << endl;
    // auto& e1 = hhmm->seq[0]->param.children[1]->etaOut;
    // for(auto& a:e1){cout << a << " ";}
    // cout << endl;
    // auto& e00 = hhmm->seq[0]->param.children[0]->children[0]->etaOut;
    // for(auto& a:e00){cout << a << " ";}
    // cout << endl;
    // auto& e01 = hhmm->seq[0]->param.children[0]->children[1]->etaOut;
    // for(auto& a:e01){cout << a << " ";}
    // cout << endl;
  }

  void TestHHMM::TestHorizon()
  {
    cout << "in the Horizon algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->horizon(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    // for(uint32_t i=0;i<hhmm->seq[0]->size();++i){
    //   for(uint32_t j=0;j<=hhmm->stateNum;++j){
    //     cout << hhmm->seq[0]->param.children[0]->children[0]->xiContent(i,j) << " ";
    //   }
    //   cout << endl;
    // }
  }

  void TestHHMM::TestVertical()
  {
    cout << "in the Vertical algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->vertical(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    for(uint32_t i=0;i<hhmm->seq[0]->size();++i){
       cout << hhmm->seq[0]->param.children[0]->chi[i] << " ";
     }
    cout << endl;
    for(uint32_t i=0;i<hhmm->seq[0]->size();++i){
       cout << hhmm->seq[0]->param.children[1]->chi[i] << " ";
     }
     cout << endl;
  }

  void TestHHMM::TestCalcGamma()
  {
    cout << "in the calcGamma algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->vertical(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    for(uint32_t i=0;i<hhmm->seq[0]->size();++i){
      cout << hhmm->seq[0]->param.children[0]->gammaIn[i] << " ";
    }
    cout << endl;
  }

  void TestHHMM::TestCalcTmpPi()
  {
    cout << "in the calcTmpPi algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->vertical(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
    hhmm->calcTmpPi(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
  }
  void TestHHMM::TestCalcTmpTrans()
  {
    cout << "in the calcTmpTrans algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
    hhmm->vertical(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
    hhmm->calcTmpTrans(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
  }
  // void TestHHMM::TestCalcTmpEmit()
  // {
  //   cout << "in the calcTmpEmit algorithm" << endl;
  //   hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  //   hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  //   hhmm->auxOut(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  //   hhmm->auxIn(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  //   hhmm->vertical(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
  //   hhmm->calcTmpEmit(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));    
  // }
  
  void TestHHMM::TestEM()
  {
    cout << "in the EM algorithm" << endl;
    hhmm->EM();
  }

}

#endif

















