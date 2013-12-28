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

namespace hhmm{

  CPPUNIT_TEST_SUITE_REGISTRATION(TestHHMM);

  const uint32_t _STATENUM = 3;
  const uint32_t _DIM = 2;

  bool isNear(double x,double y)
  {
    return fabs(x-y) < x * DBL_EPSILON;
  }

  void TestHHMM::setUp()
  {
    hhmm = new HHMM(2,2,3);

    auto castn = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto castp = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    vector<uint32_t> obs = {0,1,1};

    vector<double> emit0 = {0.2,0.8};
    vector<double> emit1 = {0.6,0.4};
    MatrixXd trans(2,3);
    trans << 0.5,0.4,0.1,
      0.4,0.4,0.2;
    MatrixXd trans2(2,3);
    trans2 << 0.6,0.2,0.2,
      0.2,0.7,0.1;
    (hhmm->seq).push_back(up<Sequence>(new Sequence(obs,2,3)));
    (hhmm->root).cpyTransMat(trans2);
    castn(hhmm->root.children[0].get())->cpyTransMat(trans);
    castn(hhmm->root.children[1].get())->cpyTransMat(trans);
    castp(castn(hhmm->root.children[0].get())->children[0].get())->setEmit(emit0);
    castp(castn(hhmm->root.children[0].get())->children[1].get())->setEmit(emit1);
    castp(castn(hhmm->root.children[1].get())->children[0].get())->setEmit(emit0);
    castp(castn(hhmm->root.children[1].get())->children[1].get())->setEmit(emit1);
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
    VectorXd test(3);
    test << 3,4,3;
    
    VectorXd mean(3);
    mean << 1,3,3;
    
    DM var(3);
    for(uint32_t i=0;i<3;++i)
      var.diagonal()[i] = 1;
    auto a = prodHHMM(0,3,nullptr);

    a.swpMean(mean);
    a.swpVar(var);

    double estmt = pow(2*M_PI,-1.5)*exp(-2.5);
    
    CPPUNIT_ASSERT(a.emit(test) == estmt);
  }

  void TestHHMM::TestForward()
  {
    cout << "in the Forward algorithm" << endl;
    hhmm->forward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
  }

  void TestHHMM::TestBackward()
  {
    cout << "in the Backward algorithm" << endl;
    hhmm->backward(*(hhmm->seq[0]),&(hhmm->root),&(hhmm->seq[0]->param));
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
  //   for(uint32_t i=0;i<hhmm->seq[0]->size();++i){
  //     for(uint32_t j=0;j<=hhmm->stateNum;++j){
  //       cout << hhmm->seq[0]->param.children[0]->children[0]->xiContent(i,j) << " ";
  //     }
  //     cout << endl;
  //   }
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
  }
  
}

#endif

















