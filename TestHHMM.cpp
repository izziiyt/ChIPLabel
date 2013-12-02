#ifndef TESTHHMM_CPP
#define TESTHHMM_CPP

#include "TestHHMM.h"
#include <iostream>
#include <utility>
#include <cmath>
#include <cfloat>
#include <dirent.h>
#include <fstream>

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
    hhmm = new HHMM(3,2,2);
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
    vector<uint32_t> obs = {0,1,1};
    vector<double> emit0 = {0.2,0.8};
    vector<double> emit1 = {0.6,0.4};
    Sequence seq(obs,2,2);
    HHMM hhmm0(2,2,2);
    MatrixXd trans(2,3);
    trans << 0.6,0.3,0.1,
      0.4,0.5,0.1;
    hhmm0.root.cpyTransMat(trans);
    dynamic_cast<prodHHMM*>(hhmm0.root.children[0].get())->setEmit(emit0);
    dynamic_cast<prodHHMM*>(hhmm0.root.children[1].get())->setEmit(emit1);
    hhmm0.root.children[0].get()->setPi(0.7);
    hhmm0.root.children[1].get()->setPi(0.3);
    hhmm0.forward(seq,&(hhmm0.root),&(seq.alpha));
    // seq.alpha.print();
    // seq.alpha.children[0]->print();
    // seq.alpha.children[1]->print();
  }

  void TestHHMM::TestBackward()
  {
    cout << "in the Backward algorithm" << endl;
    auto castn = [](baseHHMM* x){return dynamic_cast<nprodHHMM*>(x);};
    auto castp = [](baseHHMM* x){return dynamic_cast<prodHHMM*>(x);};
    vector<uint32_t> obs = {0,1,1};
    vector<double> emit0 = {0.2,0.8};
    vector<double> emit1 = {0.6,0.4};
    Sequence seq(obs,2,3);
    HHMM hhmm0(2,2,3);
    MatrixXd trans(2,3);
    trans << 0.5,0.4,0.1,
      0.4,0.4,0.2;
    MatrixXd trans2(2,3);
    trans2 << 0.6,0.2,0.2,
      0.2,0.7,0.1;
    (&hhmm0.root)->cpyTransMat(trans2);
    castn(hhmm0.root.children[0].get())->cpyTransMat(trans);
    castn(hhmm0.root.children[1].get())->cpyTransMat(trans);
    castp(castn(hhmm0.root.children[0].get())->children[0].get())->setEmit(emit0);
    castp(castn(hhmm0.root.children[0].get())->children[1].get())->setEmit(emit1);
    castp(castn(hhmm0.root.children[1].get())->children[0].get())->setEmit(emit0);
    castp(castn(hhmm0.root.children[1].get())->children[1].get())->setEmit(emit1);
    castn(hhmm0.root.children[0].get())->children[0]->setPi(0.3);
    castn(hhmm0.root.children[0].get())->children[1]->setPi(0.7);
    castn(hhmm0.root.children[1].get())->children[0]->setPi(0.3);
    castn(hhmm0.root.children[1].get())->children[1]->setPi(0.7);
    hhmm0.root.children[0]->setPi(0.4);
    hhmm0.root.children[1]->setPi(0.4);
    hhmm0.backward(seq,&(hhmm0.root),&(seq.beta));
    // seq.beta.children[0]->print();
    // seq.beta.children[1]->print();
    // seq.beta.children[0]->children[0]->print();
    // seq.beta.children[0]->children[1]->print();
  }
}

#endif











