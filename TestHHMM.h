#ifndef TESTHHMM_H
#define TESTHHMM_H

#include <cppunit/extensions/HelperMacros.h>
#include "HHMM.hpp"

namespace hhmm{

  class TestHHMM: public CPPUNIT_NS::TestFixture
  {
  private:  
    CPPUNIT_TEST_SUITE(TestHHMM);
    CPPUNIT_TEST(TestEmitProb);
    CPPUNIT_TEST(TestForward);
    CPPUNIT_TEST(TestBackward);
    CPPUNIT_TEST(TestAuxIn);
    CPPUNIT_TEST(TestAuxOut);
    CPPUNIT_TEST(TestHorizon);
    CPPUNIT_TEST(TestVertical);
    CPPUNIT_TEST_SUITE_END();
    HHMM* hhmm;
  public:
    void setUp();
    void tearDown();
  protected:
    void TestConstruct();
    void TestEmitProb();
    void TestForward();
    void TestBackward();
    void TestAuxIn();
    void TestAuxOut();
    void TestHorizon();
    void TestVertical();
    void TestCalcGamma();
  };
}
#endif
