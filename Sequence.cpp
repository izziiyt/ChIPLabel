#ifndef SEQUENCE_CPP
#define SEQUENCE_CPP

#include "Sequence.h"

namespace hhmm{
  
  //  Sequence::Sequence(vector<uint32_t> const& _V,vector<uint32_t> _stateNum,uint32_t _dim)
  //    :len(_V.size()),
  //     testV(_V),
  //     param(_depth,_stateNum,_V.size(),_dim)
  //  {}

  Sequence::Sequence(vector<VectorXld> const& _V,vector<uint32_t> const& _stateNum,uint32_t _dim)
    :len(_V.size()),
     V(_V),
     param(_stateNum,_V.size(),_dim)
  {}

  void Sequence::print(string const& filename) const
  {
    ofstream of(filename);
    for(auto& s:state){
      of << s << endl;
    }
  }
}

#endif










