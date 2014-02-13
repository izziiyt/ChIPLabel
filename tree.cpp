#ifndef TREE_H
#define TREE_H

#include <vector>
#include <memory>
#include <string>
#include <sstream>

using namespace std;

template<typename T> 
using up = unique_ptr<T>;

namespace hhmm{
  
  template<typename T>
  class tree{
  private:
    T value;
    vector<up<tree<T>> children;
    tree<T>* mother;
  public:
    T& operator ()()
    {
      return value;
    }
    
    //constructor for root
    tree<T>(vector<uint32_t> const& list)
      childre
    {

  };
    
  


}

#endif TREE_H
