#ifndef MYUTILITY_HPP
#define MYUTILITY_HPP

#include <tuple>
#include <iostream>

//std::cout for std::tuple
template<uint32_t IDX,uint32_t MAX,typename... Args>
struct PRINT_TUPLE{
  static void print(std::ostream& strm,const std::tuple<Args...>& t){
    strm << std::get<IDX>(t) << (IDX+1==MAX ? "":",");
    PRINT_TUPLE<IDX+1,MAX,Args...>::print(strm,t);
      }
};
template<uint32_t MAX,typename...Args>
struct PRINT_TUPLE<MAX,MAX,Args...>{
  static void print(std::ostream& strm,const std::tuple<Args...>& t){
  }
};
template<typename...Args>
std::ostream& operator << (std::ostream& strm,const std::tuple<Args...>& t)
{
  strm << "[";
  PRINT_TUPLE<0,sizeof...(Args),Args...>::print(strm,t);
  return strm << "]";
}


namespace ytd{

template<class T1,class T2,class Func>
Func double_for(T1& a,T2& b,Func f){

  if(a.size()!=b.size()){
    std::cerr << "size difference error"<<std::endl;
    exit(0);
  }

  auto x=a.begin();auto y=b.begin();

  for(;x!=a.end();++x,++y){
    f(*x,*y);
   }

  return f;
}

}
#endif
