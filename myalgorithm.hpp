#ifndef MYALGORITHM_HPP
#define MYALGORITHM_HPP

#include<vector>
#include<random>
#include<numeric>
#include"myutility.hpp"
#include<cstdint>

namespace ytd{

  //DBL is double or long double.
  template<typename DBL>
  class dirichlet_distribution{
  private:
    std::vector<uint32_t> alpha;
  public:
    dirichlet_distribution(uint32_t size)
      :alpha(size)
    {}

    dirichlet_distribution(const std::vector<uint32_t>& args)
      :alpha(args)
    {}

    template<typename T1>
    void setParam(const T1& args)
    {
      std::copy(begin(args),end(args),begin(alpha));
    }

    template<typename T2>
    std::vector<DBL> operator()(T2& gen) const
    {
      std::vector<DBL> beta(alpha.size());
    
      ytd::double_for(alpha,beta,[&](const int& a,DBL& b){
          std::gamma_distribution<DBL> dist(a,1.0);
          b = dist(gen);
        });

      DBL beta_sum = accumulate(begin(beta),end(beta),0.0);
      for(auto& b:beta){b = b/beta_sum;}
      return beta;
    }

    template<typename T2>
    void operator()(T2& gen,std::vector<DBL>& args) const
    {
      if(args.size() != alpha.size()){
        std::cout << "Error In ytd::dirichlet_distribution::operator()" \
                  << std::endl;
        exit(1);
      }
    
      ytd::double_for(alpha,args,[&](const int& a,DBL& b){
          std::gamma_distribution<DBL> dist(a,1.0);
          b = dist(gen);
        });

      DBL argsSum = accumulate(begin(args),end(args),0.0);
      for(auto& b:args){b = b/argsSum;}
    }
  };
}

#endif
