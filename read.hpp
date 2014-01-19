#ifndef READ_HPP
#define READ_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <sstream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

using VectorXld = Matrix<long double,Dynamic,1>;

namespace hhmm{

  vector<VectorXld> read(string const& filename)
  {
    vector<VectorXld> result;
    string str;
    string substr;
    fstream ifs(filename);
    while(getline(ifs,str)){

      istringstream iss(str);

      iss >> substr;
      //string chrname = substr;

      iss >> substr;
      //uint32_t indexBegin = stoi(substr);
      iss >> substr;
      //uint32_t indexEnd = stoi(substr);
    
      vector<long double> tmp;
      while(iss >> substr){
        long double obs = stod(substr);
        tmp.push_back(obs);
      }
      VectorXld innerResult(tmp.size());
      for(uint32_t i=0;i<tmp.size();++i){innerResult(i) = tmp[i];}
      result.push_back(innerResult);
    }
    result.shrink_to_fit();
    return result;
  }

}

#endif









