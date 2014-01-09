#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <sstream>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

vector<VectorXd> readSample(string const& filename)
{
  vector<VectorXd> result;
  string str;
  string substr;
  fstream ifs(filename);
  while(getline(ifs,str)){

    istringstream iss(str);

    iss >> substr;
    //string chrname = substr;

    iss >> substr;
    //uint32_t index = stoi(substr);
    
    vector<double> tmp;
    while(iss >> substr){
      double obs = stod(substr);
      tmp.push_back(obs);
    }
    VectorXd innerResult(tmp.size());
    for(uint32_t i=0;i<tmp.size();++i){innerResult(i) = tmp[i];}
    result.push_back(innerResult);
  }
  result.shrink_to_fit();
  return result;
}











