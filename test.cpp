#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main(){

  Vector2d v0(2,3);
  DiagonalMatrix<double,2> d;
  d.diagonal()[0]=2;
  d.diagonal()[1]=5;
  cout << v0.transpose() * d * v0 << endl;
}


















