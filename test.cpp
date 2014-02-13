#include <vector>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

int main()
{
  cout << "start" << endl;
  ofstream ofs("sample.txt",ios::out | ios::app);
  ofs << "hoge" << endl;
}
