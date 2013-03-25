#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

	// to input the ground truth, we define a input protocol 
	/* 
	clusterNumber, framestart, frameend
	example 
	1,20,50

	then truth.txt will be writen as 
	1 20 1
	1 21 1
	1 22 1 
	1 23 1
	...
	1 50 1
	*/
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


vector<int> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::vector<int> v;
    vector<string> sv = split(s, delim, elems);
    for (int i = 0; i < sv.size(); ++i)
    {
    	string myStream = sv[i];
		istringstream buffer(myStream);
		int value;
		buffer >> value;
		v.push_back(value);
    }
    return v;
}

int main(int argc, char const *argv[])
{
	string a;

	ofstream out;
	out.open("truth.txt");

	while(cin>>a){
		if(a == "q")
			break;

		vector<int> v = split(a,',');

		int clusterNumber = v[0];
		int startNumber = v[1];
		int endNumber = v[2];

		for(int i = startNumber; i<= endNumber; i++){
			out<<clusterNumber<<' '<<i<<' '<<'1'<<endl;
		}

		cout<<"current frameIndex "<<endNumber<<endl;
	}

	out.close();
	cout<<"end normally"<<endl;
	return 0;
}

