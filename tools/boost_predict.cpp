#include <cstdlib>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <assert.h>


using namespace std;
using namespace cv;
// function declare
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


vector<float> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::vector<float> v;
    vector<string> sv = split(s, delim, elems);
    for (int i = 0; i < sv.size(); ++i)
    {
    	string myStream = sv[i];
		istringstream buffer(myStream);
		float value;
		buffer >> value;
		v.push_back(value);
    }
    return v;
}

int main(int argc, char** argv) {

	/* STEP 2. Opening the file */
	//1. Declare a structure to keep the data

	if(argc<1){
		cout<<"please provide the input filename"<<endl;
		return -1;
	}

    char *s1 = argv[1];
    string *s2 = new string (argv[1]);

    ifstream datafile(s2->c_str());
    ofstream outputFile("predictions.txt");
	CvBoost boost;
	boost.load("trained_boost.xml", "boost");


	float positive = 0;				
	float negative = 0;				
	float total_number = 0;
	int true_positive = 0;
	int true_negative = 0;
	int false_negative = 0;
	int false_positive = 0;

	if(datafile.is_open()){
		while(datafile.good()){
			string dataline = "";
			getline(datafile, dataline);
			if(dataline == "")
				break;

			vector<float> features = split(dataline, ',');

			Mat	featureMat = (Mat_<float>(1,9) << features[0],
											      features[1], 
   												  features[2], 
												  features[3],
												  features[4],
												  features[5],
												  features[6],
												  features[7],
												  features[8]);

			// This is a openCV Bug for missing_data mat
			Mat	missing_data = (Mat_<float>(1,9) << 0,0,0,0,0,0,0,0,0);

			outputFile<<boost.predict(featureMat, missing_data)<<' ';
			outputFile<<dataline<<endl;
			float prediction = boost.predict(featureMat, missing_data); 

			if(prediction == 1 && dataline[0] == 'a' ){
				true_positive++; // true shoot
				positive++;
			}
			else if(prediction != 1 && dataline[0] == 'a' ){
				false_positive++;
				positive ++;
			}
			else if(prediction == 1 && dataline[0] != 'a' ){
				true_negative++;
				negative++;
			}
			else if(prediction != 1 && dataline[0] != 'a' ){
				false_negative++;
				negative++;
			}
			total_number++;
		}
	}

	assert(total_number = negative + positive);

	float acc = (true_positive+false_negative)/total_number;

	cout<<"Accuracy:"<<acc<<endl;
	cout<<"type I error:"<<false_positive/positive<<endl;
	cout<<"type II error:"<<true_negative/negative<<endl;


	outputFile.close();
	datafile.close();
	return EXIT_SUCCESS;
}
