#include <cstdlib>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

// function declare
void generateData();
vector< vector<float> > readText(char* fileName);

int maxClusterSize, maxFrameIndex;

int main(int argc, char** argv) {

	generateData();

	/* STEP 2. Opening the file */
	//1. Declare a structure to keep the data
	CvMLData cvml;
	//2. Read the file
	cvml.read_csv("samples.csv");
	//3. Indicate which column is the response
	cvml.set_response_idx(0);

	/* STEP 3. Splitting the samples */
	//1. Select 40 for the training
	CvTrainTestSplit cvtts(15, true);
	//2. Assign the division to the data
	cvml.set_train_test_split(&cvtts);

	printf("Training ... ");
	/* STEP 4. The training */
	//1. Declare the classifier
	CvBoost boost;
	//2. Train it with 100 features
	boost.train(&cvml, CvBoostParams(CvBoost::REAL, 100, 0, 1, false, 0), false);

	/* STEP 5. Calculating the testing and training error */
	// 1. Declare a couple of vectors to save the predictions of each sample
	vector<float> train_responses; 
	vector<float> test_responses;
	// 2. Calculate the training error
	float fl1 = boost.calc_error(&cvml,CV_TRAIN_ERROR,&train_responses);
	// 3. Calculate the test error
	float fl2 = boost.calc_error(&cvml,CV_TEST_ERROR,&test_responses);
	printf("Error train %f \n", fl1);
	printf("Error test %f \n", fl2);

	/* STEP 6. Save your classifier */
	// Save the trained classifier
	boost.save("./trained_boost.xml", "boost");

	return EXIT_SUCCESS;
}


// generate Training data and test data
void generateData(){
	maxClusterSize = 0;
	maxFrameIndex = 0;
	vector< vector<float> > loca_diff = readText("loca_diff.txt");

	cout<<loca_diff.size()<<endl;

}

vector< vector<float> > readText(char* fileName){

	ifstream infile(fileName);

	int clusterNumber, frameIndex;
	float data;

	vector< vector<float> > v;

	while( !infile.eof() && infile>>clusterNumber>>frameIndex>>data){
		if(maxClusterSize < clusterNumber + 1){
			v.resize(clusterNumber + 1);
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(maxFrameIndex + 1);
			}	
			maxClusterSize = clusterNumber + 1;
		}

		if(maxFrameIndex < frameIndex + 1){
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(frameIndex + 1);
			}	
			maxFrameIndex = frameIndex;
		}
		v[clusterNumber][frameIndex] = data;
	}
	return v;
}
