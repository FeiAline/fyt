#include <cstdlib>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

// function declare

int main(int argc, char** argv) {

	//generateData();

	/* STEP 2. Opening the file */
	//1. Declare a structure to keep the data
	CvMLData cvml;
	//2. Read the file
	cvml.read_csv("training_boost.txt");
	//3. Indicate which column is the response
	cvml.set_response_idx(0);

	cout<<"Please input the number of training data"<<endl;

	int a;
	cin>>a;

	/* STEP 3. Splitting the samples */
	//1. Select 40 for the training
	CvTrainTestSplit cvtts(a, true);
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

