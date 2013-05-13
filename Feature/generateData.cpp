#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

vector< vector<float> > readText(char* fileName);
vector< vector<char> > readTruth(char* fileName);
vector< vector<int> > readTruthLogistic(char* fileName);

int maxClusterSize, maxFrameIndex;
// generate Training data and test data
vector< vector<float> > readText(char* fileName) {

	ifstream infile(fileName);

	int clusterNumber, frameIndex;

	maxClusterSize = 0;
	frameIndex = 0;


	vector< vector<float> > v;
	float data;
	while( !infile.eof() && infile>>clusterNumber>>frameIndex>>data){
		if(maxClusterSize < clusterNumber + 1){
			v.resize(clusterNumber + 1);
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(maxFrameIndex + 1, -1);
			}	
			maxClusterSize = clusterNumber + 1;
		}

		if(maxFrameIndex < frameIndex + 1){
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(frameIndex + 1, -1);
			}	
			maxFrameIndex = frameIndex + 1;
		}
		v[clusterNumber][frameIndex] = data;
	}
	return v;
}

vector< vector<char> > readTruth(char* fileName) {
	ifstream infile(fileName);

	int clusterNumber, frameIndex;

	maxClusterSize = 0;
	frameIndex = 0;

	vector< vector<char> > v;
	char data;
	while( !infile.eof() && infile>>clusterNumber>>frameIndex>>data){
		if(maxClusterSize < clusterNumber + 1){
			v.resize(clusterNumber + 1);
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(maxFrameIndex + 1, 'b');
			}	
			maxClusterSize = clusterNumber + 1;
		}

		if(maxFrameIndex < frameIndex + 1){
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(frameIndex + 1, 'b');
			}	
			maxFrameIndex = frameIndex + 1;
		}
		v[clusterNumber][frameIndex] = data;
	}
	return v;
}

vector< vector<int> > readTruthLogistic(char* fileName) {
	ifstream infile(fileName);

	int clusterNumber, frameIndex;

	maxClusterSize = 0;
	frameIndex = 0;


	vector< vector<int> > v;
	char data;
	while( !infile.eof() && infile>>clusterNumber>>frameIndex>>data){
		if(maxClusterSize < clusterNumber + 1){
			v.resize(clusterNumber + 1);
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(maxFrameIndex + 1, -1);
			}	
			maxClusterSize = clusterNumber + 1;
		}

		if(maxFrameIndex < frameIndex + 1){
			for (int i = 0; i < v.size(); ++i){
				v[i].resize(frameIndex + 1, -1);
			}	
			maxFrameIndex = frameIndex + 1;
		}
		v[clusterNumber][frameIndex] = 1;
	}
	return v;
}
void generateData(){

	ofstream out_boost;
	ofstream out_boost_test;
	ofstream out_logistic;
	ofstream out_logistic_test;

	out_boost.open("training_boost.txt");
	out_boost_test.open("testing_boost.txt");
	out_logistic.open("training_logistic.txt");	
	out_logistic_test.open("testing_logistic.txt");	

	maxClusterSize = 0;
	maxFrameIndex = 0;
	vector< vector<float> > loca_diff = readText("loca_diff.txt");
	vector< vector<float> > mouth_diff = readText("mouth_diff.txt");
	vector< vector<float> > mouth_pix = readText("mouth_pix.txt");
	vector< vector<float> > pose_diff = readText("pose_diff.txt");
	vector< vector<float> > face_pix_diff = readText("face_pix_diff.txt");
	vector< vector<float> > face_pix = readText("face_pix.txt");

	vector< vector<float> > distance = readText("distance.txt");
	vector< vector<float> > normalized_distance = readText("normalized_distance.txt");

	vector< vector<char> > truth = readTruth("truth.txt");
	vector< vector<int> > truth_int = readTruthLogistic("truth.txt");

    for( int i = 0; i < face_pix_diff.size(); i ++ ) {
    	int j = 0;
        for( j; j < face_pix_diff[i].size()* 4 / 5; j ++ ) {
            if(pose_diff[i][j] != -1 ) {

            	/* boost part */
            	// if one of the following data is missing
            	out_boost<<truth[i][j]<<','<<loca_diff[i][j]<<','<<mouth_diff[i][j]<<',';
            	out_boost<<mouth_pix[i][j]<<','<<pose_diff[i][j]<<','<<face_pix_diff[i][j]<<','<<face_pix[i][j]<<',';
            	out_boost<<distance[i][j]<<','<<normalized_distance[i][j];

            	out_boost<<endl;

            	int idx = 0;
            	/* logistic regression part */
            	out_logistic<<truth_int[i][j]<<' ';
            	out_logistic<<++idx<<':'<<loca_diff[i][j]<<' ';
            	out_logistic<<++idx<<':'<<mouth_diff[i][j]<<' ';
            	out_logistic<<++idx<<':'<<mouth_pix[i][j]<<' ';
            	out_logistic<<++idx<<':'<<pose_diff[i][j]<<' ';
            	out_logistic<<++idx<<':'<<face_pix_diff[i][j]<<' ';
            	out_logistic<<++idx<<':'<<face_pix[i][j]<<' ';
            	out_logistic<<++idx<<':'<<distance[i][j]<<' ';
            	out_logistic<<++idx<<':'<<normalized_distance[i][j]<<' ';
            	out_logistic<<endl;
            }
        }

        // testing
        for(j; j < face_pix_diff[i].size(); j++){
            	/* boost testing  part */
            	// if one of the following data is missing
            	out_boost_test<<truth[i][j]<<','<<loca_diff[i][j]<<','<<mouth_diff[i][j]<<',';
            	out_boost_test<<mouth_pix[i][j]<<','<<pose_diff[i][j]<<','<<face_pix_diff[i][j]<<','<<face_pix[i][j]<<',';
            	out_boost_test<<distance[i][j]<<','<<normalized_distance[i][j];

            	out_boost_test<<endl;

            	int idx = 0;
            	/* logistic regression part */
            	out_logistic_test<<truth_int[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<loca_diff[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<mouth_diff[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<mouth_pix[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<pose_diff[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<face_pix_diff[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<face_pix[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<distance[i][j]<<' ';
            	out_logistic_test<<++idx<<':'<<normalized_distance[i][j]<<' ';
            	out_logistic_test<<endl;

        }

    }

    out_boost.close();
    out_logistic_test.close();
    out_logistic.close();
    out_logistic_test.close();
}

int main(int argc, char const *argv[])
{
	generateData();
	return 0;
}
