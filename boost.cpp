#include "boost.hpp"

using namespace std;
using namespace okapi;

#define TRUTH 2

Boost::Boost(char* model){
	boost.load(model, "boost");
}

Boost::~Boost(){

}

int Boost::getFocus(vector<Face> curFrame, int pre_focus){
	// load the file for individual features

	// a file writer to write detected output
	ofstream out;
    out.open("features/boost_detected_output.txt", fstream::in | fstream::out | fstream::app);

	int frameIndex = curFrame[0].frameIndex; 
	if(frameIndex < 0) // error
		return -1;

	// feature loca_diff and pose_diff, read from files
	vector< vector<float> > loca_diff = readText("features/loca_diff.txt");
	vector< vector<float> > mouth_diff = readText("features/mouth_diff.txt");
	vector< vector<float> > mouth_pix = readText("features/mouth_pix.txt");
	vector< vector<float> > pose_diff = readText("features/pose_diff.txt");
	vector< vector<float> > face_pix_diff = readText("features/face_pix_diff.txt");
	vector< vector<float> > face_pix = readText("features/face_pix.txt");
	vector< vector<float> > distance = readText("features/distance.txt");
	vector< vector<float> > normalized_distance = readText("features/normalized_distance.txt");

	// protection from dummy
	if(face_pix_diff.size() != curFrame.size()){
		return -1;	
	}
/*
	cout<<loca_diff.size()<<endl;
	cout<<mouth_diff.size()<<endl;
	cout<<mouth_pix.size()<<endl;
	cout<<pose_diff.size()<<endl;
	cout<<face_pix_diff.size()<<endl;
	cout<<face_pix.size()<<endl;
	cout<<distance.size()<<endl;
	cout<<normalized_distance.size()<<endl;
	*/

	cv::Mat	featureMat;
	vector<float> predictions;
	predictions.resize(curFrame.size(),-1);

	int truth_count = 0;
	for (int i = 0; i < curFrame.size(); ++i)
	{
		featureMat = (cv::Mat_<float>(1,9) <<  0,
											  loca_diff[i][frameIndex],
										      mouth_diff[i][frameIndex], 
										      mouth_pix[i][frameIndex], 
										      pose_diff[i][frameIndex], 
										      face_pix_diff[i][frameIndex], 
										      face_pix[i][frameIndex], 
										      distance[i][frameIndex], 
										      normalized_distance[i][frameIndex]);

		predictions[i] = predict(featureMat);
		cout<<"get one prediction"<<endl;
		if(predictions[i] == TRUTH)
			truth_count ++;
	}


	// Mode 2: Based on speaker
	if(truth_count == 1){
		for (int i = 0; i < predictions.size(); ++i){
			if(predictions[i] == TRUTH){
				out<<i<<" "<<frameIndex<<" "<<'a'<<endl;
				out.close();
				return i;
			}
		}
	}
	else if(truth_count == 0){
		return pre_focus;
		cout<<"previous"<<endl;
	}
	else if(truth_count > 1){
		if(predictions[pre_focus] == TRUTH){
			out<<pre_focus<<" "<<frameIndex<<" "<<'a'<<endl;
			out.close();
			return pre_focus;
			cout<<"previous"<<endl;
		}
		else {
			vector<int> possibles;
			for (int i = 0; i < predictions.size(); ++i)
			{
				if(predictions[i] == TRUTH){
					possibles.push_back(i);
				}
			}

			int ranN = rand() % possibles.size();
			int toBeReturn = possibles[ranN];
			out<<possibles[ranN]<<' '<<frameIndex<<' '<<'a'<<endl;
			out.close();
			return possibles[ranN];
		}
	}

}

void Boost::getErrorRate(){
	cout<<'h'<<endl;
	vector<vector<char> > grand_truth = readTruth("features/truth.txt");
	vector<vector<char> > detection_output = readTruth("features/boost_detected_output.txt");

	float correct_count = 0;
	int frameMax = detection_output[0].size();

	for (int i = 0; i < detection_output.size(); ++i)
	{
		for(int j = 0; j< detection_output[i].size(); j++)
		{
			if(grand_truth[i][j] == detection_output[i][j]){
				correct_count ++;
			}
		}
	}

	// only 3 person in the code -> I hard coded this
	float accu = correct_count / (3 * frameMax);

	cout<<"Accuracy is: "<< accu<<endl;
	return;
}


bool Boost::isSpeaker(Face f, int i){
	vector< vector<float> > loca_diff = readText("features/loca_diff.txt");
	vector< vector<float> > mouth_diff = readText("features/mouth_diff.txt");
	vector< vector<float> > mouth_pix = readText("features/mouth_pix.txt");
	vector< vector<float> > pose_diff = readText("features/pose_diff.txt");
	vector< vector<float> > face_pix_diff = readText("features/face_pix_diff.txt");
	vector< vector<float> > face_pix = readText("features/face_pix.txt");
	vector< vector<float> > distance = readText("features/distance.txt");
	vector< vector<float> > normalized_distance = readText("features/normalized_distance.txt");

	int frameIndex = f.frameIndex;

	if(f.isDummy()){
		return false;
	}

	if(face_pix_diff.size() <= i) {
		return false;	
	}else if (face_pix_diff[i].size() <= f.frameIndex){
		return false;
	}
/*
	cout<<loca_diff.size()<<endl;
	cout<<mouth_diff.size()<<endl;
	cout<<mouth_pix.size()<<endl;
	cout<<pose_diff.size()<<endl;
	cout<<face_pix_diff.size()<<endl;
	cout<<face_pix.size()<<endl;
	cout<<distance.size()<<endl;
	cout<<normalized_distance.size()<<endl;

									  cout<<loca_diff[i][frameIndex]<<endl;
								      cout<<mouth_diff[i][frameIndex]<<endl;
								      cout<<mouth_pix[i][frameIndex]<<endl; 
								      cout<<pose_diff[i][frameIndex]<<endl; 
								      cout<<face_pix_diff[i][frameIndex]<<endl; 
								      cout<<face_pix[i][frameIndex]<<endl; 
								      cout<<distance[i][frameIndex]<<endl; 
								      cout<<normalized_distance[i][frameIndex]<<endl;
								      */

	cv::Mat	featureMat;

	cout<<'a'<<endl;

	int truth_count = 0;
	featureMat = (cv::Mat_<float>(1,9) <<  0,
										  loca_diff[i][frameIndex],
									      mouth_diff[i][frameIndex], 
									      mouth_pix[i][frameIndex], 
									      pose_diff[i][frameIndex], 
									      face_pix_diff[i][frameIndex], 
									      face_pix[i][frameIndex], 
									      distance[i][frameIndex], 
									      normalized_distance[i][frameIndex]);

	bool a = predict(featureMat) == TRUTH;
	cout<<"output:"<<a<<endl;
	return a;
}

float Boost::predict(cv::Mat features){

	cv::Mat	missing_data = (cv::Mat_<float>(1,9) << 0,0,0,0,0,0,0,0,0);
	float prediction = boost.predict(features, missing_data); 

	return prediction;
}

vector<vector<float> > Boost::readText(char* fileName){
	ifstream infile(fileName);

	int clusterNumber, frameIndex;
	int maxFrameIndex = 0;

	int maxClusterSize = 0;

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

vector< vector<char> > Boost::readTruth(char* fileName) {
	ifstream infile(fileName);

	int clusterNumber, frameIndex;

	int maxClusterSize = 0;

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
