#pragma once

#include <iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>
using namespace std;


//教師データ
struct teaching_data {

public:
	vector<double> input;//入力データ数
	vector<double> output;//出力データ数
	/********************************************************/
	
	
	//値の初期化
	teaching_data(int input_siz, int output_siz)
	{
		input.resize(input_siz, 0);
		output.resize(output_siz, 0);
	}
};

