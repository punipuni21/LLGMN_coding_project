#pragma once

#include <iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>
using namespace std;


//���t�f�[�^
struct teaching_data {

public:
	vector<double> input;//���̓f�[�^��
	vector<double> output;//�o�̓f�[�^��
	/********************************************************/
	
	
	//�l�̏�����
	teaching_data(int input_siz, int output_siz)
	{
		input.resize(input_siz, 0);
		output.resize(output_siz, 0);
	}
};

