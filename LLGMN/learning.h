#pragma once
#include <iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<sstream>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<random>
#include<string>

#include"in_out.h";
#include"macro.h";
#include"teaching_data.h";
#include"learning.h";
using namespace std;

//�z��͑S��1-index�ň�������


//�w�K
void batch_learning(
	vector<vector<vector<double> > >& weight,//�d�݌W��
	vector<teaching_data>& T_data,//���t�f�[�^
	vector<int>& each_class_siz,//�e�N���X�̃R���|�[�l���g��
	int class_siz,//�N���X��
	int input_siz,//���͎���
	int data_siz,//�f�[�^��
	int maxi_component_siz,//�ő�R���|�[�l���g��
	//double study_rate,//�w�K�W��
	int non_linear_input_siz,//����`�����������͎���
	vector<double>& progress
);



//�������`��

void forward(

	vector<teaching_data>& T_data,//���t�f�[�^
	vector<vector<vector<double> > >& weight,//�d�݌W��
	vector<vector<double> >& input_layer,//���͑w
	vector<vector<double> >& output_layer,//�o�͑w
	vector<vector<vector<double> > >& mid_layer_input,//�e�w�̓��͒l
	vector<vector<vector<double> > >& mid_layer_output,//�e�w�̏o�͒l
	vector<int>& each_class_siz,//�e�N���X�̃R���|�[�l���g��
	int class_siz,//�N���X��
	double& log_likelihood,//�ΐ��ޓx
	int non_linear_input_siz,//����`�����������͎�����
	int data_siz,//�f�[�^��
	int maxi_component_siz,//�ő�R���|�[�l���g��
	int input_siz,//���͎���
	bool flag,//�����l���X�V���邩�ǂ����̃t���O
	double& study_rate//�w�K�W��
);


//�t�����`�d

void backward(

	vector<teaching_data>& T_data,//���t�f�[�^
	vector<vector<vector<double> > >& weight,//�d�݌W��
	vector<vector<double> >& input_layer,//���͑w
	vector<vector<double> >& output_layer,//�o�͑w
	vector<vector<vector<double> > >& mid_layer_output,//�e�w�̏o�͒l
	vector<int>& each_class_siz,//�e�N���X�̃R���|�[�l���g��
	int class_siz,//�N���X��
	int non_linear_input_siz,//����`�����������͎�����
	int data_siz,//�f�[�^��
	double study_rate,//�w�K�W��
	double log_likelihood,//�ΐ��ޓx
	vector<vector<vector<double> > >& before//�O��̒l
);

