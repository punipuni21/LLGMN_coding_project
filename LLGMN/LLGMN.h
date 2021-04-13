#pragma once
#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>

#include "macro.h";

//�N���X�̃����o�͖����ɃA���_�[�X�R�A����������

using namespace std;

class LLGMN {

private:

	const double eps = 1e-5;//�ΐ��ޓx�̋��e�l
	const double beta = 0.2;//�萔

	double lr_ = 0.01;
	int epochs_ = 5;
	int batch_size_ = 1;
	int input_dim_ = 2;
	int output_dim_ = 0;
	int class_num_ = 4;
	int component_size_ = 2;
	int non_linear_input_siz_ = 1;
	int data_size_ = 10;
	double log_likelihood_ = 0;

	vector<vector<vector<double>>> weight_;
	vector<double> progress_;//log

	vector<vector<double>> input_layer_;
	vector<vector<vector<double>>> mid_layer_input_;
	vector<vector<vector<double>>> mid_layer_output_;
	vector<vector<double>> output_layer_;

public:

	//���t�f�[�^�F(�o�b�`�T�C�Y�C����...)
	//�������x���F(�o�b�`�T�C�Y�C����...)���܂���one-hot�Ŏ������邱��

	//�R���X�g���N�^
	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size);

	//�w�K���񂷃��\�b�h
	void train(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//�t�����v�Z
	void forward(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//�덷�t�`�d
	void backward(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//�e�X�g�f�[�^�ł̐��_
	void eval(vector<vector<double>>& test_data, vector<vector<double>>& test_label);

	//�o�͑w�̌��ʂ̕ۑ�
	void save_result(vector<vector<double>>& test_data, vector<vector<double>>& test_label, vector<vector<double>>& output_layer);

	//���𗦂̌v�Z
	double calc_accuracy(vector<vector<double>>& test_label, vector<vector<double>>& output_layer);

	//�d�݂̕ۑ�
	void save_weight();

	//�����s��̎Z�o
	void save_confusion_matrix(vector<vector<double>>& test_label, vector<vector<double>>& output_layer);

	//���O�̕ۑ�
	void save_log();

	//�d�݂̏�����
	void weight_initialize();

};