#pragma once
#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>
#include<fstream>
#include<iomanip>
#include<direct.h>

#include "macro.h";
#include "LLGMN.h"
#include "utility.h";

//�N���X�̃����o�͖����ɃA���_�[�X�R�A����������

using namespace std;

LLGMN::LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size) {

	lr_ = lr;
	epochs_ = epochs;
	batch_size_ = batch_size;
	input_dim_ = input_dim;
	output_dim_ = class_num;
	component_size_ = component_size;
	data_size_ = data_size;
	non_linear_input_siz_ = 1 + input_dim * (input_dim + 3) / 2;
	weight_ = make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_size_ + 5);//�d�݌W��
	current_time_ = get_current_time("_");
	progress_.push_back(0);//1-index

	input_layer_ = make_v<double>(data_size_ + 10, 0);
	mid_layer_input_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
	mid_layer_output_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
	output_layer_ = make_v<double>(data_size_ + 10, class_num_ + 10);
};


void LLGMN::train(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

	weight_initialize();
	
	//�����Todo
	//for �ŏ��̃f�[�^���F
		//for �o�b�`�T�C�Y���F
			//forward
			//���X�̌v�Z
			//���𗦂Ȃǂ̌v�Z
			//backward

	//�Ƃ肠�����ꊇ�w�K�Ŏ������ė]�T������΃o�b�`�T�C�Y��ύX�ł���悤�ɂ��邱��
	bool flag = true;
	double accuracy = 0;

	for (int i = 0; i < epochs_; i++) {
		//������
		double log_likelihood = 0;
		fill_v(mid_layer_input_, 0);
		fill_v(mid_layer_output_, 0);
		fill_v(output_layer_, 0);


		//forward
		forward(training_data, training_label);
		progress_.push_back(log_likelihood_);

		if (false)
		{
			lr_ = pow(log_likelihood_, 1 - beta) / (epochs_ * (1 - beta));
		}

		cout << "epoch: " << i << " log_likelihood= " << log_likelihood_ << " lr= " << lr_ << endl;

		//backward
		backward(training_data, training_label);

		flag = false;
	}

	//���O�̏o��

	save_loss();

	//�d�݂̕ۑ��Ȃ�

	save_weight();
}




void LLGMN::forward(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

	auto exp_num = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);//�O��������exp�̒l
	fill_v(exp_num, 0);

	//*************************************************************************************************
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		//1-index
		//data_num�Ԗڂ̃f�[�^�ɂ��ď������`�d������
		for (int layer_num = 1; layer_num <= 3; layer_num++)
		{
			//���͑w�C���ԑw�C�o�͑w
			if (layer_num == 1)
			{
				//���͑w�̏���
				//*******************************************************************************************************************
				//����`����				
				input_layer_[data_num].push_back(0);//�ԕ�������1-index�ɂ���
				//0���̍�
				input_layer_[data_num].push_back(1);
				//1���̍�
				for (int index = 1; index <= input_dim_; index++)
				{
					input_layer_[data_num].push_back(training_data[data_num][index]);
				}

				//2���̍�
				for (int outside = 1; outside <= input_dim_; outside++)
				{
					for (int inside = outside; inside <= input_dim_; inside++)
					{
						input_layer_[data_num].push_back(training_data[data_num][outside] * training_data[data_num][inside]);
					}
				}

				//���̎��_�œ��͎����� 1 + input_siz * (input_siz + 3) / 2�ɂȂ��Ă���(�ԕ��𐔂��Ȃ��ꍇ)

				//*********************************************************************************************************************

				for (int tm = 0; tm < 10; tm++)
				{
					input_layer_[data_num].push_back(0);//���ɂ��ԕ��i�������߂ɓ����j
				}
				//***********************************************************************************************************************

				//���ԑw�ɓ`��������

				for (int now_node = 1; now_node <= non_linear_input_siz_; now_node++)//���͑w�̃m�[�h�ԍ�
				{
					for (int class_num = 1; class_num <= class_num_; class_num++)//���ԑw�̃N���X�ԍ�
					{
						for (int component_num = 1; component_num <= component_size_; component_num++)//class_num�Ԗڂ̃N���X�̃R���|�[�l���g�ԍ�
						{

							mid_layer_input_[data_num][class_num][component_num] +=
								input_layer_[data_num][now_node] * weight_[now_node][class_num][component_num];

						}
					}
				}
			}
			else if (layer_num == 2)
			{
				//���ԑw

				//****************************************************************************************************************************
				//exp�̑O�������s��

				double cnt = 0;//����(�S�ẴN���X�C�R���|�[�l���g�̓��͂�exp�̑��a)
				for (int class_num = 1; class_num <= class_num_; class_num++)
				{
					for (int component_num = 1; component_num <= component_size_; component_num++)
					{
						exp_num[data_num][class_num][component_num]
							= exp(mid_layer_input_[data_num][class_num][component_num]);
						cnt += exp_num[data_num][class_num][component_num];
					}
				}
				//****************************************************************************************************************************

				//���ԑw�̓��͂���o�͒l�����߂�

				for (int class_num = 1; class_num <= class_num_; class_num++)//�N���X�ԍ�
				{
					for (int component_num = 1; component_num <= component_size_; component_num++)//class_num�Ԗڂ̃N���X�ԍ��̃R���|�[�l���g�ԍ�
					{
						mid_layer_output_[data_num][class_num][component_num]
							= exp_num[data_num][class_num][component_num] / cnt;
					}
				}
				//***********************************************************************************************************************************
				//�o�͑w�ɓ`�d������

				for (int class_num = 1; class_num <= class_num_; class_num++)//�N���X�ԍ�k�ɑ�������̂��܂Ƃ߂�
				{
					for (int component_num = 1; component_num <= component_size_; component_num++)//class_num�Ԗڂ̃N���X�ԍ��̃R���|�[�l���g�ԍ�
					{
						//���ԑw��class_num�Ԗڂ̃N���X�ɑ�����R���|�[�l���g��S�ďo�͑w��class_num�Ԃ̃N���X�ɂ܂Ƃ߂�
						output_layer_[data_num][class_num] += mid_layer_output_[data_num][class_num][component_num];
					}
				}
			}
			else if (layer_num == 3)
			{
				//�o�͑w

				//�ΐ��ޓx*(-1)���v�Z����(-1)�������邱�Ƃł�����ŏ����ł���Αΐ��ޓx���ő剻���邽��)
				for (int class_num = 1; class_num <= class_num_; class_num++)
				{
					log_likelihood_ += training_label[data_num][class_num] * log(output_layer_[data_num][class_num]) * (-1);
				}
			}
		}
	}

	log_likelihood_ /= data_size_;
}


void LLGMN::backward(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

	double denom = 0.0;
	double gamma = 0.0;


	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		for (int now_node = 1; now_node <= non_linear_input_siz_; now_node++)
		{
			for (int class_num = 1; class_num <= class_num_; class_num++)
			{
				for (int component_num = 1; component_num <= component_size_; component_num++)
				{
					denom += pow((output_layer_[data_num][class_num] - training_label[data_num][class_num])
						* mid_layer_output_[data_num][class_num][component_num] / output_layer_[data_num][class_num] * input_layer_[data_num][now_node], 2);
				}
			}
		}
	}

	if (denom == 0)
	{
		denom += 1e-8;
	}

	//�K���}�����߂邽�߂̕���̑O����


	gamma = pow(log_likelihood_, beta) / denom;


	//�d�݌W���̍X�V
	for (int now_node = 1; now_node <= non_linear_input_siz_; now_node++)
	{
		for (int class_num = 1; class_num <= class_num_; class_num++)
		{
			for (int component_num = 1; component_num <= component_size_; component_num++)
			{
				if (class_num == class_num_ && component_num == component_size_)
				{
					continue;
				}
				double cnt = 0;
				for (int data_num = 1; data_num <= data_size_; data_num++)
				{
					cnt += (output_layer_[data_num][class_num] - training_label[data_num][class_num])
						* mid_layer_output_[data_num][class_num][component_num] / output_layer_[data_num][class_num] * input_layer_[data_num][now_node];
				}
				weight_[now_node][class_num][component_num] -= lr_ * gamma * cnt;

			}
		}
	}

}


void LLGMN::eval(vector<vector<double>>& test_data, vector<vector<double>>& test_label) {

	//forward

	fill_v(mid_layer_input_, 0);
	fill_v(mid_layer_output_, 0);
	fill_v(output_layer_, 0);
	double accuracy = 0;

	//forward
	forward(test_data, test_label);

	save_result(test_data, test_label, output_layer_);

	accuracy = calc_accuracy(test_label, output_layer_);
	//���𗦁C�����s��̎Z�o�Ȃ�

	save_confusion_matrix(test_label, output_layer_);

}


void LLGMN::save_result(vector<vector<double>>& test_data, vector<vector<double>>& test_label, vector<vector<double>>& output_layer) {

	if (is_current_time_exist_) {
		//���Ɍ��ʕۑ��p�f�B���N�g�����쐬����Ă���D
		cout << "���ɍ쐬����Ă��܂��D" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "�f�B���N�g��" << current_time_ << "���쐬���܂���" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "�f�B���N�g��" << current_time_ << "���쐬�ł��܂���ł���" << endl;
		}
	}

	//���X�̕ۑ��ƍ����s��̎Z�o�C�ۑ�
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_result.csv");

	if (ofs.fail())
	{
		cout << "failed " << endl;
		exit(1);
	}

	for (int i = 0; i < input_dim_; i++)
	{
		ofs << "dim" << i << ",";
	}
	ofs << ",";
	for (int i = 0; i < class_num_; i++)
	{
		ofs << "class" << i << ",";
	}
	ofs << ",";
	ofs << "class_label" << endl;

	int success_case = 0;

	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		//���̓f�[�^�i���w�K�j
		for (int input_num = 1; input_num <= input_dim_; input_num++)
		{
			ofs << test_data[data_num][input_num] << ",";
		}
		double maxi = 0;
		int idx = 0;
		//�o�͒l
		ofs << " " << ",";
		for (int class_num = 1; class_num <= class_num_; class_num++)
		{
			ofs << output_layer[data_num][class_num] << ",";
			if (maxi < output_layer[data_num][class_num])
			{
				maxi = output_layer[data_num][class_num];
				idx = class_num;
			}
		}

		//����m���ő�̃N���X��one-hot�\����1�������Ă����ꍇ
		if (test_label[data_num][idx] == 1)
		{
			success_case++;
		}
		//���ʂ����N���X
		ofs << "," << idx << endl;
	}

	ofs << "���ʗ� =," << success_case / (double)data_size_ << endl;
}


double LLGMN::calc_accuracy(vector<vector<double>>& test_label, vector<vector<double>>& output_layer) {

	//���𗦂̎Z�o
	double maxi = 0;
	int idx = 0;
	int success_case = 0;
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		double maxi = 0;
		int idx = 0;
		//�o�͒l
		for (int class_num = 1; class_num <= class_num_; class_num++)
		{
			if (maxi < output_layer_[data_num][class_num])
			{
				maxi = output_layer_[data_num][class_num];
				idx = class_num;
			}
		}
		if (test_label[data_num][idx] == 1)
		{
			success_case++;
		}
	}
	return success_case / (double)data_size_;
}


void LLGMN::save_weight() {

	if (is_current_time_exist_) {
		//���Ɍ��ʕۑ��p�f�B���N�g�����쐬����Ă���D
		cout << "���ɍ쐬����Ă��܂��D" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "�f�B���N�g��" << current_time_ << "���쐬���܂���" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "�f�B���N�g��" << current_time_ << "���쐬�ł��܂���ł���" << endl;
		}
	}
	//�d�݂̕ۑ�
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_weight.csv");

	if (ofs.fail())
	{
		cout << "failed " << endl;
		exit(1);
	}


	//�d�݂̕ۑ�

	//�N���X�̗񖼂����
	ofs << ",,";
	for (int i = 0; i < class_num_; i++)
	{
		ofs << "class " << i + 1 << ",";
		for (int j = 0; j < component_size_; j++)
		{
			ofs << ",";
		}
	}
	ofs << endl;
	//�R���|�[�l���g�̗񖼂����
	ofs << ",,";
	for (int i = 0; i < class_num_; i++)
	{
		for (int j = 0; j < component_size_; j++)
		{
			ofs << "component " << i + 1 << ",";
		}
		ofs << ",";
	}
	ofs << endl;




	for (int i = 0; i < non_linear_input_siz_; i++)
	{
		ofs << "node " << i + 1 << ",";
		for (int j = 0; j < class_num_; j++)
		{
			ofs << ",";
			for (int k = 0; k < component_size_; k++)
			{
				ofs << weight_[i][j][k] << ",";
			}
		}
		ofs << endl;
	}
}


void LLGMN::save_confusion_matrix(vector<vector<double>>& test_label, vector<vector<double>>& output_layer) {

	//�����s��̌v�Z�C�ۑ�

	if (is_current_time_exist_) {
		//���Ɍ��ʕۑ��p�f�B���N�g�����쐬����Ă���D
		cout << "���ɍ쐬����Ă��܂��D" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "�f�B���N�g��" << current_time_ << "���쐬���܂���" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "�f�B���N�g��" << current_time_ << "���쐬�ł��܂���ł���" << endl;
		}
	}

	//���X�̕ۑ��ƍ����s��̎Z�o�C�ۑ�
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_confusion_matrix.csv");

	auto confusion_matrix = make_v<int>(class_num_ + 1, class_num_ + 1);//1-index
	fill_v(confusion_matrix, 0);

	//�����s��̎Z�o
	double maxi = 0;
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		double maxi = 0;
		int estimated_class_id = 0;
		int label_class_id = 0;
		//�o�͒l
		for (int class_num = 1; class_num <= class_num_; class_num++)
		{
			//���肵���N���X�̎Z�o
			if (maxi < output_layer_[data_num][class_num])
			{
				maxi = output_layer_[data_num][class_num];
				estimated_class_id = class_num;
			}
			//one-hot���x�����狳�t���x���̃N���X�����߂�
			if (test_label[data_num][class_num] == 1) {
				label_class_id = class_num;
			}
		}
		confusion_matrix[label_class_id][estimated_class_id]++;
	}

	//�����s��̕ۑ�
	ofs << ",";
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << "estimated class " << i << ",";
	}
	ofs << endl;
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << "true value " << i << ",";
		for (int j = 1; j <= class_num_; j++)
		{
			ofs << confusion_matrix[i][j] << ",";

		}
		ofs << endl;
	}
}


void LLGMN::save_loss() {
	

	if (is_current_time_exist_) {
		//���Ɍ��ʕۑ��p�f�B���N�g�����쐬����Ă���D
		cout << "���ɍ쐬����Ă��܂��D" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "�f�B���N�g��" << current_time_ << "���쐬���܂���" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "�f�B���N�g��" << current_time_ << "���쐬�ł��܂���ł���" << endl;
		}
	}

	//���X�̕ۑ��ƍ����s��̎Z�o�C�ۑ�
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_loss.csv");

	ofs << "epoch" << "," << "loss" << endl;

	if (ofs.fail())
	{
		cout << "failed " << endl;
		exit(1);
	}
	
	for (int data_num = 1; data_num <= epochs_; data_num++) {
		ofs << data_num << "," << progress_[data_num] << endl;
	}
}


void LLGMN::weight_initialize() {

	// 0.0�ȏ�1.0�����̒l�𓙊m���Ŕ���������
	std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�
	std::mt19937 mt(rnd());     //  �����Z���k�E�c�C�X�^��32�r�b�g�ŁA�����͏����V�[�h�l
	std::uniform_real_distribution<> rand12(-1.0, 1.0);//[-1.0,1.0]�̗����𐶐�

	for (int i = 0; i < weight_.size(); i++) {

		for (int j = 0; j < weight_[i].size(); j++)
		{
			for (int k = 0; k < weight_[i][j].size(); k++)
			{
				weight_[i][j][k] = rand12(mt);
			}
		}
	}
}
