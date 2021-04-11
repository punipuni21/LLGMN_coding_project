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


public:


	//�w�K�� lr
	//�G�|�b�N��
	//�o�b�`�T�C�Y
	//�R���|�[�l���g���F
	//���͎�����
	//�o�͎����i�N���X���j
	const double eps = 1e-5;//�ΐ��ޓx�̋��e�l
	const double beta = 0.8;//�萔
	const double sampling_time = 0.001;

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
	//vector<vector<double>> training_data_;//(�o�b�`�T�C�Y�C����...)
	//vector<vector<double>> training_label_;//(�o�b�`�T�C�Y�C����...)

	vector<vector<double>> input_layer_;
	vector<vector<vector<double>>> mid_layer_input_;
	vector<vector<vector<double>>> mid_layer_output_;
	vector<vector<double>> output_layer_;

	//auto input_layer = make_v<double>(data_siz + 10, 0);//����`�����������͑w�̓��͎���
	//auto mid_layer_input = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//�e�w�̓��͒l
	//auto mid_layer_output = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//�e�w�̏o�͒l
	//auto output_layer = make_v<double>(data_siz + 10, class_siz + 10);//�o�͑w�̊e�N���X�̏o�͒l�i����m���j

	//���t�f�[�^�F(�o�b�`�T�C�Y�C����...)
	//�������x���F(�o�b�`�T�C�Y�C����...)���܂���one-hot�Ŏ������邱��


	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size) {

		lr_ = lr;
		epochs_ = epochs;
		batch_size_ = batch_size;
		input_dim_ = input_dim;
		output_dim_ = class_num;
		component_size_ = component_size;
		data_size_ = data_size;
		non_linear_input_siz_ = 1 + input_dim * (input_dim + 3) / 2;
		weight_ = make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_size_ + 5);//�d�݌W��
		//training_data_ = make_v<double>(data_size_, input_dim_);
		//training_label_ = make_v<double>(data_size_, class_num_);

		input_layer_ = make_v<double>(data_size_ + 10, 0);
		mid_layer_input_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
		mid_layer_output_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
		output_layer_ = make_v<double>(data_size_ + 10, class_num_ + 10);

		cout << epochs << " " << batch_size_ << " " << input_dim_ << " " << output_dim_ << " " << class_num << " " << component_size_ << endl;
		cout << non_linear_input_siz_ << endl;
	}


	void train(vector<vector<double>> & training_data, vector<vector<double>> & training_label) {

		weight_initialize();
		//�f�[�^�͐���ɓn����Ă���
		
		//�����Todo
		//for �ŏ��̃f�[�^���F
			//for �o�b�`�T�C�Y���F
				//forward
				//���X�̌v�Z
				//���𗦂Ȃǂ̌v�Z
				//backward

		//�Ƃ肠�����ꊇ�w�K�Ŏ������ė]�T������΃o�b�`�T�C�Y��ύX�ł���悤�ɂ��邱��
		bool flag = true;

		for (int i = 0; i < epochs_; i++) {
			//������
			double log_likelihood = 0;
			fill_v(mid_layer_input_, 0);
			fill_v(mid_layer_output_, 0);
			fill_v(output_layer_, 0);


			//forward

			forward(training_data, training_label, flag);

			cout << "epoch: " << i << " log_likelihood= " << log_likelihood_ << " lr= " << lr_ << endl;
			//backward

			backward(training_data, training_label);




			flag = false;
		}

		//���O�̏o��


		//�d�݂̕ۑ��Ȃ�
	}


	void forward(vector<vector<double>> & training_data, vector<vector<double>> & training_label, bool flag) {

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
		progress_.push_back(log_likelihood_);
		if (true)
		{
			lr_ = pow(log_likelihood_, 1 - beta) / (epochs_ * (1 - beta));
			//cout << lr_ << " " << log_likelihood_ << endl;
		}

	}

	void backward(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

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


	void eval(vector<vector<double>> & training_data, vector<vector<double>> & training_label) {

		//forward

		//���𗦁C�����s��̎Z�o�Ȃ�

	}


	//�d�݂̏�����
	void weight_initialize() {

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
};