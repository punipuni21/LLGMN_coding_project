#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>

#include "macro.h";

//�N���X�̃����o�͖����ɃA���_�[�X�R�A����������

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
	int component_siz_ = 2;
	int non_linear_input_siz_ = 1;

	int non_linear_input_siz_ = 1;//����`�����������͂̎���
	vector<vector<vector<double>>> weight_;

	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_siz) {

		lr_ = lr;
		epochs_ = epochs;
		batch_size_ = batch_size;
		input_dim_ = input_dim;
		output_dim_ = class_num;
		component_siz_ = component_siz;
		non_linear_input_siz_ = 1 + input_dim * (input_dim + 3) / 2;
		make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_siz_ + 5);//�d�݌W��

	}



	//forward�i���t�f�[�^�C�������x���j



	//backward



	//�d�݂̏�����
	void value_initialize() {

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