#include<iostream>
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

const double eps = 1e-5;//�ΐ��ޓx�̋��e�l
const int rep = 1000;//���s�񐔂̏��
const double beta = 0.8;//�萔
const double sampling_time = 0.001;

//******************************************************************************************************************

//�v���g�^�C�v�錾

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
	bool flag,//�����l���X�V���邩�̃t���O
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




//�d�݌W���̏�����
void value_initialize(vector<vector<vector<double> > >& weight);


//weight[i][j][k]:=i�Ԗڂ̔���`�������������̓m�[�h�Ƒ�2�w�̃N���Xj�̃R���|�[�l���gk�̃m�[�h�̊Ԃ̏d�݌W��
//mid_layer_input[i][j][k]:=i�Ԗڂ̃f�[�^�̃N���Xj�̃R���|�[�l���gk�̓���
//mid_layer_output[i][j][k]:=i�Ԗڂ̃f�[�^�̃N���Xj�̃R���|�[�l���gk�̏o��


//*********************************************************************************************************************


//�w�K

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
	vector<double>& progress//�ΐ��ޓx�̉ߒ�
) {

	//*****************************************************************************
	//�ϐ�
	int cnt = 0;//�w�K��
	double log_likelihood = 0.0;//�ΐ��ޓx
	bool flag = false;//�����l���X�V���邩�̃t���O(�ŏ��̊w�K�̎��̂�true)
	double study_rate = 0;
	
	//*****************************************************************************
	//�d�݌W���̏�����

	value_initialize(weight);


	//�Ō�̃N���X�̍Ō�̃R���|�[�l���g�ւ̏d�݂�0�ɂ���

	for (int node = 1; node <= non_linear_input_siz; node++)
	{
		weight[node][class_siz][each_class_siz[class_siz]] = 0;
	}
	//*****************************************************************************

	auto input_layer = make_v<double>(data_siz + 10, 0);//����`�����������͑w�̓��͎���
	auto mid_layer_input = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//�e�w�̓��͒l
	auto mid_layer_output = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//�e�w�̏o�͒l
	auto output_layer = make_v<double>(data_siz + 10, class_siz + 10);//�o�͑w�̊e�N���X�̏o�͒l�i����m���j
	auto before = make_v<double>(non_linear_input_siz + 10, class_siz + 5, maxi_component_siz + 5);//�O��̒l

	do {

		
		//������
		log_likelihood = 0;
		fill_v(mid_layer_input, 0);
		fill_v(mid_layer_output, 0);
		fill_v(output_layer, 0);
		fill_v(before, 0);
		

		//�������`���i�ΐ��ޓx�͎Q�Ɠn���j
		if (cnt == 0)
		{
			//����̂�true
			flag = true;
		}
		else
		{
			flag = false;
		}
		forward(T_data, weight, input_layer, output_layer, mid_layer_input, mid_layer_output,
			each_class_siz, class_siz, log_likelihood, non_linear_input_siz, data_siz, maxi_component_siz, input_siz, flag, study_rate);
		
		
		//	�t�����`�d

		backward(T_data, weight, input_layer, output_layer, mid_layer_output,
			each_class_siz, class_siz, non_linear_input_siz, data_siz, study_rate, log_likelihood, before);


		//�w�K�񐔂̍X�V��Y��Ȃ�
			   
		cnt++;

		cout << "cnt = " << cnt << " log_likelihood = " << log_likelihood << endl;
		progress.push_back(log_likelihood);

	} while (abs(log_likelihood) > eps && cnt < rep);
}


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
	bool flag,//�����l���X�V���邩�̃t���O
	double& study_rate//�w�K�W��
) {
	//����

	//**********************************************************************************************
	//�ϐ�

	auto exp_num = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//�O��������exp�̒l
	fill_v(exp_num, 0);

	//*************************************************************************************************
	for (int data_num = 1; data_num <= data_siz; data_num++)
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
				input_layer[data_num].push_back(0);//�ԕ�������1-index�ɂ���
				//0���̍�
				input_layer[data_num].push_back(1);
				//1���̍�
				for (int index = 1; index <= input_siz; index++)
				{
					input_layer[data_num].push_back(T_data[data_num].input[index]);
				}

				//2���̍�
				for (int outside = 1; outside <= input_siz; outside++)
				{
					for (int inside = outside; inside <= input_siz; inside++)
					{
						input_layer[data_num].push_back(T_data[data_num].input[outside] * T_data[data_num].input[inside]);
					}
				}

				//���̎��_�œ��͎�����1+input_siz * (input_siz + 3) / 2�ɂȂ��Ă���(�ԕ��𐔂��Ȃ��ꍇ)

				//*********************************************************************************************************************

				for (int tm = 0; tm < 10; tm++)
				{
					input_layer[data_num].push_back(0);//���ɂ��ԕ��i�������߂ɓ����j
				}
				//***********************************************************************************************************************

				//���ԑw�ɓ`��������

				for (int now_node = 1; now_node <= non_linear_input_siz; now_node++)//���͑w�̃m�[�h�ԍ�
				{
					for (int class_num = 1; class_num <= class_siz; class_num++)//���ԑw�̃N���X�ԍ�
					{
						for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num�Ԗڂ̃N���X�̃R���|�[�l���g�ԍ�
						{

							mid_layer_input[data_num][class_num][component_num] +=
								input_layer[data_num][now_node] * weight[now_node][class_num][component_num];

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
				for (int class_num = 1; class_num <= class_siz; class_num++)
				{
					for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)
					{
						exp_num[data_num][class_num][component_num]
							= exp(mid_layer_input[data_num][class_num][component_num]);
						cnt += exp_num[data_num][class_num][component_num];
					}
				}
				//****************************************************************************************************************************

				//���ԑw�̓��͂���o�͒l�����߂�

				for (int class_num = 1; class_num <= class_siz; class_num++)//�N���X�ԍ�
				{
					for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num�Ԗڂ̃N���X�ԍ��̃R���|�[�l���g�ԍ�
					{
						mid_layer_output[data_num][class_num][component_num]
							= exp_num[data_num][class_num][component_num] / cnt;
					}
				}
				//***********************************************************************************************************************************
				//�o�͑w�ɓ`�d������

				for (int class_num = 1; class_num <= class_siz; class_num++)//�N���X�ԍ�k�ɑ�������̂��܂Ƃ߂�
				{
					for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num�Ԗڂ̃N���X�ԍ��̃R���|�[�l���g�ԍ�
					{
						//���ԑw��class_num�Ԗڂ̃N���X�ɑ�����R���|�[�l���g��S�ďo�͑w��class_num�Ԃ̃N���X�ɂ܂Ƃ߂�
						output_layer[data_num][class_num] += mid_layer_output[data_num][class_num][component_num];
					}
				}
			}
			else if (layer_num == 3)
			{
				//�o�͑w

				//�ΐ��ޓx*(-1)���v�Z����(-1)�������邱�Ƃł�����ŏ����ł���Αΐ��ޓx���ő剻���邽��)
				for (int class_num = 1; class_num <= class_siz; class_num++)
				{
					log_likelihood += T_data[data_num].output[class_num] * log(output_layer[data_num][class_num]) * (-1);
				}
			}
		}
	}

	if (flag)
	{
		study_rate = pow(log_likelihood, 1 - beta) / (rep * (1 - beta));
		cout << study_rate << " " << log_likelihood << endl;
	}
}
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
) {

	double denom = 0.0;
	double gamma = 0.0;
		

	for (int data_num = 1; data_num <= data_siz; data_num++)
	{
		for (int now_node = 1; now_node <= non_linear_input_siz; now_node++)
		{
			for (int class_num = 1; class_num <= class_siz; class_num++)
			{
				for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)
				{
					denom += pow((output_layer[data_num][class_num] - T_data[data_num].output[class_num])
						* mid_layer_output[data_num][class_num][component_num] / output_layer[data_num][class_num] * input_layer[data_num][now_node], 2);
				}
			}
		}
	}

	if (denom == 0)
	{
		denom += 1e-8;
	}

	//�K���}�����߂邽�߂̕���̑O����


	gamma = pow(log_likelihood, beta) / denom;


	//�d�݌W���̍X�V
	for (int now_node = 1; now_node <= non_linear_input_siz; now_node++)
	{
		for (int class_num = 1; class_num <= class_siz; class_num++)
		{
			for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)
			{
				if (class_num == class_siz && component_num == each_class_siz[class_siz])
				{
					continue;
				}
				double cnt = 0;
				for (int data_num = 1; data_num <= data_siz; data_num++)
				{
					cnt += (output_layer[data_num][class_num] - T_data[data_num].output[class_num])
						* mid_layer_output[data_num][class_num][component_num] / output_layer[data_num][class_num] * input_layer[data_num][now_node];
				}
				weight[now_node][class_num][component_num] -= study_rate * gamma * cnt;

			}
		}
	}
}





//�d�݌W�������肷��֐�
void value_initialize(vector<vector<vector<double> > >& weight) {

	// 0.0�ȏ�1.0�����̒l�𓙊m���Ŕ���������
	std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�
	std::mt19937 mt(rnd());     //  �����Z���k�E�c�C�X�^��32�r�b�g�ŁA�����͏����V�[�h�l
	std::uniform_real_distribution<> rand12(-1.0, 1.0);//[-1.0,1.0]�̗����𐶐�

	for (int i = 0; i < weight.size(); i++) {

		for (int j = 0; j < weight[i].size(); j++)
		{
			for (int k = 0; k < weight[i][j].size(); k++)
			{
				weight[i][j][k] = rand12(mt);
			}
		}
	}
}
