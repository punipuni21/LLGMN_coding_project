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

#include "in_out.h";
#include "macro.h";
#include "teaching_data.h";
#include "learning.h"; 
#include "LLGMN.h";
using namespace std;

int main() {

	//***************************************************************************************************************************
	//入力データの次元数，出力データの次元数，学習率，データ数


	//***********************************************************************************************************
	//変数
	int data_siz = 0;//教師データの総データ数
	int input_siz = 0;//入力データの次元
	int output_siz = 0;//出力データの次元
	double study_rate = 0;//学習率
	int class_siz = 0;//クラス数
	int maxi_component_siz = 0;//コンポーネントサイズの最大
	vector<int> each_class_siz;//教師データの各クラスのコンポーネント数
	each_class_siz.push_back(0);//1-indexにするために番兵を入れておく
	vector<double> progress;//尤度の経過
	double cognitive_rate = 0;
	int success_case = 0;

	
	//***********************************************************************************************************
	

	string name;
	//printf("入力データ数nを入力してください,なおここでの入力データ数は,X = [x1,x2,....xn]におけるnである: ");

	//input(name);
	//input_siz = (int)string_to_double(name);//入力データ数(今回は入力は2個)
	input_siz = 2;

	//printf("出力データ数mを入力してください,なおここでの出力データ数は,Y = [y1,y2,....ym]におけるmである: ");
	//input(name);
	//output_siz = (int)string_to_double(name);//出力データ数(今回は入力は4個)
	output_siz = 4;


	//printf("学習率を入力してください: ");
	//input(name);
	//study_rate = string_to_double(name);//学習率の設定
	study_rate = 0.01;

	//printf("クラス数を入力してください: ");
	//input(name);
	//class_siz = (int)string_to_double(name);//クラス数数の設定(今回は4個に固定(各クラスについて200個の教師データ))
	class_siz = 4;

	for (int class_num = 1; class_num <= class_siz; class_num++)
	{
		//printf("クラス %dのコンポーネント数: ", class_num);
		//input(name);
		//each_class_siz.push_back((int)string_to_double(name));
		//maxi_component_siz = max(maxi_component_siz, (int)string_to_double(name));
		each_class_siz.push_back(2);
		maxi_component_siz = max(maxi_component_siz, 2);

		//printf("クラス %dのデータ数: ", class_num);
		//input(name);
		//data_siz += (int)string_to_double(name);//学習率の設定
		data_siz += 200;//学習率の設定
	}
	
	auto training_data = make_v<double>(data_siz + 5, input_siz + 1);
	auto training_label = make_v<double>(data_siz + 5, class_siz + 1);
	auto test_data = make_v<double>(data_siz + 5, input_siz + 1);
	auto test_label = make_v<double>(data_siz + 5, class_siz + 1);

	//***************************************************************************************************************************************
	//教師データと未学習データのファイルを先に開いておく


	//vector<teaching_data> T_data(data_siz + 10, teaching_data(input_siz + 10, output_siz + 10));//教師データ配列を用意(念のため大きめに用意)
	//vector<teaching_data> NT_data(data_siz + 10, teaching_data(input_siz + 10, output_siz + 10));//未学習データ配列を用意(念のため大きめに用意)
	
	//*************************************************
	//ファイルを開く
	//printf("教師データファイル名(入力): ");
	//select_choice(name);

	ifstream ifs_T_input("lea_sig.csv");
	if (ifs_T_input.fail())
	{
		printf("failed\n");
		exit(1);
	}

	//printf("教師データファイル名(出力) ");
	//select_choice(name);

	ifstream ifs_T_output("lea_T_sig.csv");
	if (ifs_T_output.fail())
	{
		printf("failed\n");
		exit(1);
	}

	//未学習データ
	//printf("未学習データファイル名(入力): ");
	//select_choice(name);

	ifstream ifs_NT_input("dis_sig.csv");
	if (ifs_NT_input.fail())
	{
		printf("failed\n");
		exit(1);
	}

	//printf("未学習データファイル名(出力): ");
	//select_choice(name);

	ifstream ifs_NT_output("dis_T_sig.csv");
	if (ifs_NT_output.fail())
	{
		printf("failed\n");
		exit(1);
	}
	printf("parameter setting Done\n");

	//************************************************************************************

	//教師データを読み込む
	//教師データ(入力)
	string str;
	int index = 1;

	while (getline(ifs_T_input, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		//T_data[index].input[0] = 0;//0番目は番兵
		training_data[index][0] = 0;
		cnt++;

		// 区切り文字がなくなるまで文字を区切っていく
		while (getline(stream, tmp, ','))
		{
			//T_data[index].input[cnt] = stod(tmp);
			training_data[index][cnt] = stod(tmp);
			cnt++;
		}
		index++;
	}

	//教師データ(出力)
	index = 1;

	while (getline(ifs_T_output, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		//T_data[index].output[0] = 0;//0番目は番兵
		training_label[index][0] = 0;
		cnt++;

		// 区切り文字がなくなるまで文字を区切っていく
		while (getline(stream, tmp, ','))
		{
			//T_data[index].output[cnt] = stod(tmp);
			training_label[index][cnt] = stod(tmp);
			cnt++;
		}
		index++;
	}

	//未学習データを読み込む(入力)
	index = 1;

	while (getline(ifs_NT_input, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		test_data[index][0] = 0;
		//NT_data[index].input[0] = 0;//0番目は番兵
		cnt++;

		// 区切り文字がなくなるまで文字を区切っていく
		while (getline(stream, tmp, ','))
		{
			test_data[index][cnt] = stod(tmp);
			//NT_data[index].input[cnt] = stod(tmp);
			cnt++;
		}
		index++;
	}


	//未学習データを読み込む(出力)
	index = 1;

	while (getline(ifs_NT_output, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		//NT_data[index].output[0] = 0;//0番目は番兵
		test_label[index][0] = 0;
		cnt++;

		// 区切り文字がなくなるまで文字を区切っていく
		while (getline(stream, tmp, ','))
		{
			//NT_data[index].output[cnt] = stod(tmp);
			test_label[index][cnt] = stod(tmp);
			cnt++;
		}
		index++;
	}

	printf("data load Done\n");
	
	//入力のデバッグ用

	//以下一括学習を行う

	//****************************************************************************************
	//学習に使う変数
	int non_linear_input_siz = 1 + input_siz * (input_siz + 3) / 2;//非線形処理した入力の次元
	auto weight = make_v<double>(non_linear_input_siz + 10, class_siz + 5, maxi_component_siz + 5);//重み係数

	
	//*****************************************************************************************
	
	//cout << non_linear_input_siz << endl;
	//batch_learning( weight, T_data, each_class_siz, class_siz, input_siz, data_siz, maxi_component_siz, /*study_rate,*/ non_linear_input_siz,progress);


	//LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size)
	LLGMN model(study_rate, 10, data_siz, input_siz, output_siz, 2, data_siz);
	model.train(training_data, training_label);
	model.eval(test_data, test_label);

	return 0;
	


	//学習終了
	//以降未学習クラスの識別を行う

	ofstream ofs("abc2.csv");


	if (ofs.fail())
	{
		cout << "failed " << endl;
		exit(1);
	}

	auto input_layer = make_v<double>(data_siz + 10, 0);//非線形処理した入力層の入力次元
	auto mid_layer_input = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//各層の入力値
	auto mid_layer_output = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//各層の出力値
	auto output_layer = make_v<double>(data_siz + 10, class_siz + 10);//出力層の各クラスの出力値（事後確率）


	//初期化
	double log_likelihood = 0;
	fill_v(mid_layer_input, 0);
	fill_v(mid_layer_output, 0);
	fill_v(output_layer, 0);

	cout << "未学習データの識別" << endl;

	forward(NT_data, weight, input_layer, output_layer, mid_layer_input, mid_layer_output, each_class_siz, class_siz, 
		log_likelihood, non_linear_input_siz, data_siz, maxi_component_siz, input_siz, true, study_rate);


	for (int data_num = 1; data_num <= data_siz; data_num++)
	{
		//入力データ（未学習）
		for (int input_num = 1; input_num  <= input_siz ; input_num ++)
		{
			ofs << NT_data[data_num].input[input_num] << ",";
		}
		double maxi = 0;
		int idx = 0;
		ofs << " " << ",";
		//出力値
		for (int class_num = 1;class_num <= class_siz; class_num++)
		{
			ofs << output_layer[data_num][class_num] << ",";
			if (maxi < output_layer[data_num][class_num])
			{
				maxi = output_layer[data_num][class_num];
				idx = class_num;
			}
		}
		//識別したクラス
		ofs << idx;
		ofs << endl;

		if (NT_data[data_num].output[idx] == 1)
		{
			success_case++;
		}
	}

	ofs << "識別率 =," << success_case / (double)data_siz << endl;

	for (int i = 0; i < progress.size(); i++)
	{
		ofs << i+1 << "," << progress[i] << endl;
	}

}