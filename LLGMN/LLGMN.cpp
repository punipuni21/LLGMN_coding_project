#pragma once
#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>
#include<fstream>
#include<iomanip>
#include<numeric>
#include<direct.h>

#include "macro.h";
#include "LLGMN.h"
#include "utility.h";

//クラスのメンバは末尾にアンダースコアを書くこと

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
	weight_ = make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_size_ + 5);//重み係数
	current_time_ = get_current_time("_");
	progress_.push_back(0);//1-index

	input_layer_ = make_v<double>(data_size_ + 10, 0);
	mid_layer_input_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
	mid_layer_output_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
	output_layer_ = make_v<double>(data_size_ + 10, class_num_ + 10);
};


void LLGMN::train(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

	//重みの初期か
	weight_initialize();
	
	//今後のTodo
	//for 最初のデータ数：
		//for バッチサイズ分：
			//forward
			//ロスの計算
			//正解率などの計算
			//backward

	//とりあえず一括学習で実装して余裕があればバッチサイズを変更できるようにすること
	bool flag = true;
	double accuracy = 0;

	for (int i = 0; i < epochs_; i++) {
		//初期化
		double log_likelihood = 0;
		fill_v(mid_layer_input_, 0);
		fill_v(mid_layer_output_, 0);
		fill_v(output_layer_, 0);


		//forward
		forward(training_data, training_label);
		progress_.push_back(log_likelihood_);

		if (i == 0) {
			lr_ = pow(log_likelihood_, 1 - beta) / (epochs_ * (1 - beta));
		}

		cout << "epoch: " << i << " log_likelihood= " << log_likelihood_ << " lr= " << lr_ << endl;

		//backward
		backward(training_data, training_label);

	}

	//ログの出力
	save_loss();

	//重みの保存など
	save_weight();
}


void LLGMN::forward(vector<vector<double>>& training_data, vector<vector<double>>& training_label) {

	auto exp_num = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);//前処理したexpの値
	fill_v(exp_num, 0);

	//*************************************************************************************************
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		//1-index
		//data_num番目のデータについて順方向伝播させる
		for (int layer_num = 1; layer_num <= 3; layer_num++)
		{
			//入力層，中間層，出力層
			if (layer_num == 1)
			{
				//入力層の処理
				//*******************************************************************************************************************
				//非線形処理				
				input_layer_[data_num].push_back(0);//番兵を入れて1-indexにする
				//0次の項
				input_layer_[data_num].push_back(1);
				//1次の項
				for (int index = 1; index <= input_dim_; index++)
				{
					input_layer_[data_num].push_back(training_data[data_num][index]);
				}

				//2次の項
				for (int outside = 1; outside <= input_dim_; outside++)
				{
					for (int inside = outside; inside <= input_dim_; inside++)
					{
						input_layer_[data_num].push_back(training_data[data_num][outside] * training_data[data_num][inside]);
					}
				}

				//この時点で入力次元は 1 + input_siz * (input_siz + 3) / 2になっている(番兵を数えない場合)

				//*********************************************************************************************************************

				for (int tm = 0; tm < 10; tm++)
				{
					input_layer_[data_num].push_back(0);//後ろにも番兵（少し多めに入れる）
				}
				//***********************************************************************************************************************

				//中間層に伝搬させる

				for (int now_node = 1; now_node <= non_linear_input_siz_; now_node++)//入力層のノード番号
				{
					for (int class_num = 1; class_num <= class_num_; class_num++)//中間層のクラス番号
					{
						for (int component_num = 1; component_num <= component_size_; component_num++)//class_num番目のクラスのコンポーネント番号
						{

							mid_layer_input_[data_num][class_num][component_num] +=
								input_layer_[data_num][now_node] * weight_[now_node][class_num][component_num];

						}
					}
				}
			}
			else if (layer_num == 2)
			{
				//中間層

				//****************************************************************************************************************************
				//expの前処理を行う

				double cnt = 0;//分母(全てのクラス，コンポーネントの入力のexpの総和)
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

				//中間層の入力から出力値を求める

				for (int class_num = 1; class_num <= class_num_; class_num++)//クラス番号
				{
					for (int component_num = 1; component_num <= component_size_; component_num++)//class_num番目のクラス番号のコンポーネント番号
					{
						mid_layer_output_[data_num][class_num][component_num]
							= exp_num[data_num][class_num][component_num] / cnt;
					}
				}
				//***********************************************************************************************************************************
				//出力層に伝播させる

				for (int class_num = 1; class_num <= class_num_; class_num++)//クラス番号kに属するものをまとめる
				{
					for (int component_num = 1; component_num <= component_size_; component_num++)//class_num番目のクラス番号のコンポーネント番号
					{
						//中間層のclass_num番目のクラスに属するコンポーネントを全て出力層のclass_num番のクラスにまとめる
						output_layer_[data_num][class_num] += mid_layer_output_[data_num][class_num][component_num];
					}
				}
			}
			else if (layer_num == 3)
			{
				//出力層

				//対数尤度*(-1)を計算する(-1)をかけることでこれを最小化できれば対数尤度が最大化するため)
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
	//denom /= data_size_;
	//ガンマを求めるための分母の前処理


	gamma = pow(log_likelihood_, beta) / denom;

	//重み係数の更新
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

	//accuracy = calc_accuracy(test_label, output_layer_);
	
	//混同行列の算出など
	save_confusion_matrix(test_label, output_layer_);
}


void LLGMN::save_result(vector<vector<double>>& test_data, vector<vector<double>>& test_label, vector<vector<double>>& output_layer) {

	if (is_current_time_exist_) {
		//既に結果保存用ディレクトリが作成されている．
		cout << "既に作成されています．" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "ディレクトリ" << current_time_ << "を作成しました" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "ディレクトリ" << current_time_ << "を作成できませんでした" << endl;
		}
	}

	//ロスの保存と混同行列の算出，保存
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
		//入力データ（未学習）
		for (int input_num = 1; input_num <= input_dim_; input_num++)
		{
			ofs << test_data[data_num][input_num] << ",";
		}
		double maxi = 0;
		int idx = 0;
		//出力値
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

		//事後確率最大のクラスがone-hot表現で1が立っていた場合
		if (test_label[data_num][idx] == 1)
		{
			success_case++;
		}
		//識別したクラス
		ofs << "," << idx << endl;
	}

	ofs << "識別率 =," << success_case / (double)data_size_ << endl;
}


double LLGMN::calc_accuracy(vector<vector<double>>& test_label, vector<vector<double>>& output_layer) {

	//正解率の算出
	double maxi = 0;
	int idx = 0;
	int success_case = 0;
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		double maxi = 0;
		int idx = 0;
		//出力値
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
		//既に結果保存用ディレクトリが作成されている．
		cout << "既に作成されています．" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "ディレクトリ" << current_time_ << "を作成しました" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "ディレクトリ" << current_time_ << "を作成できませんでした" << endl;
		}
	}
	//重みの保存
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_weight.csv");

	if (ofs.fail())
	{
		cout << "failed " << endl;
		exit(1);
	}


	//重みの保存

	//クラスの列名を入力
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
	//コンポーネントの列名を入力
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

	//混同行列の計算，保存

	if (is_current_time_exist_) {
		//既に結果保存用ディレクトリが作成されている．
		cout << "既に作成されています．" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "ディレクトリ" << current_time_ << "を作成しました" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "ディレクトリ" << current_time_ << "を作成できませんでした" << endl;
		}
	}

	//ロスの保存と混同行列の算出，保存
	ofstream ofs("./" + current_time_ + "/" + current_time_ + "_confusion_matrix.csv");

	auto confusion_matrix = make_v<int>(class_num_ + 1, class_num_ + 1);//1-index
	fill_v(confusion_matrix, 0);

	//混同行列の算出
	double maxi = 0;
	for (int data_num = 1; data_num <= data_size_; data_num++)
	{
		double maxi = 0;
		int estimated_class_id = 0;
		int label_class_id = 0;
		//出力値
		for (int class_num = 1; class_num <= class_num_; class_num++)
		{
			//推定したクラスの算出
			if (maxi < output_layer_[data_num][class_num])
			{
				maxi = output_layer_[data_num][class_num];
				estimated_class_id = class_num;
			}
			//one-hotラベルから教師ラベルのクラスを求める
			if (test_label[data_num][class_num] == 1) {
				label_class_id = class_num;
			}
		}
		confusion_matrix[label_class_id][estimated_class_id]++;
	}

	//混同行列の保存
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
	ofs << endl;

	//Recall, Precision, F1-measureなどを算出
	
	vector<double> Recall(class_num_ + 1, 0), Precision(class_num_ + 1, 0), F1_measure(class_num_ + 1, 0);

	for (int i = 1; i <= class_num_; i++)
	{
		int TP = 0, FN = 0, FP = 0;
		//Recall
		for (int j = 1; j <= class_num_; j++)
		{
			if (i == j) {
				TP += confusion_matrix[i][j];
			}
			else {
				FN += confusion_matrix[i][j];
			}
		}
		Recall[i] = TP / (double)(TP + FN);
		//Precision
		TP = 0, FN = 0, FP = 0;
		for (int j = 1; j <= class_num_; j++)
		{
			if (i == j) {
				TP += confusion_matrix[j][i];
			}
			else {
				FP += confusion_matrix[j][i];
			}
		}
		Precision[i] = TP / (double)(TP + FP);

		F1_measure[i] = 2 * Precision[i] * Recall[i] / (double)(Precision[i] + Recall[i]);
	}

	ofs << ",";
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << "class " << i << ",";
	}
	ofs << "average" << endl;
	ofs << "Recall,";
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << Recall[i] << ",";
	}

	ofs << accumulate(Recall.begin(), Recall.end(), 0.0) / (double)class_num_ << endl;

	ofs << "Precision,";
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << Precision[i] << ",";
	}
	ofs << accumulate(Precision.begin(), Precision.end(), 0.0) / (double)class_num_ << endl;

	ofs << "F1-measure,";
	for (int i = 1; i <= class_num_; i++)
	{
		ofs << F1_measure[i] << ",";
	}
	ofs << accumulate(F1_measure.begin(), F1_measure.end(), 0.0) / (double)class_num_ << endl;
}


void LLGMN::save_loss() {
	

	if (is_current_time_exist_) {
		//既に結果保存用ディレクトリが作成されている．
		cout << "既に作成されています．" << endl;
	}
	else {
		if (make_dir(current_time_)) {
			cout << "ディレクトリ" << current_time_ << "を作成しました" << endl;
			is_current_time_exist_ = true;
		}
		else {
			cout << "ディレクトリ" << current_time_ << "を作成できませんでした" << endl;
		}
	}

	//ロスの保存と混同行列の算出，保存
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

	// 0.0以上1.0未満の値を等確率で発生させる
	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
	std::uniform_real_distribution<> rand12(-1.0, 1.0);//[-1.0,1.0]の乱数を生成

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
