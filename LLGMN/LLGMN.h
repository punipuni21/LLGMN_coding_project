#pragma once
#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>

#include "macro.h";

//クラスのメンバは末尾にアンダースコアを書くこと

using namespace std;

class LLGMN {

private:

	const double eps = 1e-5;//対数尤度の許容値
	const double beta = 0.2;//定数

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

	//教師データ：(バッチサイズ，次元...)
	//正解ラベル：(バッチサイズ，次元...)※まずはone-hotで実装すること

	//コンストラクタ
	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size);

	//学習を回すメソッド
	void train(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//逆方向計算
	void forward(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//誤差逆伝播
	void backward(vector<vector<double>>& training_data, vector<vector<double>>& training_label);

	//テストデータでの推論
	void eval(vector<vector<double>>& test_data, vector<vector<double>>& test_label);

	//正解率の計算
	void calc_accuracy();

	//重みの保存
	void save_weight();

	//混同行列の算出
	void save_confusion_matrix();

	//ログの保存
	void save_log();

	//重みの初期化
	void weight_initialize();

};