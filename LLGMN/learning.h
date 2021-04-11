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

//配列は全て1-indexで扱うこと


//学習
void batch_learning(
	vector<vector<vector<double> > >& weight,//重み係数
	vector<teaching_data>& T_data,//教師データ
	vector<int>& each_class_siz,//各クラスのコンポーネント数
	int class_siz,//クラス数
	int input_siz,//入力次元
	int data_siz,//データ数
	int maxi_component_siz,//最大コンポーネント数
	//double study_rate,//学習係数
	int non_linear_input_siz,//非線形処理した入力次元
	vector<double>& progress
);



//順方向伝搬

void forward(

	vector<teaching_data>& T_data,//教師データ
	vector<vector<vector<double> > >& weight,//重み係数
	vector<vector<double> >& input_layer,//入力層
	vector<vector<double> >& output_layer,//出力層
	vector<vector<vector<double> > >& mid_layer_input,//各層の入力値
	vector<vector<vector<double> > >& mid_layer_output,//各層の出力値
	vector<int>& each_class_siz,//各クラスのコンポーネント数
	int class_siz,//クラス数
	double& log_likelihood,//対数尤度
	int non_linear_input_siz,//非線形処理した入力次元数
	int data_siz,//データ数
	int maxi_component_siz,//最大コンポーネント数
	int input_siz,//入力次元
	bool flag,//初期値を更新するかどうかのフラグ
	double& study_rate//学習係数
);


//逆方向伝播

void backward(

	vector<teaching_data>& T_data,//教師データ
	vector<vector<vector<double> > >& weight,//重み係数
	vector<vector<double> >& input_layer,//入力層
	vector<vector<double> >& output_layer,//出力層
	vector<vector<vector<double> > >& mid_layer_output,//各層の出力値
	vector<int>& each_class_siz,//各クラスのコンポーネント数
	int class_siz,//クラス数
	int non_linear_input_siz,//非線形処理した入力次元数
	int data_siz,//データ数
	double study_rate,//学習係数
	double log_likelihood,//対数尤度
	vector<vector<vector<double> > >& before//前回の値
);

