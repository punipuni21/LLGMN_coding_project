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

const double eps = 1e-5;//対数尤度の許容値
const int rep = 1000;//試行回数の上限
const double beta = 0.8;//定数
const double sampling_time = 0.001;

//******************************************************************************************************************

//プロトタイプ宣言

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
	bool flag,//初期値を更新するかのフラグ
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




//重み係数の初期化
void value_initialize(vector<vector<vector<double> > >& weight);


//weight[i][j][k]:=i番目の非線形処理をした入力ノードと第2層のクラスjのコンポーネントkのノードの間の重み係数
//mid_layer_input[i][j][k]:=i番目のデータのクラスjのコンポーネントkの入力
//mid_layer_output[i][j][k]:=i番目のデータのクラスjのコンポーネントkの出力


//*********************************************************************************************************************


//学習

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
	vector<double>& progress//対数尤度の過程
) {

	//*****************************************************************************
	//変数
	int cnt = 0;//学習回数
	double log_likelihood = 0.0;//対数尤度
	bool flag = false;//初期値を更新するかのフラグ(最初の学習の時のみtrue)
	double study_rate = 0;
	
	//*****************************************************************************
	//重み係数の初期化

	value_initialize(weight);


	//最後のクラスの最後のコンポーネントへの重みは0にする

	for (int node = 1; node <= non_linear_input_siz; node++)
	{
		weight[node][class_siz][each_class_siz[class_siz]] = 0;
	}
	//*****************************************************************************

	auto input_layer = make_v<double>(data_siz + 10, 0);//非線形処理した入力層の入力次元
	auto mid_layer_input = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//各層の入力値
	auto mid_layer_output = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//各層の出力値
	auto output_layer = make_v<double>(data_siz + 10, class_siz + 10);//出力層の各クラスの出力値（事後確率）
	auto before = make_v<double>(non_linear_input_siz + 10, class_siz + 5, maxi_component_siz + 5);//前回の値

	do {

		
		//初期化
		log_likelihood = 0;
		fill_v(mid_layer_input, 0);
		fill_v(mid_layer_output, 0);
		fill_v(output_layer, 0);
		fill_v(before, 0);
		

		//順方向伝搬（対数尤度は参照渡し）
		if (cnt == 0)
		{
			//初回のみtrue
			flag = true;
		}
		else
		{
			flag = false;
		}
		forward(T_data, weight, input_layer, output_layer, mid_layer_input, mid_layer_output,
			each_class_siz, class_siz, log_likelihood, non_linear_input_siz, data_siz, maxi_component_siz, input_siz, flag, study_rate);
		
		
		//	逆方向伝播

		backward(T_data, weight, input_layer, output_layer, mid_layer_output,
			each_class_siz, class_siz, non_linear_input_siz, data_siz, study_rate, log_likelihood, before);


		//学習回数の更新を忘れない
			   
		cnt++;

		cout << "cnt = " << cnt << " log_likelihood = " << log_likelihood << endl;
		progress.push_back(log_likelihood);

	} while (abs(log_likelihood) > eps && cnt < rep);
}


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
	bool flag,//初期値を更新するかのフラグ
	double& study_rate//学習係数
) {
	//処理

	//**********************************************************************************************
	//変数

	auto exp_num = make_v<double>(data_siz + 5, class_siz + 5, maxi_component_siz + 5);//前処理したexpの値
	fill_v(exp_num, 0);

	//*************************************************************************************************
	for (int data_num = 1; data_num <= data_siz; data_num++)
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
				input_layer[data_num].push_back(0);//番兵を入れて1-indexにする
				//0次の項
				input_layer[data_num].push_back(1);
				//1次の項
				for (int index = 1; index <= input_siz; index++)
				{
					input_layer[data_num].push_back(T_data[data_num].input[index]);
				}

				//2次の項
				for (int outside = 1; outside <= input_siz; outside++)
				{
					for (int inside = outside; inside <= input_siz; inside++)
					{
						input_layer[data_num].push_back(T_data[data_num].input[outside] * T_data[data_num].input[inside]);
					}
				}

				//この時点で入力次元は1+input_siz * (input_siz + 3) / 2になっている(番兵を数えない場合)

				//*********************************************************************************************************************

				for (int tm = 0; tm < 10; tm++)
				{
					input_layer[data_num].push_back(0);//後ろにも番兵（少し多めに入れる）
				}
				//***********************************************************************************************************************

				//中間層に伝搬させる

				for (int now_node = 1; now_node <= non_linear_input_siz; now_node++)//入力層のノード番号
				{
					for (int class_num = 1; class_num <= class_siz; class_num++)//中間層のクラス番号
					{
						for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num番目のクラスのコンポーネント番号
						{

							mid_layer_input[data_num][class_num][component_num] +=
								input_layer[data_num][now_node] * weight[now_node][class_num][component_num];

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

				//中間層の入力から出力値を求める

				for (int class_num = 1; class_num <= class_siz; class_num++)//クラス番号
				{
					for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num番目のクラス番号のコンポーネント番号
					{
						mid_layer_output[data_num][class_num][component_num]
							= exp_num[data_num][class_num][component_num] / cnt;
					}
				}
				//***********************************************************************************************************************************
				//出力層に伝播させる

				for (int class_num = 1; class_num <= class_siz; class_num++)//クラス番号kに属するものをまとめる
				{
					for (int component_num = 1; component_num <= each_class_siz[class_num]; component_num++)//class_num番目のクラス番号のコンポーネント番号
					{
						//中間層のclass_num番目のクラスに属するコンポーネントを全て出力層のclass_num番のクラスにまとめる
						output_layer[data_num][class_num] += mid_layer_output[data_num][class_num][component_num];
					}
				}
			}
			else if (layer_num == 3)
			{
				//出力層

				//対数尤度*(-1)を計算する(-1)をかけることでこれを最小化できれば対数尤度が最大化するため)
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

	//ガンマを求めるための分母の前処理


	gamma = pow(log_likelihood, beta) / denom;


	//重み係数の更新
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





//重み係数を決定する関数
void value_initialize(vector<vector<vector<double> > >& weight) {

	// 0.0以上1.0未満の値を等確率で発生させる
	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
	std::uniform_real_distribution<> rand12(-1.0, 1.0);//[-1.0,1.0]の乱数を生成

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
