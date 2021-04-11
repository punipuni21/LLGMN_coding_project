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


public:


	//学習率 lr
	//エポック数
	//バッチサイズ
	//コンポーネント数：
	//入力次元数
	//出力次元（クラス数）
	const double eps = 1e-5;//対数尤度の許容値
	const double beta = 0.8;//定数
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

	vector<vector<double>> input_layer_;
	vector<vector<vector<double>>> mid_layer_input_;
	vector<vector<vector<double>>> mid_layer_output_;
	vector<vector<double>> output_layer_;

	//教師データ：(バッチサイズ，次元...)
	//正解ラベル：(バッチサイズ，次元...)※まずはone-hotで実装すること


	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_size, int data_size) {

		lr_ = lr;
		epochs_ = epochs;
		batch_size_ = batch_size;
		input_dim_ = input_dim;
		output_dim_ = class_num;
		component_size_ = component_size;
		data_size_ = data_size;
		non_linear_input_siz_ = 1 + input_dim * (input_dim + 3) / 2;
		weight_ = make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_size_ + 5);//重み係数

		input_layer_ = make_v<double>(data_size_ + 10, 0);
		mid_layer_input_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
		mid_layer_output_ = make_v<double>(data_size_ + 5, class_num_ + 5, component_size_ + 5);
		output_layer_ = make_v<double>(data_size_ + 10, class_num_ + 10);

		cout << epochs << " " << batch_size_ << " " << input_dim_ << " " << output_dim_ << " " << class_num << " " << component_size_ << endl;
		cout << non_linear_input_siz_ << endl;
	}


	void train(vector<vector<double>> & training_data, vector<vector<double>> & training_label) {

		weight_initialize();
		//データは正常に渡されている
		
		//今後のTodo
		//for 最初のデータ数：
			//for バッチサイズ分：
				//forward
				//ロスの計算
				//正解率などの計算
				//backward

		//とりあえず一括学習で実装して余裕があればバッチサイズを変更できるようにすること
		bool flag = true;

		for (int i = 0; i < epochs_; i++) {
			//初期化
			double log_likelihood = 0;
			fill_v(mid_layer_input_, 0);
			fill_v(mid_layer_output_, 0);
			fill_v(output_layer_, 0);


			//forward

			forward(training_data, training_label, flag);

			cout << "epoch: " << i << " log_likelihood= " << log_likelihood_ << " lr= " << lr_ << endl;

			calc_accuracy();

			//backward

			backward(training_data, training_label);

			flag = false;
		}

		//ログの出力
		
		save_confufion_matrix();

		save_log();

		//重みの保存など

		save_weight();
	}


	void forward(vector<vector<double>> & training_data, vector<vector<double>> & training_label, bool flag) {

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
		progress_.push_back(log_likelihood_);
		if (true)
		{
			lr_ = pow(log_likelihood_, 1 - beta) / (epochs_ * (1 - beta));
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


	void eval(vector<vector<double>> & training_data, vector<vector<double>> & training_label) {

		//forward

		//正解率，混同行列の算出など

	}


	void calc_accuracy() {

		//正解率の算出

	}

	void save_weight() {

		//重みの保存

	}

	void save_confusuin_matrix() {

		//混同行列の計算，保存
	}

	void save_log() {

		//ロスの保存と混同行列の算出，保存

	}

	//重みの初期化
	void weight_initialize() {

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
};