#include<iostream>
#include<vector>
#include<algorithm>
#include<ctime>
#include<random>

#include "macro.h";

//クラスのメンバは末尾にアンダースコアを書くこと

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
	int component_siz_ = 2;
	int non_linear_input_siz_ = 1;

	int non_linear_input_siz_ = 1;//非線形処理した入力の次元
	vector<vector<vector<double>>> weight_;

	LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_num, int component_siz) {

		lr_ = lr;
		epochs_ = epochs;
		batch_size_ = batch_size;
		input_dim_ = input_dim;
		output_dim_ = class_num;
		component_siz_ = component_siz;
		non_linear_input_siz_ = 1 + input_dim * (input_dim + 3) / 2;
		make_v<double>(non_linear_input_siz_ + 10, class_num_ + 5, component_siz_ + 5);//重み係数

	}



	//forward（教師データ，正解ラベル）



	//backward



	//重みの初期化
	void value_initialize() {

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