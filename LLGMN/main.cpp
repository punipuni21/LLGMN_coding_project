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
#include <iomanip>

#include "in_out.h";
#include "macro.h";
#include "teaching_data.h";
#include "learning.h"; 
#include "LLGMN.h";
#include "utility.h";
using namespace std;

int main() {

	//***********************************************************************************************************
	//変数
	
	string name;
	const int input_size = 2;
	const int class_size = 4;
	const int component_size = 2;
	const int epoch = 40;
	const int each_class_data_size = 200;
	int data_size = 0;
	double lr = 0.1;


	const string training_data_file_name = "lea_sig.csv";
	const string training_label_file_name = "lea_T_sig.csv";
	const string test_data_file_name = "dis_sig.csv";
	const string test_label_file_name = "dis_T_sig.csv";

	//各クラスのデータ数（できればそろえること）
	for (int class_num = 1; class_num <= class_size; class_num++)
	{
		data_size += each_class_data_size;
	}

	auto training_data = make_v<double>(data_size + 5, input_size + 1);
	auto training_label = make_v<double>(data_size + 5, class_size + 1);
	auto test_data = make_v<double>(data_size + 5, input_size + 1);
	auto test_label = make_v<double>(data_size + 5, class_size + 1);

	//***************************************************************************************************************************************
	//教師データと検証データのファイルを先に開いておく

	load_data(training_data, training_data_file_name);
	load_data(training_label, training_label_file_name);
	load_data(test_data, test_data_file_name);
	load_data(test_label, test_label_file_name);

	//***************************************************************************************************************************************

	printf("data load Done\n");

	//LLGMN(double lr, int epochs, int batch_size, int input_dim, int class_size, int component_size, int data_size)
	
	LLGMN model(lr, epoch, data_size, input_size, class_size, 2, data_size);
	model.train(training_data, training_label);
	model.eval(test_data, test_label);

	return 0;
	
}