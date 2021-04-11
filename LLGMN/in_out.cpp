#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cmath>
#include<cstdio>
using namespace std;

//選択肢の表示
void print_option() {

	putchar('\n');
	printf("*****************************************************\n");

	printf("[a]一括学習\n");
	printf("[b]逐次学習\n");
	printf("[ESC]終了\n");

	printf("*****************************************************\n");

	printf("選択肢を入力してください");
}

//選択肢の選択
void select_choice(string& choice) {

	cin >> choice;

}



//入力に文字が含まれるかの判定
void input(string& data) {

	bool flag = true;
	while (flag) {

		int dot_count = 0;
		cin >> data;

		if ((isdigit(data[0]) == 0 && data[0] != '-') || data[data.length() - 1] == '.' || (data == "-")) {

			printf("再入力してください: ");
		}
		else {
			bool exist = true;
			for (int i = 1; i < data.length() && exist == true; i++) {
				if (data[i] == '.' && dot_count < 1) {
					dot_count++;
				}
				else if (data[i] == '.' && dot_count >= 1) {
					exist = false;
				}
				else if (isdigit(data[i]) == 0) {
					exist = false;
				}
			}
			if (exist) {
				flag = false;
			}
			else {
				printf("再入力してください: ");
			}
		}
	}
}


//string型をdouble型に変換
double string_to_double(string& name) {

	double num = atof(name.c_str());
	return num;
}
