#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cmath>
#include<cstdio>
using namespace std;

//�I�����̕\��
void print_option() {

	putchar('\n');
	printf("*****************************************************\n");

	printf("[a]�ꊇ�w�K\n");
	printf("[b]�����w�K\n");
	printf("[ESC]�I��\n");

	printf("*****************************************************\n");

	printf("�I��������͂��Ă�������");
}

//�I�����̑I��
void select_choice(string& choice) {

	cin >> choice;

}



//���͂ɕ������܂܂�邩�̔���
void input(string& data) {

	bool flag = true;
	while (flag) {

		int dot_count = 0;
		cin >> data;

		if ((isdigit(data[0]) == 0 && data[0] != '-') || data[data.length() - 1] == '.' || (data == "-")) {

			printf("�ē��͂��Ă�������: ");
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
				printf("�ē��͂��Ă�������: ");
			}
		}
	}
}


//string�^��double�^�ɕϊ�
double string_to_double(string& name) {

	double num = atof(name.c_str());
	return num;
}
