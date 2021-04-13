#include<iostream>
#include<string>
#include<sstream>
#include<iomanip>
#include<direct.h>
#include<vector>
#include<fstream>
#include<sstream>

using namespace std;

string get_current_time(string split) {

	time_t t = time(nullptr);
	const tm* localTime = localtime(&t);

	std::stringstream s;
	s << "20" << localTime->tm_year - 100;
	s << split;
	// setw(),setfill()��0�l��
	s << setw(2) << setfill('0') << localTime->tm_mon + 1;
	s << split;
	s << setw(2) << setfill('0') << localTime->tm_mday;
	s << split;
	s << setw(2) << setfill('0') << localTime->tm_hour;
	s << split;
	s << setw(2) << setfill('0') << localTime->tm_min;
	s << split;
	s << setw(2) << setfill('0') << localTime->tm_sec;

	return s.str();
}

int make_dir(string s) {

	const char* dir = s.c_str();

	//_mkdir�̕Ԃ�l��0�Ȃ�f�B���N�g�����V���ɍ쐬���ꂽ���ƂɂȂ�C0�ȊO�Ȃ�΃f�B���N�g���̍쐬�Ɏ��s

	return (_mkdir(dir) == 0);
}

void load_data(vector<vector<double>>& data, const string& file_name) {
	
	ifstream ifs(file_name);
	if (ifs.fail())
	{
		cout << "failed to open the file named " << file_name << endl;
		exit(1);
	}

	string str;
	int index = 1;

	while (getline(ifs, str))
	{
		string tmp = "";
		istringstream stream(str);

		int cnt = 0;
		data[index][0] = 0;
		cnt++;
		//1-index��O��Ƃ���
		// ��؂蕶�����Ȃ��Ȃ�܂ŕ�������؂��Ă���
		while (getline(stream, tmp, ','))
		{
			data[index][cnt] = stod(tmp);
			cnt++;
		}
		index++;
	}
}