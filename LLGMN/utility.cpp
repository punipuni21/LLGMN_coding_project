#include<iostream>
#include<string>
#include<sstream>
#include<iomanip>
#include<direct.h>

using namespace std;

string get_current_time(string split) {

	time_t t = time(nullptr);
	const tm* localTime = localtime(&t);

	std::stringstream s;
	s << "20" << localTime->tm_year - 100;
	s << split;
	// setw(),setfill()で0詰め
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

	//_mkdirの返り値は0ならディレクトリが新たに作成されたことになり，0以外ならばディレクトリの作成に失敗

	return (_mkdir(dir) == 0);
}