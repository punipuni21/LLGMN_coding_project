#include<iostream>
#include<string>
#include<sstream>
#include<iomanip>

using namespace std;

string get_current_time() {

	time_t t = time(nullptr);
	const tm* localTime = localtime(&t);

	std::stringstream s;
	s << "20" << localTime->tm_year - 100;
	s << "_";
	// setw(),setfill()‚Å0‹l‚ß
	s << setw(2) << setfill('0') << localTime->tm_mon + 1;
	s << "_";
	s << setw(2) << setfill('0') << localTime->tm_mday;
	s << "_";
	s << setw(2) << setfill('0') << localTime->tm_hour;
	s << "_";
	s << setw(2) << setfill('0') << localTime->tm_min;
	s << "_";
	s << setw(2) << setfill('0') << localTime->tm_sec;

	return s.str();
}