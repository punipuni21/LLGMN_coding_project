#pragma once

#include<string>
#include<vector>

//���݂̎������擾
string get_current_time(string split="");

int make_dir(string s);

void load_data(vector<vector<double>>& data, const string& file_name);