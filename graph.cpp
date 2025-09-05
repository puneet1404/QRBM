#include <matplot/matplot.h>
#include <fstream>
#include <vector>
#include <iostream>
using namespace std;

void plot(vector<double> &itt_no, vector<double> &virst_vec, vector<double> &c, string s, string g, string name)
{
    using namespace matplot;
    plot(itt_no, virst_vec, "3");
    hold(on);
    plot(itt_no, c, "-o-");
    matplot::legend({s, g});
    save("./" + name + ".png");
    cla();
}

void load_vector(vector<double> &itt, vector<double> &eloc, vector<double> &control, double control_num, string file_name)
{
    // vector<double> itt,eloc,control;
    // double control_num =0;
    ifstream wao(file_name);
    string number;
    int n = 1;
    while (getline(wao, number))
    {
        // cout<<number<<endl;
        eloc.push_back(stod(number));
        itt.push_back(n);
        n++;
        control.push_back(control_num);
    }
}
int main()
{
    vector<double> itt, quantity, control;
    string num_spin,boundary_cond;

    cout<<"number of spins"<<endl;
    cin>>num_spin;
    cout<<"boundary_cond (o/c)"<<endl;
    cin>>boundary_cond;


    double control_num = 0;
    string wao("./data_"+boundary_cond+"_"+num_spin+"/");
    
    cout<<"enter exact value for energy"<<endl;
    cin>>control_num;
    load_vector(itt, quantity, control, control_num, wao+"e_loc.txt");
    plot(itt,quantity,control,"e_loc","exact_energy","e_loc");
    
    cout<<"enter exact value for magnetization"<<endl;
    cin>>control_num;

    load_vector(itt, quantity, control, control_num, wao+"magnetization.txt");
    plot(itt,quantity,control,"magnetization","exact_value","bruh");


    // load_vector(itt, quantity, control, control_num, wao);
    // plot(itt,quantity,control,"eloc","control","bruh");


    // load_vector(itt, quantity, control, control_num, wao);
    // plot(itt,quantity,control,"eloc","control","bruh");


}