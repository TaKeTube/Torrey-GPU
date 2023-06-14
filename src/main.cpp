#include "mat_mult.cuh"
#include "mat_mult_vec.cuh"
#include <vector>
#include <iostream>
#include <variant>
#include "hw4.h"
#include "image.h"
#include <vector>
#include <string>
#include <thread>

/*int main()
{
    // vector<float> A = {
    //     1, 2, 3,
    //     4, 5, 6,
    //     7, 8, 9
    // };
    // vector<float> B = {
    //     1, 2, 3,
    //     4, 5, 6,
    //     7, 8, 9
    // };
    // vector<float> A(128*128, 1);
    // vector<float> B(128*128, 1);

    variant<int, float> v;
    v = 42; // v contains int
    int i = std::get<int>(v);
    std::cout << i << std::endl;

    vector<Vector3f> A(128*128, Vector3f{1.0, 1.0, 1.0});
    vector<Vector3f> B(128*128, Vector3f{1.0, 1.0, 1.0});
    vector<float> C(128*128);
    Vector3f* host_A = A.data();
    Vector3f* host_B = B.data();
    float* host_C = C.data();
    mat_mult_vec(host_A, host_B, host_C, 128, 128, 128, 128, 128, 128);
    // for(auto i : C) {
    //     cout << i << " " << endl;
    // }
    return 0;
}*/

int main(int argc, char* argv[])
{
    std::vector<std::string> parameters;
    std::string hw_num;
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t") {
            num_threads = std::stoi(std::string(argv[++i]));
        }
        else if (std::string(argv[i]) == "-hw") {
            hw_num = std::string(argv[++i]);
        }
        else {
            parameters.push_back(std::string(argv[i]));
        }
    }

    if (hw_num == "4_1")
    {
        Image3 img = hw_4_1(parameters);
        imwrite("hw_4_1.exr", img);
    }
    else if (hw_num == "4_2")
    {
        Image3 img = hw_4_2(parameters);
        imwrite("hw_4_2.exr", img);
    }
    else
    {
        Image3 img = hw_4_3(parameters);
        imwrite("hw_4_3.exr", img);
    }

    return 0;
}