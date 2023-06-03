#include "hw4.h"
#include "image.h"
#include "parallel.h"
#include <vector>
#include <string>
#include <thread>

int main(int argc, char *argv[]) {
    std::vector<std::string> parameters;
    std::string hw_num;
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t") {
            num_threads = std::stoi(std::string(argv[++i]));
        } else if (std::string(argv[i]) == "-hw") {
            hw_num = std::string(argv[++i]);
        } else {
            parameters.push_back(std::string(argv[i]));
        }
    }

    parallel_init(num_threads);

    if (hw_num == "4_1") {
        Image3 img = hw_4_1(parameters);
        imwrite("hw_4_1.exr", img);
    } else if (hw_num == "4_2") {
        Image3 img = hw_4_2(parameters);
        imwrite("hw_4_2.exr", img);
    } else if (hw_num == "4_3") {
        Image3 img = hw_4_3(parameters);
        imwrite("hw_4_3.exr", img);
    } else if (hw_num == "4_4") {
        Image3 img = hw_4_4(parameters);
        imwrite("hw_4_4.exr", img);
    }

    parallel_cleanup();

    return 0;
}
