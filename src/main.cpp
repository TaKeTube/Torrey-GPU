#include "render.h"
#include "image.h"
#include <vector>
#include <string>
#include <thread>

int main(int argc, char *argv[]) {
    std::vector<std::string> parameters;
    std::string hw_num;
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        parameters.push_back(std::string(argv[i]));
    }
    
    Image3 img = render(parameters);
    imwrite("Torrey_GPU.exr", img);

    return 0;
}
