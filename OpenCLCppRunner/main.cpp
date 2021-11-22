#include <iostream>

#include "implementations/vector_add.h"

int main()
{
    std::string res = measureExecTime(vector_add, 50);
    std::cout << "OpenCLRunner, Exec time: " << res << std::endl;
    return 0;
}
