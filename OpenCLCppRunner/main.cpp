#include <iostream>

#include "implementations/simple_mad.h"

int main()
{
    std::string res = measureExecTime(simple_mad, 10);
    std::cout << "OpenCLRunner, Exec time: " << res << std::endl;
    return 0;
}
