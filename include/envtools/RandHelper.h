/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 27/04/2024
 * FILE LAST UPDATED: 27/04/2024
 * 
 * DESCRIPTION: Class definition and implementation for random distributions.
*/

#include <random>

class RandHelper
{
    std::mt19937 gn;

public:
    RandHelper()
    {
        std::random_device rd;

        std::mt19937 generator(rd());
        gn = generator;

        return;
    };

    ~RandHelper(){};

    int random_int_range(const int range_from, const int range_to)
    {
        std::uniform_int_distribution<int> distr(range_from, range_to);

        return distr(gn);
    };
};