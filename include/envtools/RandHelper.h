/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 27/04/2024
 * FILE LAST UPDATED: 09/05/2024
 * 
 * DESCRIPTION: Class definition and implementation for random distributions.
*/

#include <algorithm>
#include <random>

class RandHelper
{
    std::mt19937 gn;
    std::default_random_engine rng;

public:
    RandHelper()
    {
        std::random_device rd;

        std::mt19937 generator(rd());
        gn = generator;

        rng = std::default_random_engine {};

        return;
    };

    ~RandHelper(){};

    int random_int_range(const int range_from, const int range_to)
    {
        std::uniform_int_distribution<int> distr(range_from, range_to);
        return distr(gn);
    };

    double random_double_range(const double range_from, const double range_to)
    {
        std::uniform_real_distribution<double> distr(range_from, range_to);
        return distr(gn);
    }

    int choose_random_action(const std::vector<int>& vec)
    {
        std::vector<int> cp(vec);
        std::shuffle(cp.begin(), cp.end(), rng);

        return cp[0];
    }
};