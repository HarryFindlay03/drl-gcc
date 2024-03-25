#include <vector>

#include "cpp-nn/network.h"
#include "BufferItem.h"

class Agent
{
    /* NETWORKS */
    ML_ANN* Q;
    ML_ANN* Q_hat;

    /* REPLAY BUFFER */
    std::vector<BufferItem*> buff;

    /* PARAMS */
    unsigned int buffer_size;
    unsigned int copy_period;
    unsigned int number_of_episodes;
    unsigned int episode_length;

    // state extraction function
    // loss function - passed to cpp-nn!


public:
    Agent
    (
        const std::vector<size_t>& network_config, 
        const unsigned int buffer_size, 
        const unsigned int copy_period,
        const unsigned int number_of_episodes,
        const unsigned int episode_length
    );

    ~Agent()
    {
        // delete each buffer item
        for(auto it = buff.begin(); it != buff.end(); ++it)
            delete *it;

        delete Q;
        delete Q_hat;
    };

    void copy_network_weights();
};