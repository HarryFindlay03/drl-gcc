/***
 * AUTHOR: Harry Findlay
 * LICENSE: Shipped with package - GNU GPL v3.0
 * FILE START: 25/04/2024
 * FILE LAST UPDATED: 09/05/2024
 * 
 * REQUIREMENTS: Eigen v3.4.0, src: https://eigen.tuxfamily.org/index.php?title=Main_Page

 * 
 * DESCRIPTION: Class definition for BufferItem that makes up each item in the replay buffer for DQL.
*/

#include <vector>

class BufferItem
{
    std::vector<double> curr_st;
    std::vector<double> next_st;
    std::vector<std::string> actions_avail;
    double reward;
    bool terminate;
    int action_pos; 

    inline void copy_state(std::vector<double>& to, const std::vector<double>& from)
    {
        to.resize(from.size());

        int pos = 0;
        for(const auto& v : from)
            to[pos++] = v;
        
        return;
    };

public:
    BufferItem(const std::vector<double>& curr_st, const int action_pos, const double reward, const std::vector<double>& next_st, bool terminate)
    :   reward(reward), action_pos(action_pos), terminate(terminate)
    {
        copy_state(this->curr_st, curr_st);
        copy_state(this->next_st, next_st);
    };

    ~BufferItem() {};

    inline const std::vector<double>& get_curr_st() { return curr_st; };

    inline const std::vector<double>& get_next_st() { return next_st; };

    inline const std::vector<std::string>& get_actions_avail() { return actions_avail; };

    inline int get_action_pos() { return action_pos; };

    inline double get_reward() { return reward; };

    inline bool get_terminate() { return terminate; };
};
