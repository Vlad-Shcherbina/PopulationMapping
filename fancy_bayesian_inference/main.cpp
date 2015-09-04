#include <iostream>
#include <vector>
#include <string>
#include <cassert>

using namespace std;

#define debug(x) cerr << #x " = " << (x) << endl;


class Population {
public:
    static int queryRegion(int x1, int y1, int x2, int y2) {
        cout << "?" << endl;
        cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
        cout.flush();
        int reply;
        cin >> reply;
        debug(x1);
        debug(y1);
        debug(x2);
        debug(y2);
        debug(reply);
        assert(reply >= 0);
        return reply;
    }
};


#define CIMG

#include "solver.h"


int main() {
    int max_percentage;
    cin >> max_percentage;
    debug(max_percentage);

    int h;
    cin >> h;
    debug(h);

    vector<string> world_map;
    for (int i = 0; i < h; i++) {
        world_map.emplace_back();
        cin >> world_map.back();
    }
    int w= world_map.front().size();
    debug(w);

    int total_population;
    cin >> total_population;
    debug(total_population);

    auto result = PopulationMapping().mapPopulation(
        max_percentage, world_map, total_population);
    assert(result.size() == h);
    cout << result.size() << endl;
    for (const auto &row : result)
        cout << row << endl;
    return 0;
}
