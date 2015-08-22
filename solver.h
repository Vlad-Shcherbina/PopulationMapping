#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <random>
#define _USE_MATH_DEFINES
#include <cmath>
#include <utility>
#include <functional>
#include <algorithm>
#include <map>
#include <stdint.h>


#include "pretty_printing.h"


using namespace std;


#define debug(x) cerr << #x " = " << (x) << endl;
#define debug2(x, y) cerr << #x " = " << (x) << ", " #y " = " << y << endl;


struct Block {
    int x1, y1, x2, y2;
    int area;
    int pop;

    double center_x;
    double center_y;
    double variance_x;
    double variance_y;

    int rect_area() const {
        return (x2 - x1) * (y2 - y1);
    }
};


std::ostream& operator<<(std::ostream &out, const Block &b) {
  out << "Block(";
  out << b.x1 << ", " << b.y1 << ", " << b.x2 << ", " << b.y2;
  out << ", area=" << b.area;
  out << ", pop=" << b.pop;
  out << ")";
  return out;
}


struct CumulativeRectangle {
    vector<vector<int64_t>> s;

    CumulativeRectangle() {}

    template<typename Fn>
    CumulativeRectangle(int w, int h, Fn f) :
        s(h + 1, vector<int64_t>(w + 1, 0)) {

        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                s[i + 1][j + 1] = f(j, i);

        for (int i = 0; i <= h; i++)
            for (int j = 0; j < w; j++)
                s[i][j + 1] += s[i][j];

        for (int i = 0; i < h; i++)
            for (int j = 0; j <= w; j++)
                s[i + 1][j] += s[i][j];
    }

    int64_t get(int x1, int y1, int x2, int y2) const {
        return s[y2][x2] + s[y1][x1] - s[y1][x2] - s[y2][x1];
    }
};


double sqr(double x) {
    return x * x;
}

class PopulationMapping {
public:
    int W;
    int H;
    int num_queries;

    vector<vector<bool>> land;
    CumulativeRectangle land_count;
    CumulativeRectangle x_land;
    CumulativeRectangle x2_land;
    CumulativeRectangle y_land;
    CumulativeRectangle y2_land;

    Block create_block(int x1, int y1, int x2, int y2, int pop) const {
        Block block;
        block.x1 = x1;
        block.y1 = y1;
        block.x2 = x2;
        block.y2 = y2;
        block.pop = pop;

        auto R = [=](const CumulativeRectangle &cr) {
            return cr.get(x1, y1, x2, y2);
        };

        block.area = R(land_count);

        double a = max(block.area, 1);
        double sx = R(x_land);
        double sy = R(y_land);

        block.center_x = sx / a;
        block.variance_x =
            R(x2_land)
            - 2.0 * sx * block.center_x
            + 1.0 * block.area * block.center_x * block.center_x;
        block.variance_x /= a;

        block.center_y = sy / a;
        block.variance_y =
            R(y2_land)
            - 2.0 * sy * block.center_y
            + 1.0 * block.area * block.center_y * block.center_y;
        block.variance_y /= a;

        return block;
    }

    typedef pair<Block, Block> Split;
    vector<Split> find_splits(const Block &b) const {
        vector<Split> result;
        Block b1, b2;

        //if (b.x2 - b.x1 > b.y2 - b.y1) {
        debug2(b.center_x, b.center_y);
        debug2(b.variance_x, b.variance_y);
        if (b.variance_x > b.variance_y) {
            for (int x = b.x1 + 1; x < b.x2 - 1; x++) {
                b1 = create_block(b.x1, b.y1, x, b.y2, 0);
                b2 = create_block(x, b.y1, b.x2, b.y2, 0);
                result.emplace_back(b1, b2);
            }
        } else {
            for (int y = b.y1 + 1; y < b.y2 - 1; y++) {
                b1 = create_block(b.x1, b.y1, b.x2, y, 0);
                b2 = create_block(b.x1, y, b.x2, b.y2, 0);
                result.emplace_back(b1, b2);
            }
        }
        return result;
    }

    double rank_split(const Split &s) const {
        Block b1 = s.first;
        Block b2 = s.second;
        //return 1.0 * b1.rect_area() * b2.rect_area();
        //return 1.0 * b1.area * b2.area;

        double area_split = 1.0 * b1.area / (b1.area + b2.area);
        double rect_area_split = 1.0 * b1.rect_area() / (b1.rect_area() + b2.rect_area());

        area_split = sqr(area_split - 0.5);
        rect_area_split = sqr(rect_area_split - 0.5);
        //return - rect_area_split - 5 * area_split;

        return - sqr(b1.variance_x + b1.variance_y) * b1.area
               - sqr(b2.variance_x + b2.variance_y) * b2.area;
    }

    Split split_block(const Block &b) {
        auto splits = find_splits(b);
        //debug(splits);
        auto ps = max_element(splits.begin(), splits.end(),
            [this](const Split &s1, const Split &s2) {
                return rank_split(s1) < rank_split(s2);
            });
        debug(ps - splits.begin());

        Block b1, b2;
        b1 = ps->first;
        b2 = ps->second;

        b1.pop = query_region(b1.x1, b1.y1, b1.x2, b1.y2);
        b2.pop = b.pop - b1.pop;

        //assert(b2.pop == query_region(b2.x1, b2.y1, b2.x2, b2.y2));

        return {b1, b2};
    }

    vector<string> mapPopulation(
        int max_percentage,
        vector<string> world_map,
        int total_population) {

        num_queries = 0;
        int max_pop = total_population * max_percentage / 100;

        H = world_map.size();
        W = world_map.front().size();
        for (const auto &row : world_map)
            assert(row.size() == W);

        land = vector<vector<bool>>(H, vector<bool>(W, false));
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                if (world_map[i][j] == 'X')
                    land[i][j] = true;

        land_count = CumulativeRectangle(W, H, [&](int x, int y) {
            return land[y][x] ? 1 : 0;
        });

        x_land = CumulativeRectangle(W, H, [&](int x, int y) {
            return land[y][x] ? x : 0;
        });
        x2_land = CumulativeRectangle(W, H, [&](int x, int y) {
            return land[y][x] ? x * x : 0;
        });
        y_land = CumulativeRectangle(W, H, [&](int x, int y) {
            return land[y][x] ? y : 0;
        });
        y2_land = CumulativeRectangle(W, H, [&](int x, int y) {
            return land[y][x] ? y * y : 0;
        });

        vector<Block> blocks;
        blocks.push_back(create_block(0, 0, W, H, total_population));

        for (int i = 0; i < 22; i++) {
            auto m = max_element(blocks.begin(), blocks.end(),
                [](const Block &b1, const Block &b2) {
                return b1.pop < b2.pop;
            });
            Block b = *m;
            blocks.erase(m);
            auto s = split_block(b);
            blocks.push_back(s.first);
            blocks.push_back(s.second);
        }

        sort(blocks.begin(), blocks.end(),
            [](const Block &b1, const Block &b2) {
                return (int64_t)b1.pop * b2.area < (int64_t)b2.pop * b1.area;
            });

        vector<string> result(H, string(W, 'X'));
        int pop = 0;
        double lerp_area = 0;
        for (const auto &b : blocks) {
            if (pop >= max_pop) {
                for (int y = b.y1; y < b.y2; y++)
                    for (int x = b.x1; x < b.x2; x++)
                        result[y][x] = '.';
            } else if (pop + b.pop > max_pop) {
                vector<pair<int, int>> lands;
                for (int y = b.y1; y < b.y2; y++)
                    for (int x = b.x1; x < b.x2; x++)
                        if (land[y][x])
                            lands.emplace_back(x, y);
                assert(lands.size() == b.area);
                default_random_engine gen;
                shuffle(lands.begin(), lands.end(), gen);

                // TODO: more thorough estimate!!!
                int k;
                double sigma = 8 * (W + H) / sqrt(3);
                for (k = lands.size(); k >= 1; k--) {
                    double q = (double)k * b.pop / b.area
                        + 2.0 * sigma * sqrt(min(k, (int)lands.size() - k));
                    if (q < max_pop - pop)
                        break;
                }

                for (int i = k; i < lands.size(); i++)
                    result[lands[i].second][lands[i].first] = '.';
                lerp_area += 1.0 * (max_pop - pop) * b.area / b.pop;
            } else {
                lerp_area += b.area;
            }

            pop += b.pop;
        }

        debug(max_pop);

        cerr << "Perfect score: " << lerp_area * pow(0.996, num_queries) << endl;

        return result;
    }

    int query_region(int x1, int y1, int x2, int y2) {
        num_queries++;
        assert(0 <= x1);
        assert(x1 < x2);
        assert(x2 <= W);
        assert(0 <= y1);
        assert(y1 < y2);
        assert(y2 <= H);
        return Population::queryRegion(x1, y1, x2 - 1, y2 - 1);
    }
};
