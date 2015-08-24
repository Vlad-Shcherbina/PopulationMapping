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
#include <chrono>
#include <thread>

#include "pretty_printing.h"

using namespace std;

#include "prediction.h"

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

    double variance() const {
        return variance_x + variance_y;
    }

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


struct Split {
    Block parent;
    Block child1;
    Block child2;

    Split(const Block &parent, const Block &child1, const Block &child2)
        : parent(parent), child1(child1), child2(child2) {
    }
};


class PopulationMapping {
public:
    int W;
    int H;
    int num_queries;
    int max_pop;

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

    vector<Split> find_splits(const Block &b) const {
        vector<Split> result;
        Block b1, b2;

        for (int x = b.x1 + 1; x <= b.x2 - 1; x++) {
            b1 = create_block(b.x1, b.y1, x, b.y2, 0);
            b2 = create_block(x, b.y1, b.x2, b.y2, 0);
            result.emplace_back(b, b1, b2);
        }

        for (int y = b.y1 + 1; y <= b.y2 - 1; y++) {
            b1 = create_block(b.x1, b.y1, b.x2, y, 0);
            b2 = create_block(b.x1, y, b.x2, b.y2, 0);
            result.emplace_back(b, b1, b2);
        }

        return result;
    }

    double rank_split_alpha = 2.0;
    double rank_split_beta = 1.0;

    double rank_split(const Split &s) const {
        Block b1 = s.child1;
        Block b2 = s.child2;
        //return 1.0 * b1.rect_area() * b2.rect_area();
        //return 1.0 * b1.area * b2.area;

        double area_split = 1.0 * b1.area / (b1.area + b2.area);
        double rect_area_split = 1.0 * b1.rect_area() / (b1.rect_area() + b2.rect_area());

        area_split = sqr(area_split - 0.5);
        rect_area_split = sqr(rect_area_split - 0.5);
        //return - rect_area_split - 5 * area_split;

         static default_random_engine gen;
        // return -(40.5 + uniform_real_distribution<double>()(gen)) *
        //      (b1.variance_x + b1.variance_y + b2.variance_x + b2.variance_y);

        return - pow(b1.variance_x + b1.variance_y, rank_split_alpha) *
                 pow(b1.area, rank_split_beta)
               - pow(b2.variance_x + b2.variance_y, rank_split_alpha) *
                 pow(b2.area, rank_split_beta) + 0.1*uniform_real_distribution<double>()(gen);
    }

    void pick_candidate_splits(const Block &b, vector<Split> &results) const {
        auto splits = find_splits(b);
        if (splits.empty()) {
            return;
        }

        assert(!splits.empty());
        auto ps = max_element(splits.begin(), splits.end(),
            [this](const Split &s1, const Split &s2) {
                return rank_split(s1) < rank_split(s2);
            });
        results.push_back(*ps);
    }

    void update_split_population(Split &split) {
        const Block &b = split.parent;
        Block &b1 = split.child1;
        Block &b2 = split.child2;

        b1.pop = query_region(b1.x1, b1.y1, b1.x2, b1.y2);
        b2.pop = b.pop - b1.pop;

        double area_fraction = 1.0 * b1.area / b.area;
        double pop_fraction = 1.0 * b1.pop / b.pop;
        /*if (pop_fraction > area_fraction) {
            area_fraction = 1.0 - area_fraction;
            pop_fraction = 1.0 - pop_fraction;
        }*/

        cerr << "## split[] = {\"area_fraction\": " << area_fraction
             << ", \"pop_fraction\": " << pop_fraction
             << ", \"area\": " << b.area
             << ", \"pop\": " << b.pop
             << ", \"w\": " << (b.x2 - b.x1)
             << ", \"h\": " << (b.y2 - b.y1)
             << ", \"variance\": " << b.variance()
             << ", \"variance1\": " << b1.variance()
             << ", \"variance2\": " << b2.variance()
             << "}" << endl;

        //assert(b2.pop == query_region(b2.x1, b2.y1, b2.x2, b2.y2));

        //return {b, b1, b2};
    }

    double expected_split_effect(
        const Split& split, double slope) {

        const Block& block = split.parent;
        const Block& b1 = split.child1;

        if (block.area <= 1)
            return -1e10;

        double ave = 0;
        for (int j = 0; j < prediction.front().size(); j++) {

            //double dx1 = 0.5 * block.area;
            //double dy1 = 0.4 * block.pop;
            double dx1 = b1.area;
            //const vector &xs =
            int i = 1.0 * b1.area / block.area * prediction.size() + 0.5;
            if (i < 0) i = 0;
            if (i >= prediction.size()) i = prediction.size() - 1;
            const auto &xs = prediction[i];
            assert(xs.size() == prediction.front().size());
            //double dy1 = 0.4 * block.pop;
            double dy1 = xs[j] * block.pop;

            // dx1 = 0.5 * b1.area;
            // dy1 = 0.4 * block.pop;


            double dx2 = block.area - dx1;
            double dy2 = block.pop - dy1;

            if (dy1 * dx2 > dy2 * dx1) {
                swap(dx1, dx2);
                swap(dy1, dy2);
            }

            double s = 1.0 * block.pop / block.area;
            double s1 = 1.0 * dy1 / dx1;
            double s2 = 1.0 * dy2 / dx2;

            if (s1 < slope && s2 < slope) {
                return 0;
            }

            if (s1 > slope && s2 > slope) {
                return 0;
            }

            if (false && s == slope) {
                ave += 0.5 * (dx1 - dy1 / slope);
            }
            else if (s < slope) {
                double result = dy2 / slope - dx2;
                // debug(result);
                ave += result;
            } else {
                double result = - dy1 / slope + dx1;
                // debug(result);
                ave += result;
            }
        }

        ave /= prediction.front().size();
        return ave;
        //return block.pop;
    }

    pair<double, double> get_current_slope_and_area(vector<Block> blocks) const {
        sort(blocks.begin(), blocks.end(),
            [](const Block &b1, const Block &b2) {
                return (int64_t)b1.pop * b2.area < (int64_t)b2.pop * b1.area;
            });

        int pop = 0;
        double lerp_area = 0;
        for (const auto &b : blocks) {
            pop += b.pop;
            if (pop >= max_pop) {
                lerp_area += 1.0 * (max_pop - (pop - b.pop)) * b.area / b.pop;
                //debug(lerp_area);
                return {1.0 * b.pop / b.area, lerp_area};
            }
            lerp_area += b.area;
        }
        assert(false);
    }

    vector<string> mapPopulation(
        int max_percentage,
        vector<string> world_map,
        int total_population) {

        cerr << "## "; debug(rank_split_alpha);
        cerr << "## "; debug(rank_split_beta);

        num_queries = 0;
        max_pop = (int64_t)total_population * max_percentage / 100;


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

        cerr << "## "; debug(W);
        cerr << "## "; debug(H);
        cerr << "## "; debug(total_population);
        cerr << "## "; debug(max_percentage);
        int64_t land_area = land_count.get(0, 0, W, H);
        cerr << "## "; debug(land_area);

        vector<Block> blocks;
        blocks.push_back(create_block(0, 0, W, H, total_population));

        for (int i = 0; i < 200; i++) {
            auto sa = get_current_slope_and_area(blocks);
            double slope = sa.first;
            double area = sa.second;

            vector<Split> candidate_splits;
            for (const auto &block : blocks) {
                pick_candidate_splits(block, candidate_splits);
            }
            assert(!candidate_splits.empty());

            auto m = max_element(candidate_splits.begin(), candidate_splits.end(),
                [=](const Split &s1, const Split &s2) {
                return expected_split_effect(s1, slope) <
                       expected_split_effect(s2, slope);
            });

            Split s = *m;
            //Block b = *m;
            double effect = expected_split_effect(s, slope);
            debug2(area, effect);
            if (i >= 6 && effect < area * 0.004) {
                break;
            }

            update_split_population(s);
            //blocks.erase(m);
            blocks.erase(
                std::remove_if(blocks.begin(), blocks.end(),
                    [=](const Block &b) {
                        return b.x1 == s.parent.x1 &&
                               b.y1 == s.parent.y1 &&
                               b.x2 == s.parent.x2 &&
                               b.y2 == s.parent.y2;
                    }),
                blocks.end());

            //auto s = split_block(b);
            blocks.push_back(s.child1);
            blocks.push_back(s.child2);
        }
        cerr << "## "; debug(num_queries);

        sort(blocks.begin(), blocks.end(),
            [](const Block &b1, const Block &b2) {
                return (int64_t)b1.pop * b2.area < (int64_t)b2.pop * b1.area;
            });

        vector<string> result(H, string(W, 'X'));
        int64_t pop = 0;
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
                int worst = b.pop / (8 * (W + H)) + 1;
                for (k = lands.size(); k >= 1; k--) {
                    double q = (double)k * b.pop / b.area
                        + 1.0 * sigma * sqrt(min(k, worst));
                    if (q < max_pop - pop)
                        break;
                }
                debug2(k, lands.size());
                debug(b.pop / (8 * (W + H)));
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

        cerr.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
