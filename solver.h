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


double sqr(double x) {
    return x * x;
}


struct Block {
    int x1, y1, x2, y2;
    int area;
    int pop;

    double center_x;
    double center_y;
    double variance_x;
    double variance_y;

    double stdev_x;
    double stdev_y;

    double grad_x;
    double grad_y;

    double variance() const {
        return variance_x + variance_y;
    }

    int rect_area() const {
        return (x2 - x1) * (y2 - y1);
    }

    void transpose() {
        swap(x1, y1);
        swap(x2, y2);
        swap(center_x, center_y);
        swap(variance_x, variance_y);
        swap(stdev_x, stdev_y);
        swap(grad_x, grad_y);
    }

    bool touch(const Block &other) const {
        if (x1 == other.x1 &&
            y1 == other.y1 &&
            x2 == other.x2 &&
            y2 == other.y2)
            return false;
        return
            x2 >= other.x1 &&
            x1 <= other.x2 &&
            y2 >= other.y1 &&
            y1 <= other.y2;
    }
};


vector<Block> find_neighbors(const Block &b, const vector<Block> &blocks) {
    vector<Block> result;
    copy_if(blocks.begin(), blocks.end(), back_inserter(result),
        [&](const Block &q) {
            assert(q.touch(b) == b.touch(q));
            return q.touch(b);
        });
    return result;
}

double estimate_density_gradient_x(const Block &b, const vector<Block> &neighbors) {
    double base_density = 1.0 * b.pop / b.area;

    double covar = 0;
    double var = 0;

    for (const auto &q : neighbors) {
        double delta_x = q.center_x - b.center_x;
        double delta_density = 1.0 * q.pop / q.area - base_density;

        double discount = 1.0 / sqrt(sqr(delta_x) + sqr(q.center_y - b.center_y));
        delta_x *= discount;
        delta_density *= discount;

        // debug2(delta_x, delta_density);

        covar += delta_x * delta_density;
        var += sqr(delta_x);
    }

    var += sqr(b.stdev_x);

    if (var > 1e-6)
        return covar / var * b.stdev_x / b.pop * b.area;
    return 0.0;
}

double estimate_density_gradient_y(Block b, vector<Block> neighbors) {
    b.transpose();
    for (auto &n : neighbors)
        n.transpose();
    return estimate_density_gradient_x(b, neighbors);
}

void update_all_gradients(vector<Block> &blocks) {
    for (auto &b : blocks) {
        auto neighbors = find_neighbors(b, blocks);
        b.grad_x = estimate_density_gradient_x(b, neighbors);
        b.grad_y = estimate_density_gradient_y(b, neighbors);
    }
}

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


struct Split {
    Block parent;
    Block child1;
    Block child2;

    double bets;

    Split(const Block &parent, const Block &child1, const Block &child2)
        : parent(parent), child1(child1), child2(child2) {
    }

    void dump_datapoint(
            double slope, double expected_effect, double actual_effect) const {
        Split(*this).destructive_dump_datapoint(
            slope, expected_effect, actual_effect);
    }

    void canonicalize_for_evaluation() {
        if (child1.y2 == child2.y1) {
            child1.transpose();
            child2.transpose();
            parent.transpose();
        }
        assert(child1.x2 == child2.x1);

        bool sw = child1.area > child2.area;
        if (sw) {
            // TODO: mirror gradient
            swap(child1, child2);
        }
    }

    void destructive_dump_datapoint(
            double slope, double expected_effect, double actual_effect) {

        canonicalize_for_evaluation();

        cerr << "## split[] = {\"hz\": " << 0
             // << ", \"parent.area\": " << parent.area
             << ", \"parent.pop\": " << parent.pop
             << ", \"child1.area\": " << child1.area
             << ", \"child2.area\": " << child2.area
             << ", \"parent.stdev_x\": " << parent.stdev_x
             << ", \"parent.stdev_y\": " << parent.stdev_y
             << ", \"child1.stdev_x\": " << child1.stdev_x
             << ", \"child1.stdev_y\": " << child1.stdev_y
             << ", \"child2.stdev_x\": " << child2.stdev_x
             << ", \"child2.stdev_y\": " << child2.stdev_y
             << ", \"actual_effect\": " << actual_effect
             << ", \"expected_effect\": " << expected_effect
             << ", \"bets\": " << bets
             << ", \"slope\": " << slope
             << "}" << endl;
    }

    double predictor(double slope) const {
        return Split(*this).destructive_predictor(slope);
    }

    double destructive_predictor(double slope) {
        canonicalize_for_evaluation();
        return 0;
        //#include "predictor.h"
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
        block.stdev_x = sqrt(block.variance_x);

        block.center_y = sy / a;
        block.variance_y =
            R(y2_land)
            - 2.0 * sy * block.center_y
            + 1.0 * block.area * block.center_y * block.center_y;
        block.variance_y /= a;
        block.stdev_y = sqrt(block.variance_y);

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

        double grad;
        if (b1.x2 == b2.x1) {
            grad = s.parent.grad_x;
        } else {
            assert(b1.y2 == b2.y1);
            grad = s.parent.grad_y;
        }
        //grad = grad * abs(grad);
        //debug(grad);
        //assert(abs(grad) < 0.5);
        //grad *= 0.9;
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
                 pow(b1.area, rank_split_beta) * pow(max(1 - grad, 0.5), 4)
               - pow(b2.variance_x + b2.variance_y, rank_split_alpha) *
                 pow(b2.area, rank_split_beta) * pow(max(1 + grad, 0.5), 4)
               + 0.01*uniform_real_distribution<double>()(gen);
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

        //double area_fraction = 1.0 * b1.area / b.area;
        //double pop_fraction = 1.0 * b1.pop / b.pop;
        /*if (pop_fraction > area_fraction) {
            area_fraction = 1.0 - area_fraction;
            pop_fraction = 1.0 - pop_fraction;
        }*/

        //assert(b2.pop == query_region(b2.x1, b2.y1, b2.x2, b2.y2));

        //return {b, b1, b2};
    }

    double expected_split_effect(
        const Split& split, double slope) const {

        const Block& block = split.parent;
        const Block& b1 = split.child1;

        if (block.area <= 1)
            return -1e10;

        /*double pred = split.predictor(slope);
        if (pred < 0)
            pred = 0;
        if (pred > block.area)
            pred = block.area;
        return pred;*/

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

    static vector<Block> apply_split(
            const vector<Block> blocks, const Split &split) {
        vector<Block> result = blocks;
        auto it = std::remove_if(result.begin(), result.end(),
                [=](const Block &b) {
                    return b.x1 == split.parent.x1 &&
                           b.y1 == split.parent.y1 &&
                           b.x2 == split.parent.x2 &&
                           b.y2 == split.parent.y2;
                });
        assert(it + 1 == result.end());
        result.erase(it, result.end());

        result.push_back(split.child1);
        result.push_back(split.child2);
        return result;
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

            update_all_gradients(blocks);

            vector<Split> candidate_splits;
            for (const auto &block : blocks) {
                pick_candidate_splits(block, candidate_splits);
            }
            assert(!candidate_splits.empty());

            ///////// For training

            // for (auto candidate_split : candidate_splits) {
            //     update_split_population(candidate_split);
            //     double actual_effect = get_current_slope_and_area(
            //         apply_split(blocks, candidate_split)).second - area;
            //     candidate_split.dump_datapoint(slope, actual_effect);
            // }

            ////////////

            auto m = max_element(candidate_splits.begin(), candidate_splits.end(),
                [=](const Split &s1, const Split &s2) {
                return expected_split_effect(s1, slope) <
                       expected_split_effect(s2, slope);
            });

            Split split = *m;
            double effect = expected_split_effect(split, slope);
            debug2(area, effect);
            if (i >= 6 && effect < area * 0.004) {
                break;
            }

            // ====================
            cerr << "******" << endl;
            //auto neighbors = find_neighbors(split.parent, blocks);
            debug(split.parent);
            //debug(neighbors);
            // double grad_x =
            //     estimate_density_gradient_x(split.parent, neighbors);
            // double grad_y =
            //     estimate_density_gradient_y(split.parent, neighbors);
            debug2(split.parent.grad_x, split.parent.grad_y);
            cerr << "******" << endl;
            //-------------------

            update_split_population(split);

            blocks = apply_split(blocks, split);

            double actual_effect =
                get_current_slope_and_area(blocks).second - area;

            // TODO: take maximum with best actual effect of all candidates
            split.bets = area * 0.004;

            split.dump_datapoint(slope, effect, actual_effect);
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
                        + 1.5 * sigma * sqrt(min(k, worst));
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
