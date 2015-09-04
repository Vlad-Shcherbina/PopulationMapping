#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <random>
#include <tuple>
#define _USE_MATH_DEFINES
#include <cmath>
#include <utility>
#include <functional>
#include <algorithm>
#include <map>


#include "pretty_printing.h"


using namespace std;

#ifdef CIMG
#include "CImg.h"
using namespace cimg_library;
#endif

#define debug(x) cerr << #x " = " << (x) << endl;
#define debug2(x, y) cerr << #x " = " << (x) << ", " #y " = " << y << endl;


//typedef pair<int, int> Coord;

typedef int PackedCoord;

int pack_coord(int x, int y) {
    assert(x >= 0);
    assert(y >= 0);
    return (y << 16) + x;
}

int unpack_x(PackedCoord p) {
    return p & 0xffff;
}

int unpack_y(PackedCoord p) {
    return p >> 16;
}

int packed_distance(PackedCoord p1, PackedCoord p2) {
    int dx = unpack_x(p1) - unpack_x(p2);
    int dy = unpack_y(p1) - unpack_y(p2);
    return abs(dx) + abs(dy);
}

double observation_log_likelihood(
    int pop, double sum_densities, double sum_squared_densities) {

    double mean = 0.5 * sum_densities;
    double sigma2 = sum_squared_densities / 12.0;

    return - (pop - mean) * (pop - mean) / (2.0 * sigma2)
           - 0.5 * log(2 * M_PI * sigma2);
}


class PopulationMapping;


struct Observation {
    int x1, y1, x2, y2;

    int center_x, center_y;
    PackedCoord center;

    int num_land;
    int pop;

    int w, h;

    vector<vector<PackedCoord>> land_by_distance;
    vector<int> cumulative_land_area;

    Observation(
        int x1, int y1, int x2, int y2, int pop, const PopulationMapping &pm);

    void precompute(const PopulationMapping &pm);

    #ifdef CIMG
    void draw(CImg<unsigned char> &img) const {
        const unsigned char YELLOW[] = {255, 255, 0};
        img.draw_rectangle(x1, y1, x2, y2, YELLOW);

        img.draw_rectangle(x2, y2, x2 + 60, y2, YELLOW);
    }
    #endif
};


// Represents distribution for a given observation and given city size.
struct SizedBell {
    // TODO: This could be a pointer to vector element. Risky.
    const Observation *observation;
    int size;

    vector<double> log_likelihood;
    vector<double> cumulative_likelihood;
    int max_likelihood_dist;

    int truncated_dist;
    double truncated_likelihood;

    double total_truncated_likelihood;
    double total_regular_likelihood;

    int w_plus_h() const {
        return observation->w + observation->h;
    }

    SizedBell(const Observation &obs, int size)
        : observation(&obs), size(size) {

        log_likelihood.resize(w_plus_h() - 1);

        cumulative_likelihood.resize(w_plus_h());
        cumulative_likelihood[0] = 0;

        max_likelihood_dist = 0;

        for (int d = 0; d < log_likelihood.size(); d++) {

            double center_distance = max(1.0 * d / size, 1.0);
            double density = w_plus_h() * 8 / center_distance;

            log_likelihood[d] = observation_log_likelihood(
                observation->pop,
                observation->num_land * density,
                observation->num_land * density * density);

            cumulative_likelihood[d + 1] =
                cumulative_likelihood[d] +
                exp(log_likelihood[d]) * observation->land_by_distance[d].size();

            if (log_likelihood[d] > log_likelihood[max_likelihood_dist]) {
                max_likelihood_dist = d;
            }
        }

        set_truncation(log_likelihood.size(), 1e-100);
    }

    void set_truncation(int dist, double likelihood) {
        dist = min(dist, (int)log_likelihood.size());
        assert(dist >= 0);
        assert(dist <= log_likelihood.size());
        truncated_dist = dist;
        truncated_likelihood = likelihood;

        int truncated_land_area =
            observation->cumulative_land_area.back() -
            observation->cumulative_land_area[truncated_dist];
        total_truncated_likelihood = truncated_likelihood * truncated_land_area;
        total_regular_likelihood = cumulative_likelihood[truncated_dist];
    }

    double total_likelihood() const {
        return total_regular_likelihood + total_truncated_likelihood;
    }

    double max_log_likelihood() const {
        if (max_likelihood_dist < truncated_dist)
            return log_likelihood[max_likelihood_dist];
        else
            return log(truncated_likelihood);
    }

    double get_log_likelihood(int dist) const {
        if (dist < truncated_dist)
            return log_likelihood[dist];
        else
            return log(truncated_likelihood);
    }

    template<typename RndGen>
    PackedCoord random_sample(RndGen &gen) const {
        uniform_real_distribution<double> urd;

        if (urd(gen) * total_likelihood() < total_regular_likelihood) {
            double x = urd(gen) * total_regular_likelihood;
            int lo = 0;
            int hi = truncated_dist;
            // TODO: what if it's empty
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x < cumulative_likelihood[mid])
                    hi = mid;
                else
                    lo = mid;
            }
            assert(hi - lo == 1);
            const auto &lands = observation->land_by_distance[lo];
            // TODO: what if it happens because of rounding error?
            assert(!lands.empty());
            return lands[uniform_int_distribution<int>(0, lands.size() - 1)(gen)];

        } else {  // truncated part
            // TODO: what if it's empty
            const auto &cla = observation->cumulative_land_area;
            int x = uniform_int_distribution<int>(
                cla[truncated_dist],
                cla.back() - 1)(gen);

            int lo = truncated_dist;
            int hi = cla.size() - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x < cla[mid]) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            assert(hi - lo == 1);
            const auto &lands = observation->land_by_distance[lo];
            // TODO: what if this is empty (this really shouldn't happen)
            return lands.at(x - cla[lo]);
        }
    }
};


struct AllBells {
    vector<SizedBell> bells;
    mutable discrete_distribution<int> distr;

    AllBells(const Observation &obs) {
        vector<double> bell_probs;
        for (int size = 1; size <= 5; size++) {
            bells.push_back(SizedBell(obs, size));
            bell_probs.push_back(bells.back().total_likelihood());
        }
        // TODO: what if they all are zeros?
        distr = discrete_distribution<int>(
            bell_probs.begin(), bell_probs.end());
    }

    void clear_truncation() {
        for (auto &bell : bells)
            bell.set_truncation(bell.log_likelihood.size(), 1e-100);
    }

    void update_truncation(int size, int pt) {
        assert(size >= 1);
        assert(size <= 5);

        int distance = packed_distance(pt, bells.front().observation->center);

        double likelihood = exp(bells.at(size - 1).log_likelihood[distance]);

        for (int s = 1; s <= 5; s++) {
            int new_distance = (distance * s + size / 2) / size;
            if (new_distance < bells[s - 1].truncated_dist)
                bells[s - 1].set_truncation(new_distance, likelihood);
        }
    }

    // Return pair (size, packed_coord)
/*    template<typename RndGen>
    pair<int, PackedCoord> random_sample(RndGen &gen) const {
        int size = 1 + distr(gen);
        return {size, bells[size - 1].random_sample(gen)};
    }*/
};


struct PosteriorDrawer {
    vector<AllBells> abs;
    map<int, int> primary_by_size;
    mutable discrete_distribution<int> size_distr;

    mutable int num_samples;
    mutable int cnt;

    PosteriorDrawer() {
        num_samples = 0;
        cnt = 0;
    }

    void prepair() {
        if (abs.empty())
            return;

        int total_land = abs[0].bells[0].observation->cumulative_land_area.back();
        primary_by_size.clear();

        for (int size = 1; size <= 5; size++) {
            primary_by_size[size] = 0;
            double best_acceptance_rate = 1e200;
            for (int i = 0; i < abs.size(); i++) {
                const auto &bell = abs[i].bells[size - 1];
                double acceptance_rate =
                    bell.total_likelihood() / (exp(bell.max_log_likelihood()) * total_land);
                //debug2(size, i);
                //debug(acceptance_rate);
                if (acceptance_rate < best_acceptance_rate) {
                    primary_by_size[size] = i;
                    best_acceptance_rate = acceptance_rate;
                }
            }
        }

        // debug(primary_by_size);

        vector<double> weight_by_size;
        for (int size = 1; size <= 5; size++) {
            int p = primary_by_size[size];
            double w = abs[p].bells[size - 1].total_likelihood();
            for (int i = 0; i < abs.size(); i++) {
                if (i == p)
                    continue;
                const auto &bell = abs[i].bells[size - 1];
                w *= exp(bell.max_log_likelihood()) * total_land;
            }
            weight_by_size.push_back(w);
        }
        // debug(weight_by_size);

        size_distr = discrete_distribution<int>(
            weight_by_size.begin(), weight_by_size.end());
    }

    // Return pair (size, packed_coord)
    template<typename RndGen>
    pair<int, PackedCoord> random_sample(
            RndGen &gen, double rejection_strength=1.0) const {
        uniform_real_distribution<double> urd;
        while (true) {
            cnt++;
            int size = 1 + size_distr(gen);
            PackedCoord pt =
                abs[primary_by_size.at(size)].bells[size - 1].random_sample(gen);

            double log_acceptance_rate = 0.0;
            for (int i = 0; i < abs.size(); i++) {
                if (i == primary_by_size.at(size))
                    continue;
                const auto &bell = abs[i].bells[size - 1];

                log_acceptance_rate += bell.get_log_likelihood(
                    packed_distance(pt, bell.observation->center));
                log_acceptance_rate -= bell.max_log_likelihood();
            }

            if (urd(gen) < exp(log_acceptance_rate * rejection_strength)) {
                //debug(cnt);
                num_samples++;
                return {size, pt};
            }
        }
    }
};


struct GibbsSampler {
    PosteriorDrawer pd;
    vector<pair<int, PackedCoord> > current_sample;

    GibbsSampler(const PosteriorDrawer &pd, int n) : pd(pd) {
        static default_random_engine gen;

        uniform_int_distribution<int> size_distr(1, 5);
        // TODO: use actual map size
        uniform_int_distribution<int> x_distr(0, 40);
        uniform_int_distribution<int> y_distr(0, 40);
        for (int i = 0; i < n; i++) {
            current_sample.emplace_back(
                size_distr(gen),
                pack_coord(x_distr(gen), y_distr(gen)));
        }
    }

    void update(double rejection_strength=1.0) {
        static default_random_engine gen;
        for (int i = 0; i < current_sample.size(); i++) {
            for (auto &ab : pd.abs) {
                ab.clear_truncation();
            }
            for (int j = 0; j < current_sample.size(); j++) {
                if (i == j)
                    continue;
                int size = current_sample[j].first;
                PackedCoord pt = current_sample[j].second;
                for (auto &ab : pd.abs) {
                    ab.update_truncation(size, pt);
                }
            }
            pd.prepair();
            current_sample[i] = pd.random_sample(gen, rejection_strength);
        }
    }

    void warmup() {
        cerr << "warmup" << endl;
        for (double f = 0.00001; f < 1.0; f *= 1.5) {
            update(f);
            debug(f);
        }
    }
};


class PopulationMapping {
public:
    int W;
    int H;

    vector<vector<bool>> land;
    int count_land(int x1, int y1, int x2, int y2) const {
        // TODO: speedup by precomputing cumulative sums
        int result = 0;
        for (int i = y1; i < y2; i++)
            for (int j = x1; j < x2; j++)
                if (land[i][j])
                    result++;
        return result;
    }

    vector<string> mapPopulation(
        int max_percentage,
        vector<string> world_map,
        int total_population) {

        H = world_map.size();
        W = world_map.front().size();
        for (const auto &row : world_map)
            assert(row.size() == W);

        land = vector<vector<bool>>(H, vector<bool>(W, false));
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                if (world_map[i][j] == 'X')
                    land[i][j] = true;

        interact();

        int y = H;
        do {
            y = y * 9 / 10;
        } while (Population::queryRegion(0, 0, W - 1, y - 1) * 100 >
                 max_percentage * total_population);

        for (int i = y; i < H; i++)
            world_map[i] = string(W, '.');

        return world_map;
    }

    int pop_fn(int dx, int dy, int size) {
        int d = (abs(dx) + abs(dy)) / size;
        if (d < 1)
            d = 1;
        return (W + H) * 8 / d;
    }

    float log_likelihood(int num_land, int pop, float expected_density) {
        float sigma2 = expected_density * expected_density / 12.0 * num_land;
        float mean = 0.5 * expected_density * num_land;

        float x = pop - mean;
        return - x*x / (2*sigma2)
               - 0.5 * log(sigma2 * 2 * M_PI);
    }

    float most_likely_density(int num_land, int pop) {
        return 2.0 * pop / num_land;
    }

    vector<tuple<int, int, int>> random_draw(
        int n, std::function<float(int, int, int)> log_prob) {

        default_random_engine gen;
        uniform_int_distribution<int> x_distribution(0, W - 1);
        uniform_int_distribution<int> y_distribution(0, H - 1);
        uniform_int_distribution<int> size_distribution(1, 5);
        uniform_real_distribution<double> distribution(0.0, 1.0);

        vector<tuple<int, int, int>> result;
        int i;
        for (i = 0; ; i++) {
            int xx = x_distribution(gen);
            int yy = y_distribution(gen);
            int size = size_distribution(gen);
            if (!land[yy][xx])
                continue;
            float lp = log_prob(xx, yy, size);
            float p = exp(lp);
            //assert(p < 1.00001);
            if (distribution(gen) < p) {
                result.emplace_back(xx, yy, size);
                if (result.size() == n)
                    break;
            }
        }
        debug(i);
        return result;
    }

    Observation create_observation(int x, int y, int size) const {
        int x1 = x - size / 2;
        int y1 = y - size / 2;
        int x2 = x + (size + 1) / 2;
        int y2 = y + (size + 1) / 2;

        int pop = Population::queryRegion(x1, y1, x2 - 1, y2 - 1);

        Observation result(x1, y1, x2, y2, pop, *this);
        result.precompute(*this);
        return result;
    }

    void interact() {
        #ifdef CIMG
        CImg<unsigned char> map(W, H, 1, 3, 0);
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                if (!land[i][j])
                    map(j, i, 2) = 255;

        CImg<unsigned char> display_image = map;
        CImgDisplay display(display_image, "zzz");
        const int SCALE = 3;
        display.resize(SCALE * W, SCALE * H);

        vector<Observation> observations;

        // CImg<unsigned char> distr_img(W + H, 200, 1, 3, 0);
        // CImgDisplay display2(distr_img, "distr");

        while (!display.is_closed()) {
            if (display.is_keyBACKSPACE()) {
                //display.key();
                debug("backspace");
                if (!observations.empty())
                    observations.pop_back();
            } else if (display.button() & 1) {
                int x = display.mouse_x() / SCALE;
                int y = display.mouse_y() / SCALE;
                debug2(x, y);

                //observations.clear(); ///////

                observations.push_back(create_observation(x, y, 4));
            } else {
                display.wait();
                continue;
            }

            display_image = map;
            for (const auto &obs : observations) {
                obs.draw(display_image);
            }

            //vector<tuple<int, int, int> > candidates;
            cerr << "before all" << endl;
            default_random_engine gen;
            if (!observations.empty()) {
                PosteriorDrawer pd;
                for (const auto &obs : observations) {
                    AllBells ab(obs);
                    pd.abs.push_back(ab);
                }
                //pd.prepair();


                //ab.set_truncation(1, 20);
                GibbsSampler gs(pd, 10);
                gs.warmup();

                cerr << "sampling" << endl;
                for (int i = 0; i < 100; i++) {
                    gs.update();
                    draw_sample(display_image, gs.current_sample);
                }
                debug(1.0 * gs.pd.cnt / gs.pd.num_samples);
            }

            display_image.display(display);
            display.wait();
        }
        #endif
    }

    #ifdef CIMG
    void draw_sample(
            CImg<unsigned char> &img,
            const vector<pair<int, PackedCoord>> &sample) {
        int prev_x = -1;
        int prev_y = -1;
        for (auto city : sample) {
            int size = city.first;
            int x = unpack_x(city.second);
            int y = unpack_y(city.second);

            unsigned char color[] = {0, 0, 0};
            switch (size) {
                case 1: color[1] = 255; break;
                case 2: color[0] = color[1] = 255; break;
                case 3: color[0] = 255; break;
                case 4: color[1] = color[2] = 255; break;
                case 5: color[0] = color[1] = color[2] = 255; break;
                default: assert(false); break;
            }

            img(x, y, 0) = color[0];
            img(x, y, 1) = color[1];
            img(x, y, 2) = color[2];

            const unsigned char GRAY[] = {120, 120, 120};
            if (prev_x != -1)
                img.draw_line(prev_x, prev_y, x, y, GRAY, 0.3);

            prev_x = x;
            prev_y = y;
        }
    }
    #endif
};


Observation::Observation(
    int x1, int y1, int x2, int y2, int pop, const PopulationMapping &pm)
    : x1(x1), y1(y1), x2(x2), y2(y2), pop(pop) {

    w = pm.W;
    h = pm.H;

    num_land = 0;
    center_x = 0;
    center_y = 0;
    for (int i = y1; i < y2; i++) {
        for (int j = x1; j < x2; j++) {
            if (!pm.land[i][j])
                continue;
            num_land++;

            center_x += j;
            center_y += i;
        }
    }
    assert(num_land > 0);
    center_x += num_land / 2;
    center_x /= num_land;
    center_y += num_land / 2;
    center_y /= num_land;
    center = pack_coord(center_x, center_y);
}


void Observation::precompute(const PopulationMapping &pm) {
    assert(land_by_distance.empty());
    land_by_distance.resize(pm.H + pm.W - 1);
    for (int i = 0; i < pm.H; i++) {
        for (int j = 0; j < pm.W; j++) {
            if (!pm.land[i][j])
                continue;
            PackedCoord pt = pack_coord(j, i);
            int d = packed_distance(center, pt);

            land_by_distance[d].push_back(pt);
        }
    }

    assert(cumulative_land_area.empty());
    cumulative_land_area.resize(land_by_distance.size() + 1);

    cumulative_land_area[0] = 0;
    for (int i = 0; i < land_by_distance.size(); i++)
        cumulative_land_area[i + 1] =
            cumulative_land_area[i] + land_by_distance[i].size();
}
