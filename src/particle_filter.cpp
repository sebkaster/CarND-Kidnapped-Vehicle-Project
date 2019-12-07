/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
static std::default_random_engine gen;

const double epsilon = 0.000001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

    num_particles = 100;  // TODO: Set the number of particles

    // standard deviations
    const double std_x = std[0];
    const double std_y = std[1];
    const double std_theta = std[2];

    // normal distributions
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    // generate particles with mean on GPS values
    particles.reserve(num_particles);
    for (size_t i = 0; i < (size_t) num_particles; i++) {

        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles.emplace_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    // standard deviations
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // normal distributions
    std::normal_distribution<double> dist_x(0.0, std_x);
    std::normal_distribution<double> dist_y(0.0, std_y);
    std::normal_distribution<double> dist_theta(0.0, std_theta);

    // calculate new state
    for (auto &p : particles) {

        double theta = p.theta;

        if (fabs(yaw_rate) < epsilon) { // yaw not changing
            p.x += velocity * delta_t * cos(theta);
            p.y += velocity * delta_t * sin(theta);
        } else {
            p.x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            p.y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        // add noise
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector <LandmarkObs> predicted,
                                     vector <LandmarkObs> &observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

    if (predicted.empty()) {
        std::runtime_error("no predictions!");
    }

    for (auto &observation : observations) {

        double min_dist = dist(predicted.front().x, predicted.front().y, observation.x, observation.y);

        observation.id = predicted.front().id;
        for (size_t i = 1; i < predicted.size(); i++) {
            double cur_dist = dist(predicted[i].x, predicted[i].y, observation.x, observation.y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                observation.id = predicted[i].id;
            }
        }
        //std::cout << "id: " << observation.id << " " << observation.x << " " << observation.y << std::endl;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector <LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    int i = 0;
    weights = std::vector<double>(particles.size());
    for (auto &p: particles) {

        std::vector <LandmarkObs> in_range_landmarks;
        in_range_landmarks.reserve(map_landmarks.landmark_list.size());
        for (auto const &landmark_obj: map_landmarks.landmark_list) {
            if (dist(p.x, p.y, landmark_obj.x_f, landmark_obj.y_f) <= sensor_range) {
                in_range_landmarks.emplace_back(LandmarkObs{landmark_obj.id_i, landmark_obj.x_f, landmark_obj.y_f});
            }
        }
        in_range_landmarks.shrink_to_fit();
        std::vector <LandmarkObs> mapped_observations;
        mapped_observations.reserve(observations.size());
        for (auto const &observation : observations) {
            double x_mapped = cos(p.theta) * observation.x - sin(p.theta) * observation.y + p.x;
            double y_mapped = sin(p.theta) * observation.x + cos(p.theta) * observation.y + p.y;
            mapped_observations.emplace_back(LandmarkObs{0, x_mapped, y_mapped});
        }

        mapped_observations.shrink_to_fit();

        dataAssociation(in_range_landmarks, mapped_observations);

        // resetting weight
        p.weight = 1.0;
        for (auto const &m_obs : mapped_observations) {
            auto in_range_landmark_it = std::find_if(in_range_landmarks.begin(), in_range_landmarks.end(),
                                                     [&m_obs](LandmarkObs const &obj) {
                                                         return obj.id == m_obs.id;
                                                     });

            if (in_range_landmark_it != in_range_landmarks.end()) {
                double e_1 =
                        std::pow(m_obs.x - (*in_range_landmark_it).x, 2.0) / (2.0 * std_landmark[0] * std_landmark[0]);
                double e_2 =
                        std::pow(m_obs.y - (*in_range_landmark_it).y, 2.0) / (2.0 * std_landmark[1] * std_landmark[1]);
                double weight = (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])) *
                                exp(-(e_1 + e_2));
                weight > epsilon ? p.weight *= weight : p.weight *= epsilon;
            } else {
                std::runtime_error("ERROR");
            }
        }
        weights[i] = p.weight;
        i++;
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    double maxWeight = *max_element(std::begin(weights), std::end(weights));

    // Creating distributions.
    std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
    std::uniform_int_distribution<int> distInt(0, num_particles - 1);

    // Generating index.
    int index = distInt(gen);

    double beta = 0.0;

    // the wheel
    std::vector<Particle> resampled_particles;
    resampled_particles.reserve(particles.size());
    for (int i = 0; i < num_particles; i++) {
        beta += distDouble(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampled_particles.emplace_back(particles[index]);
    }

    particles = std::move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}