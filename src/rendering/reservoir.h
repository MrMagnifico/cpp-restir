#pragma once
#ifndef _RESERVOIR_H_
#define _RESERVOIR_H_

#include <ray_tracing/bvh_interface.h>
#include <utils/common.h>

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <framework/ray.h>

#include <limits>
#include <span>
#include <vector>

struct LightSample {
    glm::vec3 position  = {0.0f, 0.0f, 0.0f},
              color     = {0.0f, 0.0f, 0.0f};
};

struct SampleData {
    LightSample lightSample;
    float outputWeight = 0.0f;
};

struct Reservoir {
    Reservoir(size_t numSamples) : outputSamples(numSamples) {}

    // Intersection position info
    Ray cameraRay;
    HitInfo hitInfo;

    // Light sampling
    std::vector<SampleData> outputSamples;   
    size_t numSamples   = 1ULL;                                 // Avoid division by zero issues
    float wSum          = std::numeric_limits<float>::min();    // Avoid division by zero issues

    void update(LightSample sample, float weight);

    /**
     * Combine a number of reservoirs in a single final reservoir in a biased fashion (Algorithm 5 in ReSTIR paper)
     * 
     * @param reservoirs The reservoirs to be combined
     * @param finalReservoir Struct where final combined reservoir data will be stored. Should have the intersection position info of the relevant pixel
     * @param features Features configuration
    */
    static void combineBiased(const std::span<Reservoir>& reservoirStream, Reservoir& finalReservoir, const Features& features);

    /**
     * Combine a number of reservoirs in a single final reservoir in an unbiased fashion (Algorithm 6 in ReSTIR paper)
     * 
     * @param reservoirs The reservoirs to be combined
     * @param finalReservoir Struct where final combined reservoir data will be stored. Should have the intersection position info of the relevant pixel
     * @param features Features configuration
    */
    static void combineUnbiased(const std::span<Reservoir>& reservoirStream, Reservoir& finalReservoir, const BvhInterface& bvh, const Features& features);
};

using ReservoirGrid = std::vector<std::vector<Reservoir>>;

float targetPDF(const LightSample& sample, const Ray& cameraRay, const HitInfo& hitInfo, const Features& features);

#endif
