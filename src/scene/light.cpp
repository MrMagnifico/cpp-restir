#include <rendering/reservoir.h>
#include <scene/light.h>
#include <scene/texture.h>
#include <utils/config.h>
#include <utils/utils.h>

// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/geometric.hpp>
#include <glm/gtx/transform.hpp>
DISABLE_WARNINGS_POP()
#include <cmath>
#include <random>
#include <vector>


void sampleSegmentLight(const SegmentLight& segmentLight, glm::vec3& position, glm::vec3& color) {
    float segFrac   = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, 1.0f);
    position        = glm::mix(segmentLight.endpoint0, segmentLight.endpoint1, segFrac);
    color           = glm::mix(segmentLight.color0, segmentLight.color1, segFrac);
}

void sampleParallelogramLight(const ParallelogramLight& parallelogramLight, glm::vec3& position, glm::vec3& color) {
    float axOneFrac = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, 1.0f);
    float axTwoFrac = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, 1.0f);
    position            = parallelogramLight.v0 + (axOneFrac * parallelogramLight.edge01) + (axTwoFrac * parallelogramLight.edge02);
    glm::vec3 linLerp01 = glm::mix(parallelogramLight.color0, parallelogramLight.color1, axOneFrac);
    glm::vec3 linLerp23 = glm::mix(parallelogramLight.color2, parallelogramLight.color3, axOneFrac);
    color               = glm::mix(linLerp01, linLerp23, axTwoFrac);
}

void sampleDiskLight(const DiskLight& diskLight, glm::vec3& position, glm::vec3& color) {
    glm::vec3 planeVector;
    if (diskLight.normal.x == 0.0f) { planeVector = glm::normalize(glm::cross(diskLight.normal, glm::vec3(1.0f, 0.0f, 0.0f))); }
    else                            { planeVector = glm::normalize(glm::cross(diskLight.normal, glm::vec3(0.0f, 0.0f, 1.0f))); }
    float randomAngle           = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, 360.0f);
    glm::mat3 rotation          = glm::rotate(randomAngle, diskLight.normal);
    float distanceFromCenter    = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, diskLight.radius);
    position                    = diskLight.position + (distanceFromCenter * rotation * planeVector);
    color                       = diskLight.color;
}

Reservoir genCanonicalSamples(const Scene& scene, const EmbreeInterface& embreeInterface, const Features& features, const RayHit& rayHit) {
    // Commit primary hit info to reservoir
    Reservoir reservoir(features.numSamplesInReservoir);
    reservoir.cameraRay = rayHit.ray;
    reservoir.hitInfo   = rayHit.hit;
    
    // No lights to sample, just return
    if (scene.lights.size() == 0UL) { return reservoir; }

    // Uniform selection of light sources
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, scene.lights.size() - 1UL);

    // Compute intersection point properties
    glm::vec3 intersectionPosition  = reservoir.cameraRay.origin + (reservoir.cameraRay.t * reservoir.cameraRay.direction);
    glm::vec3 diffuseColor          = diffuseAlbedo(reservoir.hitInfo, features);

    // Zero out cautionary one sample for zero division avoidance
    reservoir.numSamples = 0ULL;

    // Obtain initial light samples
    for (uint32_t sampleIdx = 0U; sampleIdx < features.initialLightSamples; sampleIdx++) {
        // Generate sample
        LightSample sample;
        const auto& light = scene.lights[distr(gen)];
        if (std::holds_alternative<PointLight>(light)) {
            const PointLight pointLight = std::get<PointLight>(light);
            sample.position             = pointLight.position;
            sample.color                = pointLight.color;
        } else if (std::holds_alternative<SegmentLight>(light)) {
            const SegmentLight segmentLight = std::get<SegmentLight>(light);
            sampleSegmentLight(segmentLight, sample.position, sample.color);
        } else if (std::holds_alternative<ParallelogramLight>(light)) {
            const ParallelogramLight parallelogramLight = std::get<ParallelogramLight>(light);
            sampleParallelogramLight(parallelogramLight, sample.position, sample.color);
        } else if (std::holds_alternative<DiskLight>(light)) {
            const DiskLight diskLight = std::get<DiskLight>(light);
            sampleDiskLight(diskLight, sample.position, sample.color);
        }

        // Update reservoir
        float sampleWeight = targetPDF(sample, reservoir.cameraRay, reservoir.hitInfo, features) / (1.0f / static_cast<float>(scene.lights.size())); // We uniformly sample all lights, so distribution PDF is uniform
        reservoir.update(sample, sampleWeight); 
    }

    // Set output weight and do optional visibility check
    for (SampleData& reservoirSample : reservoir.outputSamples) {
        if (features.initialSamplesVisibilityCheck && !testVisibilityLightSample(reservoirSample.lightSample.position, embreeInterface, features, reservoir.cameraRay, reservoir.hitInfo)) { reservoirSample.outputWeight = 0.0f; }
        else {
            float pdfValue = targetPDF(reservoirSample.lightSample, reservoir.cameraRay, reservoir.hitInfo, features);
            if (pdfValue == 0.0f)   { reservoirSample.outputWeight  = 0.0f; }
            else                    { reservoirSample.outputWeight  = (1.0f / pdfValue) * 
                                                                      (1.0f / reservoir.numSamples) *
                                                                      reservoir.wSum; }
        }
    }
    
    // Draw debug ray and return
    drawRay(reservoir.cameraRay, CAMERA_RAY_HIT_COLOR);
    return reservoir;
}
