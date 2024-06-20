#pragma once
#include <ray_tracing/embree_interface.h>
#include <rendering/reservoir.h>
#include <rendering/shading.h>
#include <scene/scene.h>
#include <ui/draw.h>
#include <utils/config.h>

constexpr glm::vec3 CAMERA_RAY_HIT_COLOR        = {0.0f, 1.0f, 0.0f};
constexpr glm::vec3 CAMERA_RAY_NO_HIT_COLOR     = {1.0f, 0.0f, 0.0f};

// Light samplers
void sampleSegmentLight(const SegmentLight& segmentLight, glm::vec3& position, glm::vec3& color);
void sampleParallelogramLight(const ParallelogramLight& parallelogramLight, glm::vec3& position, glm::vec3& color);

// ReSTIR per-pixel canonical samples
Reservoir genCanonicalSamples(const Scene& scene, const EmbreeInterface& embreeInterface, const Features& features, Ray ray);
