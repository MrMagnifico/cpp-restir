#pragma once
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <framework/ray.h>

#include <ray_tracing/embree_interface.h>
#include <rendering/reservoir.h>

// Forward declarations.
struct Scene;
class Screen;
class Trackball;
struct Features;

ReservoirGrid genInitialSamples(const Scene& scene, const Trackball& camera, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features);

void spatialReuse(ReservoirGrid& reservoirGrid, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features);

void temporalReuse(ReservoirGrid& reservoirGrid, ReservoirGrid& previousFrameGrid, const EmbreeInterface& embreeInterface,
                   Screen& screen, const glm::vec2 motionVector, const Features& features);

// Main rendering function.
ReservoirGrid renderRayTracing(std::shared_ptr<ReservoirGrid> previousFrameGrid,
                               const Scene& scene, const Trackball& camera,
                               const EmbreeInterface& embreeInterface, Screen& screen,
                               const glm::vec2 motionVector, const Features& features);
