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

using PrimaryHitGrid    = std::vector<std::vector<RayHit>>;
using PixelGrid         = std::vector<std::vector<glm::vec3>>;

// Auxilliary components
PrimaryHitGrid genPrimaryRayHits(const Scene& scene, const Trackball& camera, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features);
glm::vec3 finalShading(const Reservoir& reservoir, const Ray& primaryRay, const EmbreeInterface& embreeInterface, const Features& features);
void combineToScreen(Screen& screen, const PixelGrid& finalPixelColors, const Features& features);

// ReSTIR core components
ReservoirGrid genInitialSamples(const PrimaryHitGrid& primaryHits, const Scene& scene, const EmbreeInterface& embreeInterface, const Features& features, const glm::ivec2& windowResolution);
void spatialReuse(ReservoirGrid& reservoirGrid, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features);
void temporalReuse(ReservoirGrid& reservoirGrid, ReservoirGrid& previousFrameGrid, const EmbreeInterface& embreeInterface,
                   Screen& screen, const Features& features);

// Main rendering function.
ReservoirGrid renderRayTracing(std::shared_ptr<ReservoirGrid> previousFrameGrid,
                               const Scene& scene, const Trackball& camera,
                               const EmbreeInterface& embreeInterface, Screen& screen,
                               const Features& features);
