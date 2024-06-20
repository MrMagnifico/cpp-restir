#pragma once
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <framework/ray.h>
#include <framework/mesh.h>

enum class DrawMode {
    Filled,
    Wireframe
};

enum class ViewMode {
    Rasterization = 0,
    RayTracing = 1
};

struct HitInfo {
    glm::vec3 normal;
    glm::vec3 barycentricCoord;
    glm::vec2 texCoord;
    Material material;
    uint32_t geometryId;
};

struct RayHit {
    Ray ray;
    HitInfo hit;
};

struct Plane {
    float D = 0.0f;
    glm::vec3 normal { 0.0f, 1.0f, 0.0f };
};

struct AxisAlignedBox {
    glm::vec3 lower { 0.0f };
    glm::vec3 upper { 1.0f };
};

struct Sphere {
    glm::vec3 center { 0.0f };
    float radius = 1.0f;
    Material material;
};

struct PointLight {
    glm::vec3 position;
    glm::vec3 color;
};

struct SegmentLight {
    glm::vec3 endpoint0, endpoint1; // Positions of endpoints
    glm::vec3 color0, color1; // Color of endpoints
};

struct ParallelogramLight {
    // A parallelogram light (see figure 3.14 of chapter 13.4.2 of Fundamentals of CG 4th Edition)
    glm::vec3 v0; // v0
    glm::vec3 edge01, edge02; // edges from v0 to v1, and from v0 to v2
    glm::vec3 color0, color1, color2, color3;
};

struct ExtraFeatures {
    bool enableEnvironmentMapping       = false;
    bool enableBvhSahBinning            = false;
    bool enableMotionBlur               = false;
    bool enableBloomEffect              = false;
    bool enableBilinearTextureFiltering = false;
    bool enableMipmapTextureFiltering   = false;
    bool enableMultipleRaysPerPixel     = false;
    bool enableGlossyReflection         = false;
    bool enableTransparency             = false;
    bool enableDepthOfField             = false;
};

struct Features {
    // Feature flags
    bool enableShading          = true;
    bool enableRecursive        = false;
    bool enableHardShadow       = true;
    bool enableSoftShadow       = true;
    bool enableNormalInterp     = true;
    bool enableTextureMapping   = true;
    bool enableAccelStructure   = true;

    // Base render parameters
    uint32_t maxReflectionRecursion = 5U;

    // ReSTIR feature flags
    bool initialSamplesVisibilityCheck  = false;
    bool unbiasedCombination            = false;
    bool spatialReuse                   = true;
    bool spatialReuseVisibilityCheck    = false;
    bool temporalReuse                  = true;

    // ReSTIR parameters
    uint32_t numSamplesInReservoir      = 1U;
    uint32_t initialLightSamples        = 32U;
    uint32_t numNeighboursToSample      = 5U;
    uint32_t spatialResamplingPasses    = 2U;
    uint32_t spatialResampleRadius      = 10U;
    uint32_t temporalClampM             = 20U;

    // Misc parameters
    bool enableToneMapping  = true;
    float gamma             = 1.0f;
    float exposure          = 1.5f;

    ExtraFeatures extra = {};
};