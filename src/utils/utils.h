#pragma once
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <ray_tracing/embree_interface.h>
#include <utils/common.h>

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <framework/trackball.h>

#include <limits>

constexpr float SHADOW_RAY_EPSILON              = 1E-3F;
constexpr glm::vec3 SHADOW_RAY_INTERSECT_COLOR  = {1.0f, 0.0f, 0.0f};
constexpr glm::vec3 SHADOW_RAY_NO_HIT_COLOR     = {0.0f, 1.0f, 1.0f};
float constexpr ZERO_EPSILON                    = 1E-5F;

// General purpose
std::string currentTime();
float zeroWithinEpsilon(float val);
float linearMap(float val, float domainMin, float domainMax, float rangeMin, float rangeMax);
template<typename T>
inline bool inRangeInclusive(T val, T low, T high) { return low <= val && val <= high; }

// Convenience or base project
glm::vec3 diffuseAlbedo(const HitInfo& hitInfo, const Features& features);
bool testVisibilityLightSample(const glm::vec3& samplePos, const EmbreeInterface& embreeInterface, const Features& features, Ray ray, HitInfo hitInfo);

// Embree
void errorFunction(void* userPtr, enum RTCError error, const char* str);

// Framework
void setOpenGLMatrices(const Trackball& camera);

#endif