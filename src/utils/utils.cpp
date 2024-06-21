#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
DISABLE_WARNINGS_POP()

#include <embree4/rtcore.h>

#include <framework/opengl_includes.h>

#include <scene/texture.h>
#include <ui/draw.h>
#include <utils/magic_enum.hpp>
#include <utils/utils.h>

#include <cstdlib>
#include <iostream>

std::string currentTime() {
    auto t  = std::time(nullptr);
    auto tm = std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(tm, "%d-%m-%Y %H-%M-%S");
    return oss.str();
}

float zeroWithinEpsilon(float val) { return std::abs(val) < ZERO_EPSILON; }

float linearMap(float val, float domainMin, float domainMax, float rangeMin, float rangeMax) {
    float ratio         = (val - domainMin) / (domainMax - domainMin);
    float scaledValue   = ratio * (rangeMax - rangeMin);
    float mappedValue   = scaledValue + rangeMin;
    return mappedValue;
}

glm::vec3 diffuseAlbedo(const HitInfo& hitInfo, const Features& features) {
    return features.enableTextureMapping && hitInfo.material.kdTexture                   ?
           acquireTexel(*hitInfo.material.kdTexture.get(), hitInfo.texCoord, features)   :
           hitInfo.material.kd;
}

// test the visibility at a given light sample
// returns true if sample is visible, false otherwise
bool testVisibilityLightSample(const glm::vec3& samplePos, const EmbreeInterface& embreeInterface, const Features& features, Ray ray, HitInfo hitInfo) {
    // Construct shadow ray
    glm::vec3 shadingPoint  = ray.origin + (ray.t * ray.direction);
    glm::vec3 pointToSample = glm::normalize(samplePos - shadingPoint); 
    shadingPoint            += pointToSample * SHADOW_RAY_EPSILON; // Small epsilon in shadow ray direction to avoid self-shadowing
    Ray shadowRay { shadingPoint, pointToSample, glm::distance(shadingPoint, samplePos) };

    // Visibility test and debug rays
    bool visible = !embreeInterface.anyHit(shadowRay);
    if (visible) {
        drawRay(shadowRay, SHADOW_RAY_NO_HIT_COLOR);
        return true;
    }
    drawRay(shadowRay, SHADOW_RAY_INTERSECT_COLOR);
    return false;
}

void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    std::cout << std::format("[EMBREE] {}: {}", magic_enum::enum_name(error), str) << std::endl;
}

void setOpenGLMatrices(const Trackball& camera) {
    // Load view matrix.
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    const glm::mat4 viewMatrix = camera.viewMatrix();
    glMultMatrixf(glm::value_ptr(viewMatrix));

    // Load projection matrix.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const glm::mat4 projectionMatrix = camera.projectionMatrix();
    glMultMatrixf(glm::value_ptr(projectionMatrix));
}
