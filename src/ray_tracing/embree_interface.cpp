#include "embree_interface.h"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/gtc/type_ptr.hpp>
DISABLE_WARNINGS_POP()

#include <ui/draw.h>
#include <utils/magic_enum.hpp>
#include <utils/utils.h>

#include <format>
#include <stdexcept>

EmbreeInterface::EmbreeInterface(const Scene& scene) {
    initDevice();
    initScene(scene);
}

EmbreeInterface::~EmbreeInterface() {
    rtcReleaseScene(m_scene);
    rtcReleaseDevice(m_device);
}

void EmbreeInterface::initDevice() {
    m_device = rtcNewDevice(NULL);
    if (!m_device) { std::runtime_error(std::format("Error {}: Cannot create device ", magic_enum::enum_name(rtcGetDeviceError(NULL)))); }
    rtcSetDeviceErrorFunction(m_device, errorFunction, NULL);
}

void EmbreeInterface::initScene(const Scene& scene) {
    m_scene = rtcNewScene(m_device);
    rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);
    for (const Mesh& mesh : scene.meshes) {
        // Create and populate buffers to house vertex and triangle data
        RTCGeometry geom    = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetGeometryVertexAttributeCount(geom, 2);
        glm::vec3* positions    = (glm::vec3*)  rtcSetNewGeometryBuffer(geom,   RTC_BUFFER_TYPE_VERTEX,             0,  RTC_FORMAT_FLOAT3,  sizeof(glm::vec3),  mesh.vertices.size());
        glm::vec3* normals      = (glm::vec3*)  rtcSetNewGeometryBuffer(geom,   RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,   0,  RTC_FORMAT_FLOAT3,  sizeof(glm::vec3),  mesh.vertices.size());
        glm::vec2* texCoords    = (glm::vec2*)  rtcSetNewGeometryBuffer(geom,   RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,   1,  RTC_FORMAT_FLOAT2,  sizeof(glm::vec2),  mesh.vertices.size());
        glm::uvec3* indices     = (glm::uvec3*) rtcSetNewGeometryBuffer(geom,   RTC_BUFFER_TYPE_INDEX,              0,  RTC_FORMAT_UINT3,   sizeof(glm::uvec3), mesh.triangles.size());
        populateVertexDataBuffers(positions, normals, texCoords, mesh.vertices);
        populateIndexBuffer(indices, mesh.triangles);

        // Commit geometry and store its ID for material info lookup
        rtcCommitGeometry(geom);
        uint32_t geomId             = rtcAttachGeometry(m_scene, geom);
        m_meshToMaterial[geomId]    = mesh.material;
        rtcReleaseGeometry(geom);
    }
    rtcCommitScene(m_scene);
}

void EmbreeInterface::changeScene(const Scene& scene) {
    rtcReleaseScene(m_scene);
    initScene(scene);
}

bool EmbreeInterface::anyHit(Ray& ray) const {
    RTCRayHit rayhit = constructEmbreeRay(ray);
    rtcOccluded1(m_scene, &rayhit.ray);
    return rayhit.ray.tfar == -std::numeric_limits<float>::infinity(); // Tfar is set to -inf if any hit is detected
}

bool EmbreeInterface::closestHit(Ray& ray, HitInfo& hitInfo) const {
    // Construct Embree ray and carry out intersection test
    RTCRayHit rayhit = constructEmbreeRay(ray);
    rtcIntersect1(m_scene, &rayhit);

    // No intersection: draw no hit debug ray and return false
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        drawRay(ray, CAMERA_RAY_NO_HIT_COLOR);
        return false;
    };

    // Intersection: update hit info and user ray
    rtcInterpolate0(rtcGetGeometry(m_scene, rayhit.hit.geomID), rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v,
                    RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,   0, glm::value_ptr(hitInfo.normal),              3);
    rtcInterpolate0(rtcGetGeometry(m_scene, rayhit.hit.geomID), rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v,
                    RTC_BUFFER_TYPE_VERTEX,             0, glm::value_ptr(hitInfo.barycentricCoord),    3);
    rtcInterpolate0(rtcGetGeometry(m_scene, rayhit.hit.geomID), rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v,
                    RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,   1, glm::value_ptr(hitInfo.texCoord),            2);
    hitInfo.material    = m_meshToMaterial.at(rayhit.hit.geomID);
    hitInfo.geometryId  = rayhit.hit.geomID;
    ray.t               = rayhit.ray.tfar;

    // Debug draw ray and normal, then return true
    drawRay(ray, CAMERA_RAY_HIT_COLOR);
    drawRay({ray.origin + (ray.t * ray.direction), hitInfo.normal, 1.0f}, hitInfo.material.kd);
    return true;
}

void EmbreeInterface::populateVertexDataBuffers(glm::vec3* positionBuffer, glm::vec3* normalBuffer, glm::vec2* texCoordBuffer,
                                                const std::vector<Vertex>& vertices) {
    for (size_t vertexIdx = 0ULL; vertexIdx < vertices.size(); vertexIdx++) {
        positionBuffer[vertexIdx]   = vertices[vertexIdx].position;
        normalBuffer[vertexIdx]     = vertices[vertexIdx].normal;
        texCoordBuffer[vertexIdx]   = vertices[vertexIdx].texCoord;
    }
}

void EmbreeInterface::populateIndexBuffer(glm::uvec3* indexBuffer, const std::vector<glm::uvec3>& indices) {
    for (size_t triangleIdx = 0ULL; triangleIdx < indices.size(); triangleIdx++) {
        indexBuffer[triangleIdx] = indices[triangleIdx];
    }
}

RTCRayHit EmbreeInterface::constructEmbreeRay(const Ray& ray) const {
    RTCRayHit rayhit;
    rayhit.ray.org_x        = ray.origin.x;
    rayhit.ray.org_y        = ray.origin.y;
    rayhit.ray.org_z        = ray.origin.z;
    rayhit.ray.dir_x        = ray.direction.x;
    rayhit.ray.dir_y        = ray.direction.y;
    rayhit.ray.dir_z        = ray.direction.z;
    rayhit.ray.tnear        = 0.0f;
    rayhit.ray.tfar         = ray.t;
    rayhit.ray.mask         = -1;
    rayhit.ray.flags        = 0;
    rayhit.hit.geomID       = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0]    = RTC_INVALID_GEOMETRY_ID;
    return rayhit;
}
