#pragma once

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <scene/scene.h>
#include <utils/common.h>

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

//! ********** NO NEED TO USE THESE FILES (FOR GRADING PURPOSES)! *********************** //

struct CameraConfig {
    float fieldOfView           = 30.0f;                    // Degrees
    float distanceFromLookAt    = 25.0f;
    glm::vec3 lookAt            = { 2.57f, 1.23f, -1.35f };
    glm::vec3 rotation          = { 10.3f, 30.0f, 0.0f };    // Degrees
};

struct Config {
    Features features = {};

    bool cliRenderingEnabled                                = false;
    glm::ivec2 windowSize                                   = { 1280, 720 };
    std::filesystem::path dataPath                          = DATA_DIR;
    std::variant<SceneType, std::filesystem::path> scene    = SceneType::CornellBoxParallelogramLight;
    std::filesystem::path outputDir                         = "";

    std::vector<CameraConfig> cameras;
    std::vector<std::variant<PointLight, SegmentLight, ParallelogramLight>> lights;
};

std::ostream& operator<<(std::ostream& arg, const Config& config);

Config readConfigFile(const std::filesystem::path& config_path);

std::string serialize(const SceneType& sceneType);
std::optional<SceneType> deserialize(const std::string& lowered);