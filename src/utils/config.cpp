#include "config.h"

DISABLE_WARNINGS_PUSH()
#define TOML_EXCEPTIONS 0
#include <toml/toml.hpp>
DISABLE_WARNINGS_POP()

#include <scene/scene.h>

#include <algorithm>
#include <cctype>
#include <iostream>

//! ********** NO NEED TO USE THESE FILES (FOR GRADING PURPOSES)! *********************** //

// Helper function to print SceneType
static std::ostream& operator<<(std::ostream& os, const SceneType& sceneType)
{
    switch (sceneType) {
    case SceneType::SingleTriangle: {
        os << "SceneType::SingleTriangle";
        break;
    }
    case SceneType::Cube: {
        os << "SceneType::Cube";
        break;
    }
    case SceneType::CubeTextured: {
        os << "SceneType::CubeTextured";
        break;
    }
    case SceneType::CornellBox: {
        os << "SceneType::CornellBox";
        break;
    }
    case SceneType::CornellBoxParallelogramLight: {
        os << "SceneType::CornellBoxParallelogramLight";
        break;
    }
    case SceneType::Monkey: {
        os << "SceneType::Monkey";
        break;
    }
    case SceneType::CornellNightClub: {
        os << "SceneType::CornellNightClub";
        break;
    }
    case SceneType::ModernHall: {
        os << "SceneType::TheWoodenStaircase";
        break;
    }
    case SceneType::TheBreakfastRoom: {
        os << "SceneType::TheBreakfastRoom";
        break;
    }
    case SceneType::TheModernLivingRoom: {
        os << "SceneType::TheModernLivingRoom";
        break;
    }
    }
    return os;
}

// Helper function to print a glm::vec3.
static std::ostream& operator<<(std::ostream& os, const glm::vec3& vec3)
{
    os << "[" << vec3.x << ", " << vec3.y << ", " << vec3.z << "]";
    return os;
}

// Helper function to print configuration.
std::ostream& operator<<(std::ostream& os, const Config& config)
{
    os << "Final Project Configurations: " << std::endl
       << std::boolalpha
       << "  + command_line_rendering: " << config.cliRenderingEnabled << std::endl
       << "  + window_size: " << config.windowSize.x << ", " << config.windowSize.y << std::endl
       << "  + data_path: " << config.dataPath << std::endl
       << "  + scene: ";

    if (std::holds_alternative<SceneType>(config.scene)) {
        os << std::get<SceneType>(config.scene) << std::endl;
    } else {
        os << std::get<std::filesystem::path>(config.scene) << std::endl;
    }

    os << "  + output_filepath: " << config.outputDir << std::endl
       << "  + features: " << std::endl
       << "    - enable_shading: " << config.features.enableShading << std::endl
       << "    - enable_recursive: " << config.features.enableRecursive << std::endl
       << "    - enable_hard_shadow: " << config.features.enableHardShadow << std::endl
       << "    - enable_soft_shadow: " << config.features.enableSoftShadow << std::endl
       << "    - enable_normal_interp: " << config.features.enableNormalInterp << std::endl
       << "    - enable_texture_mapping: " << config.features.enableTextureMapping << std::endl
       << "    - enable_accel_structure: " << config.features.enableAccelStructure << std::endl;

    os << "  + cameras: " << std::endl;
    for (const auto& camera: config.cameras) {
        os << "    - field_of_view: " << camera.fieldOfView << std::endl
           << "      distance_from_look_at: " << camera.distanceFromLookAt << std::endl
           << "      look_at: " << camera.lookAt << std::endl
           << "      rotation: " << camera.rotation << std::endl;
    }

    os << "  + lights: " << std::endl;

    for (const auto& elem : config.lights) {
        std::visit([&os](auto&& light) {
            if constexpr (std::is_same_v<std::decay_t<decltype(light)>, PointLight>) {
                os << "    - type: point" << std::endl
                   << "      position: " << light.position << ", color: " << light.color << std::endl;
            } else if constexpr (std::is_same_v<std::decay_t<decltype(light)>, SegmentLight>) {
                os << "    - type: segment" << std::endl
                   << "      endpoint0: " << light.endpoint0 << ", endpoint1: " << light.endpoint1 << std::endl
                   << "      color0: " << light.color0 << ", color1: " << light.color1 << std::endl;
            } else if constexpr (std::is_same_v<std::decay_t<decltype(light)>, ParallelogramLight>) {
                os << "    - type: parallelogram" << std::endl
                   << "      v0: " << light.v0 << std::endl
                   << "      edge01: " << light.edge01 << ", edge02: " << light.edge02 << std::endl
                   << "      color0: " << light.color0 << ", color1: " << light.color1 << std::endl
                   << "      color2: " << light.color2 << ", color3: " << light.color3 << std::endl;
            }
        },
            elem);
    }
    return os;
}

// Helper function to parse a glm::vec3 from a toml::array
std::optional<glm::vec3> tomlArrayToVec3(const toml::array* array)
{
    glm::vec3 output {};

    if (array) {
        int i = 0;
        array->for_each([&](auto&& elem) {
            if (elem.is_number()) {
                if (i > 2)
                    return;
                output[i] = static_cast<float>(elem.as_floating_point()->get());
                i += 1;
            } else {
                std::cerr << "Error: Expected a number in array, got " << elem.type() << std::endl;
                return;
            }
        });
    }

    return output;
}

// Helper function to parse a glm::ivec2 from a toml::array
std::optional<glm::ivec2> tomlArrayToIVec2(const toml::array* array)
{
    glm::ivec2 output {};

    if (array) {
        int i = 0;
        array->for_each([&](auto&& elem) {
            if (elem.is_integer()) {
                if (i > 1)
                    return;
                output[i] = static_cast<int>(elem.as_integer()->get());
                i += 1;
            } else {
                std::cerr << "Error: Expected an integer in array, got " << elem.type() << std::endl;
                return;
            }
        });
    }

    return output;
}

Config readConfigFile(const std::filesystem::path& config_path)
{
    Config config = {};

    toml::parse_result result = toml::parse_file(config_path.string());
    if (!result) {
        std::cerr << "Failed parsing " << config_path << ":\n"
                  << result.error() << "\n";
    }

    const auto& table = result.table();

    config.cliRenderingEnabled = table["command_line_rendering"].as_boolean()->value_or(true);

    config.windowSize = tomlArrayToIVec2(table["window_size"].as_array())
                            .value_or(glm::ivec2(800, 800));

    std::string data_path = table["data_path"].value<std::string>().value_or(DATA_DIR);
    if (std::strcmp(data_path.c_str(), "default") == 0) {
        data_path = DATA_DIR;
    }
    config.dataPath = data_path;

    auto scene = table["scene"];
    if (scene.is_number()) {
        auto scene_type = static_cast<SceneType>(scene.as_integer()->get());
        config.scene = scene_type;
    } else {
        auto scene_path = scene.value<std::string>().value_or("none");
        auto deserialized_scene_type = deserialize(scene_path);
        if (deserialized_scene_type.has_value()) {
            config.scene = deserialized_scene_type.value();
        } else {
            // check if the scene file exists
            auto path = config.dataPath / scene_path;
            if (std::filesystem::exists(path)) {
                config.scene = path;
            } else {
                std::cerr << "Error: Scene file " << path << " does not exist." << std::endl;
                exit(1);
            }
        }
    }

    std::string output_dir = table["output_dir"].value<std::string>().value_or("");
    if (output_dir.empty()) {
        std::cout << "Warning: No output directory specified, using current directory." << std::endl;
        config.outputDir = std::filesystem::current_path();
    } else {

#ifdef __linux__
        // check if tilde/$HOME is used in path; if so, replace it with the home directory
        if (output_dir[0] == '~') {
            output_dir.replace(0, 1, std::getenv("HOME"));
        }

        if (output_dir.starts_with("$HOME")) {
            output_dir.replace(0, 5, std::getenv("HOME"));
        }
#endif

        config.outputDir = std::filesystem::absolute(std::filesystem::path(output_dir));
    }

    config.features.enableShading = table["features"]["enable_shading"]
                                .as_boolean()
                                ->value_or(false);
    config.features.enableRecursive = table["features"]["enable_recursive"]
                                  .as_boolean()
                                  ->value_or(false);
    config.features.enableHardShadow = table["features"]["enable_hard_shadow"]
                                    .as_boolean()
                                    ->value_or(false);
    config.features.enableNormalInterp = table["features"]["enable_normal_interp"]
                                     .as_boolean()
                                     ->value_or(false);
    config.features.enableTextureMapping = table["features"]["enable_texture_mapping"]
                                       .as_boolean()
                                       ->value_or(false);
    config.features.enableAccelStructure = table["features"]["enable_accel_structure"]
                                       .as_boolean()
                                       ->value_or(false);

    const toml::array* cameras = table["cameras"].as_array();
    if (cameras) {
        cameras->for_each([&](auto&& camera) {
            float fieldOfView = camera.at_path("field_of_view").as_floating_point()->value_or(50.0f);
            float distanceFromLookAt = camera.at_path("distance_from_look_at").as_floating_point()->value_or(3.0f);
            glm::vec3 look_at = tomlArrayToVec3(camera.at_path("look_at").as_array()).value_or(glm::vec3(0.0f));
            glm::vec3 rotation = tomlArrayToVec3(camera.at_path("rotation").as_array()).value_or(glm::vec3(20.0f, 20.0f, 0.0f));
            config.cameras.emplace_back(CameraConfig { fieldOfView, distanceFromLookAt, look_at, rotation });
        });
    }

    const toml::array* lights = table["lights"].as_array();
    if (lights) {
        lights->for_each([&](auto&& light) {
            std::string type = light.at_path("type").as_string()->value_or("none");
            if (type == "point") {
                glm::vec3 position = tomlArrayToVec3(light.at_path("position").as_array())
                                         .value_or(glm::vec3(0.0f));
                glm::vec3 color = tomlArrayToVec3(light.at_path("color").as_array())
                                      .value_or(glm::vec3(0.0f));
                config.lights.emplace_back(PointLight { position, color });
            } else if (type == "segment") {
                glm::vec3 endpoint0 = tomlArrayToVec3(light.at_path("endpoints").as_array()->at(0).as_array())
                                          .value_or(glm::vec3(0.0f));
                glm::vec3 endpoint1 = tomlArrayToVec3(light.at_path("endpoints").as_array()->at(1).as_array())
                                          .value_or(glm::vec3(0.0f));
                glm::vec3 color0 = tomlArrayToVec3(light.at_path("colors").as_array()->at(0).as_array())
                                       .value_or(glm::vec3(0.0f));
                glm::vec3 color1 = tomlArrayToVec3(light.at_path("colors").as_array()->at(1).as_array())
                                       .value_or(glm::vec3(0.0f));
                config.lights.emplace_back(SegmentLight { endpoint0, endpoint1, color0, color1 });
            } else if (type == "parallelogram") {
                glm::vec3 corner = tomlArrayToVec3(light.at_path("corner").as_array())
                                       .value_or(glm::vec3(0.0f));
                glm::vec3 edge0 = tomlArrayToVec3(light.at_path("edges").as_array()->at(0).as_array())
                                      .value_or(glm::vec3(0.0f));
                glm::vec3 edge1 = tomlArrayToVec3(light.at_path("edges").as_array()->at(1).as_array())
                                      .value_or(glm::vec3(0.0f));
                glm::vec3 color0 = tomlArrayToVec3(light.at_path("colors").as_array()->at(0).as_array())
                                       .value_or(glm::vec3(0.0f));
                glm::vec3 color1 = tomlArrayToVec3(light.at_path("colors").as_array()->at(1).as_array())
                                       .value_or(glm::vec3(0.0f));
                glm::vec3 color2 = tomlArrayToVec3(light.at_path("colors").as_array()->at(2).as_array())
                                       .value_or(glm::vec3(0.0f));
                glm::vec3 color3 = tomlArrayToVec3(light.at_path("colors").as_array()->at(3).as_array())
                                       .value_or(glm::vec3(0.0f));
                config.lights.emplace_back(ParallelogramLight { corner, edge0, edge1, color0, color1, color2, color3 });
            } else {
                std::cerr << "Unknown light type: " << type << " -- Skip" << std::endl;
            }
        });
    } else {
        std::cerr << "WARN: No lights found in config file.\n";
        config.lights = {};
    }

    return std::move(config);
}

std::string serialize(const SceneType& sceneType)
{
    switch (sceneType) {
    case SceneType::SingleTriangle:
        return "single_triangle";
    case SceneType::Cube:
        return "cube";
    case SceneType::CubeTextured:
        return "cube_textured";
    case SceneType::CornellBox:
        return "cornell_box";
    case SceneType::CornellBoxParallelogramLight:
        return "cornell_box_parallelogram_light";
    case SceneType::Monkey:
        return "monkey";
    default:
        return "unknown";
    }
}

std::optional<SceneType> deserialize(const std::string& sceneTypeStr)
{
    std::string lowered;
    std::transform(sceneTypeStr.begin(), sceneTypeStr.end(), std::back_inserter(lowered), ::tolower);
    if (lowered == "single_triangle" || lowered == "singletriangle" || lowered == "single-triangle") {
        return SceneType::SingleTriangle;
    } else if (lowered == "cube") {
        return SceneType::Cube;
    } else if (lowered == "cube-textured" || lowered == "cube_textured" || lowered == "cubetextured") {
        return SceneType::CubeTextured;
    } else if (lowered == "cornell_box" || lowered == "cornellbox" || lowered == "cornell-box") {
        return SceneType::CornellBox;
    } else if (lowered == "cornell_box_parallelogram_light" || lowered == "cornellboxparallelogramlight" || lowered == "cornell-box-parallelogram-light") {
        return SceneType::CornellBoxParallelogramLight;
    } else if (lowered == "monkey") {
        return SceneType::Monkey;
    } else {
        return std::nullopt;
    }
}
