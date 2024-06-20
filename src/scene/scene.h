#pragma once

#include <framework/mesh.h>
#include <framework/ray.h>
#include <framework/trackball.h>

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <utils/common.h>

#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

enum SceneType {
    SingleTriangle,
    Cube,
    CubeTextured,
    CornellBox,
    CornellBoxParallelogramLight,
    CornellNightClub,
    Monkey,
    ModernHall,
    TheBreakfastRoom,
    TheModernLivingRoom,
};

struct Scene {
    SceneType type;
    std::vector<Mesh> meshes;
    std::vector<Sphere> spheres;
    std::vector<std::variant<PointLight, SegmentLight, ParallelogramLight, DiskLight>> lights;
};

std::vector<ParallelogramLight> regularLightGrid(glm::vec3 startPos, glm::ivec2 counts, glm::vec3 edge01, glm::vec3 edge02,
                                                 glm::vec3 color,
                                                 float emptySpacePercentage = 0.1f);

void constructNightClubLights(Scene& scene);

// Load a prebuilt scene.
Scene loadScenePrebuilt(SceneType type, const std::filesystem::path& dataDir, Trackball& camera, Features& features);

// Load a scene from a file.
Scene loadSceneFromFile(const std::filesystem::path& path, const std::vector<std::variant<PointLight, SegmentLight, ParallelogramLight, DiskLight>>& lights);
