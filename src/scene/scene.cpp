#include "scene.h"

#include <cmath>
#include <iostream>

std::vector<ParallelogramLight> regularLightGrid(glm::vec3 startPos, glm::ivec2 counts, glm::vec3 edge01, glm::vec3 edge02,
                                                 glm::vec3 color,
                                                 float emptySpacePercentage) {
    std::vector<ParallelogramLight> lights;
    glm::vec3 spaceEdge01   = edge01 / static_cast<float>(counts[0]);
    glm::vec3 spaceEdge02   = edge02 / static_cast<float>(counts[1]);
    glm::vec3 lightEdge01   = (edge01 * (1.0f - emptySpacePercentage)) / static_cast<float>(counts[0]);
    glm::vec3 lightEdge02   = (edge02 * (1.0f - emptySpacePercentage)) / static_cast<float>(counts[1]);
    for (int xLight = 0; xLight < counts.x; xLight++) {
        for (int yLight = 0; yLight < counts.y; yLight++) {
            glm::vec3 lightOrigin = startPos + (spaceEdge01 * static_cast<float>(xLight)) + (spaceEdge02 * static_cast<float>(yLight));
            lights.emplace_back(ParallelogramLight {
                .v0 = lightOrigin,
                .edge01 = lightEdge01,
                .edge02 = lightEdge02,
                .color0 = color,
                .color1 = color,
                .color2 = color,
                .color3 = color,
            });
        }
    }
    return lights;
}

void constructNightClubLights(Scene& scene) {
    glm::vec3 startingVertex, lightEdge01, lightEdge02, lightColor, interLightDist;
    glm::ivec2 counts           = glm::ivec2(16, 16);
    float freeSpacePercentage   = 0.30f;

    // Right wall
    startingVertex      = glm::vec3(-8.7f, 6.4f, -9.1f);
    lightEdge01         = glm::vec3(0.0f, 0.0f, 17.0f);
    lightEdge02         = glm::vec3(0.0f, -6.0f, 0.0f);
    lightColor          = glm::vec3(0.65f);
    std::vector<ParallelogramLight> rightWallLights = regularLightGrid(startingVertex, counts, lightEdge01, lightEdge02, lightColor, freeSpacePercentage);
    scene.lights.insert(scene.lights.end(), rightWallLights.begin(), rightWallLights.end());

    // Back wall
    startingVertex      = glm::vec3(9.2f, 6.4f, 8.6f);
    lightEdge01         = glm::vec3(-17.0f, 0.0f, 0.0f);
    lightEdge02         = glm::vec3(0.0f, -6.0f, 0.0f);
    lightColor          = glm::vec3(0.4f);
    std::vector<ParallelogramLight> backWallLights = regularLightGrid(startingVertex, counts, lightEdge01, lightEdge02, lightColor, freeSpacePercentage);
    scene.lights.insert(scene.lights.end(), backWallLights.begin(), backWallLights.end());
}

Scene loadScenePrebuilt(SceneType type, const std::filesystem::path& dataDir, Trackball& camera, Features& features) {
    Scene scene;
    scene.type = type;
    switch (type) {
    case SingleTriangle: {
        // Load a 3D model with a single triangle
        auto subMeshes = loadMesh(dataDir / "triangle.obj");
        subMeshes[0].material.kd = glm::vec3(1.0f);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        scene.lights.emplace_back(PointLight { glm::vec3(-1, 1, -1), glm::vec3(1) });
    } break;
    case Cube: {
        // Load a 3D model of a cube with 12 triangles
        auto subMeshes = loadMesh(dataDir / "cube.obj");
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        // scene.lights.push_back(PointLight { glm::vec3(-1, 1, -1), glm::vec3(1) });
        scene.lights.emplace_back(SegmentLight {
            .endpoint0 = glm::vec3(1.5f, 0.5f, -0.6f),
            .endpoint1 = glm::vec3(-1, 0.5f, -0.5f),
            .color0 = glm::vec3(0.9f, 0.2f, 0.1f), // Red-ish
            .color1 = glm::vec3(0.2f, 1, 0.3f) // Green-ish
        });
    } break;
    case CubeTextured: {
        auto subMeshes = loadMesh(dataDir / "cube-textured.obj");
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        scene.lights.emplace_back(PointLight { glm::vec3(-1.0, 1.5, -1.0), glm::vec3(1) });
    } break;
    case CornellBox: {
        // Load a 3D model of a Cornell Box
        auto subMeshes = loadMesh(dataDir / "CornellBox-Mirror-Rotated.obj", true);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        scene.lights.emplace_back(PointLight { glm::vec3(0, 0.58f, 0), glm::vec3(1) }); // Light at the top of the box
    } break;
    case CornellBoxParallelogramLight: {
        // Load a 3D model of a Cornell Box
        auto subMeshes = loadMesh(dataDir / "CornellBox-Mirror-Rotated.obj", true);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        // Light at the top of the box.
        scene.lights.emplace_back(ParallelogramLight {
            .v0 = glm::vec3(-0.2f, 0.5f, 0),
            .edge01 = glm::vec3(0.4f, 0, 0),
            .edge02 = glm::vec3(0.0f, 0.0f, 0.4f),
            .color0 = glm::vec3(1.0f, 1.0f, 1.0f), // White
            .color1 = glm::vec3(0.5f, 0.5f, 0.5f), // Grey
            .color2 = glm::vec3(0.5f, 0.5f, 0.5f), // Grey
            .color3 = glm::vec3(1.0f, 1.0f, 1.0f), // White
        });
    } break;
    case CornellNightClub: {
        auto subMeshes = loadMesh(dataDir / "cornell-nightclub.obj", false);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        
        // Light arrangement
        constructNightClubLights(scene);

        // Camera and tonemapping
        camera.setCamera(glm::vec3(2.57f, 1.23f, -1.35f), glm::radians(glm::vec3(10.3f, 30.0f, 0.0f)), 25.0f);
        features.exposure = 1.5f;
    } break;
    case Monkey: {
        // Load a 3D model of a Monkey
        auto subMeshes = loadMesh(dataDir / "monkey.obj", true);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        scene.lights.emplace_back(PointLight { glm::vec3(-1, 1, -1), glm::vec3(1) });
        scene.lights.emplace_back(DiskLight  { glm::vec3(1, -1, -1), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f), 1.0f });
    } break;
    case ModernHall: {
        auto subMeshes = loadMesh(dataDir / "modern-hall" / "modern-hall.obj", false);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
        
        // Lighting arrangement
        scene.lights.emplace_back(ParallelogramLight {
            .v0 = glm::vec3(-14.0f, 0.0f, 3.05f),
            .edge01 = glm::vec3(0.4f, 0, -2.955f),
            .edge02 = glm::vec3(0.0f, 5.0f, 0.0f),
            .color0 = glm::vec3(1.0f), .color1 = glm::vec3(1.0f), .color2 = glm::vec3(1.0f), .color3 = glm::vec3(1.0f),
        });
        std::vector<float> xPositionsRoofLights = { -11.4f, -8.0f, -5.0f, -1.38f, 1.77f };
        for (float xPosition : xPositionsRoofLights) { scene.lights.emplace_back(DiskLight { glm::vec3(xPosition, 5.6f, 1.5f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f), 1.0f }); }
        std::vector<glm::vec2> wallLightsPositionsXY = { {2.549f, 1.098}, {0.100f, 2.522f}, {-2.339f, 3.966f}, {-4.926f, 5.539f} };
        for (glm::vec2 positionXY: wallLightsPositionsXY) { scene.lights.emplace_back(DiskLight { glm::vec3(positionXY.x, positionXY.y, -2.992f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f), 0.05f }); }
    
        // Camera and tonemapping
        camera.setCamera(glm::vec3(-10.76f, 1.78f, -2.5f), glm::radians(glm::vec3(1.0f, 255.9f, 0.0f)), 18.0f);
        features.exposure = 5.0f;
    } break;
    case TheBreakfastRoom: {
        auto subMeshes = loadMesh(dataDir / "the-breakfast-room" / "the-breakfast-room.obj", false);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));

        // Lighting arrangement
        scene.lights.emplace_back(DiskLight { glm::vec3(-2.185f, 3.591f, -1.936f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f), 0.575f });
        scene.lights.emplace_back(DiskLight { glm::vec3(0.953f, 3.591f, -1.936f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f), 0.575f });

        // Camera and tonemapping
        camera.setCamera(glm::vec3(-0.46f, -1.22f, -11.99f), glm::radians(glm::vec3(14.4f, 180.0f, 0.0f)), 19.5f);
        features.exposure   = 15.0f;
    } break;
    case TheModernLivingRoom: {
        auto subMeshes = loadMesh(dataDir / "the-modern-living-room" / "the-modern-living-room.obj", false);
        std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));

        // Lighting arrangement
        scene.lights.emplace_back(DiskLight { glm::vec3(-3.193f, 2.676f, -1.803f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f), 1.0f });

        // Camera and tonemapping
        camera.setCamera(glm::vec3(-1.68f, 0.81f, -1.67f), glm::radians(glm::vec3(5.8f, -9.0f, 0.0f)), 4.0f);
        features.exposure   = 15.0f;
    } break;
    };
    return scene;
}

Scene loadSceneFromFile(const std::filesystem::path& path, const std::vector<std::variant<PointLight, SegmentLight, ParallelogramLight, DiskLight>>& lights) {
    Scene scene;
    scene.lights    = std::move(lights);
    auto subMeshes  = loadMesh(path);
    std::move(std::begin(subMeshes), std::end(subMeshes), std::back_inserter(scene.meshes));
    return scene;
}
