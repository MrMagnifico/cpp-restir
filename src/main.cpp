#include <ray_tracing/embree_interface.h>
#include <rendering/render.h>
#include <rendering/reservoir.h>
#include <rendering/screen.h>
#include <scene/light.h>
#include <ui/draw.h>
#include <ui/ui.h>
#include <utils/config.h>
#include <utils/utils.h>

// Suppress warnings in third-party code.
#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <imgui/imgui.h>
#include <nativefiledialog/nfd.h>
DISABLE_WARNINGS_POP()
#include <framework/imguizmo.h>
#include <framework/trackball.h>
#include <framework/variant_helper.h>
#include <framework/window.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <variant>

static void drawLightsOpenGL(const Scene& scene, const Trackball& camera, int selectedLight);
static void drawSceneOpenGL(const Scene& scene);


int main(int argc, char** argv) {
    // Read config file if provided
    Config config = {};
    if (argc > 1)   { config = readConfigFile(argv[1]); }
    else            { config.cameras.emplace_back(CameraConfig {}); } // Add a default camera if no config file is given.

    if (!config.cliRenderingEnabled) {
        Trackball::printHelp();
        std::cout << std::endl
                  << "Press the [R] key on your keyboard to create a ray towards the mouse cursor"       << std::endl
                  << "Press the [M] key on your keyboard to toggle between rasterized and ray traced modes" << std::endl
                  << std::endl;

        Window window       { "Seminar Implementation (ReSTIR)", config.windowSize, OpenGLVersion::GL2, true };
        Screen screen       { config.windowSize, true };
        Trackball camera    { &window, glm::radians(config.cameras[0].fieldOfView), config.cameras[0].distanceFromLookAt };
        camera.setCamera(config.cameras[0].lookAt, glm::radians(config.cameras[0].rotation), config.cameras[0].distanceFromLookAt);

        SceneType sceneType = SceneType::CornellNightClub;
        std::optional<Ray> optDebugRay;
        Scene scene         = loadScenePrebuilt(sceneType, config.dataPath);
        EmbreeInterface embreeInterface(scene);
        std::shared_ptr<ReservoirGrid> previousFrameGrid;

        ViewMode viewMode       = ViewMode::Rasterization;
        int selectedLightIdx    = scene.lights.empty() ? -1 : 0;

        UiManager uiManager(embreeInterface, camera, config, optDebugRay, previousFrameGrid, scene, sceneType, screen, viewMode, window,
                            selectedLightIdx);

        window.registerKeyCallback([&](int key, int /* scancode */, int action, int /* mods */) {
            if (action == GLFW_PRESS) {
                switch (key) {
                    // Shoot a ray. Produce a ray from camera to the far plane.
                    case GLFW_KEY_R: {
                        const auto tmp  = window.getNormalizedCursorPos();
                        optDebugRay     = camera.generateRay(tmp * 2.0f - 1.0f);
                        break;
                    }
                    case GLFW_KEY_ESCAPE: {
                        window.close();
                        break;
                    }
                    case GLFW_KEY_M: { // Change render mode and reset temporal predecessor
                        viewMode            = viewMode == ViewMode::Rasterization ? ViewMode::RayTracing : ViewMode::Rasterization;
                        previousFrameGrid   = nullptr;
                        break;
                    }
                };
            }
        });

        while (!window.shouldClose()) {
            camera.resetLastDelta();
            window.updateInput();

            // Set up the UI
            uiManager.draw();

            // Clear screen.
            glViewport(0, 0, window.getFrameBufferSize().x, window.getFrameBufferSize().y);
            glClearDepth(1.0);
            glClearColor(0.0, 0.0, 0.0, 0.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            setOpenGLMatrices(camera);

            // Draw either using OpenGL (rasterization) or the ray tracing function.
            switch (viewMode) {
                case ViewMode::Rasterization: {
                    glPushAttrib(GL_ALL_ATTRIB_BITS);
                    drawSceneOpenGL(scene);
                    if (optDebugRay) {
                        // Call getFinalColor for the debug ray. Ignore the result but tell the function that it should
                        // draw the rays instead.
                        enableDebugDraw = true;
                        glDisable(GL_LIGHTING);
                        glDepthFunc(GL_LEQUAL);
                        (void) genCanonicalSamples(scene, embreeInterface, config.features, *optDebugRay);
                        enableDebugDraw = false;
                    }
                    glPopAttrib();

                    drawLightsOpenGL(scene, camera, selectedLightIdx);
                } break;
                case ViewMode::RayTracing: {
                    const auto start    = std::chrono::high_resolution_clock::now();
                    screen.clear(glm::vec3(0.0f));
                    previousFrameGrid   = make_shared<ReservoirGrid>(renderRayTracing(previousFrameGrid, scene, camera, embreeInterface, screen, camera.getLastDelta(), config.features));
                    screen.setPixel(0, 0, glm::vec3(1.0f));
                    screen.draw(); // Takes the image generated using ray tracing and outputs it to the screen using OpenGL.
                    const auto end      = std::chrono::high_resolution_clock::now();
                    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    fmt::print("Render time: {}ms\n", duration);
                } break;
                default:
                    break;
                }

            window.swapBuffers();
        }
    } else {
        // Command-line rendering.
        std::cout << config;

        // NOTE(Yang): Trackball is highly coupled with the window,
        // so we need to create a dummy window here but not show it.
        // In this case, GLEW will not be initialized, OpenGL functions
        // will not be loaded, and the window will not be shown.
        // All debug draw calls will be disabled.
        enableDebugDraw = false;
        Window window { "Seminar Implementation (ReSTIR)", config.windowSize, OpenGLVersion::GL2, false };

        // Load scene.
        Scene scene;
        std::string sceneName;
        std::visit(make_visitor(
                       [&](const std::filesystem::path& path) {
                           scene = loadSceneFromFile(path, config.lights);
                           sceneName = path.stem().string();
                       },
                       [&](const SceneType& type) {
                           scene = loadScenePrebuilt(type, config.dataPath);
                           sceneName = serialize(type);
                       }),
            config.scene);

        EmbreeInterface embreeInterface(scene);
        std::shared_ptr<ReservoirGrid> previousFrameGrid;

        // Create output directory if it does not exist.
        if (!std::filesystem::exists(config.outputDir)) { std::filesystem::create_directories(config.outputDir); }

        const auto start                = std::chrono::high_resolution_clock::now();
        std::string start_time_string   = fmt::format("{:%Y-%m-%d-%H:%M:%S}", fmt::localtime(std::time(nullptr)));

        std::vector<std::thread> workers;
        for (int i = 0; auto const& cameraConfig : config.cameras) {
            workers.emplace_back(std::thread([&](int index) {
                Screen screen { config.windowSize, false };
                screen.clear(glm::vec3(0.0f));
                Trackball camera { &window, glm::radians(cameraConfig.fieldOfView), cameraConfig.distanceFromLookAt };
                camera.setCamera(cameraConfig.lookAt, glm::radians(cameraConfig.rotation), cameraConfig.distanceFromLookAt);
                previousFrameGrid           = make_shared<ReservoirGrid>(renderRayTracing(previousFrameGrid, scene, camera, embreeInterface, screen, camera.getLastDelta(), config.features));
                const auto filename_base    = fmt::format("{}_{}_cam_{}", sceneName, start_time_string, index);
                const auto filepath         = config.outputDir / (filename_base + ".bmp");
                fmt::print("Image {} saved to {}\n", index, filepath.string());
                screen.writeBitmapToFile(filepath);
            },
                i));
            ++i;
        }
        for (auto& worker : workers) { worker.join(); }
        const auto end      = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        fmt::print("Rendering took {} ms, {} images rendered.\n", duration, config.cameras.size());
    }
    return 0;
}

static void drawLightsOpenGL(const Scene& scene, const Trackball& camera, int /* selectedLight */) {
    glEnable(GL_NORMALIZE);             // Normals will be normalized in the graphics pipeline
    glEnable(GL_DEPTH_TEST);            // Activate rendering modes.
    glPolygonMode(GL_FRONT, GL_FILL);   // Draw front and back facing triangles filled
    glPolygonMode(GL_BACK, GL_FILL);    
    glShadeModel(GL_SMOOTH);            // Interpolate vertex colors over the triangles
    glDisable(GL_LIGHTING);

    // Draw all non-selected lights.
    for (size_t i = 0; i < scene.lights.size(); i++) {
        std::visit(
            make_visitor(
                [](const PointLight& light) { drawSphere(light.position, 0.01f, light.color); },
                [](const SegmentLight& light) {
                    glPushAttrib(GL_ALL_ATTRIB_BITS);
                    glBegin(GL_LINES);
                    glColor3fv(glm::value_ptr(light.color0));
                    glVertex3fv(glm::value_ptr(light.endpoint0));
                    glColor3fv(glm::value_ptr(light.color1));
                    glVertex3fv(glm::value_ptr(light.endpoint1));
                    glEnd();
                    glPopAttrib();
                    drawSphere(light.endpoint0, 0.01f, light.color0);
                    drawSphere(light.endpoint1, 0.01f, light.color1);
                },
                [](const ParallelogramLight& light) {
                    glPushAttrib(GL_ALL_ATTRIB_BITS);
                    glBegin(GL_QUADS);
                    glColor3fv(glm::value_ptr(light.color0));
                    glVertex3fv(glm::value_ptr(light.v0));
                    glColor3fv(glm::value_ptr(light.color1));
                    glVertex3fv(glm::value_ptr(light.v0 + light.edge01));
                    glColor3fv(glm::value_ptr(light.color3));
                    glVertex3fv(glm::value_ptr(light.v0 + light.edge01 + light.edge02));
                    glColor3fv(glm::value_ptr(light.color2));
                    glVertex3fv(glm::value_ptr(light.v0 + light.edge02));
                    glEnd();
                    glPopAttrib();
                },
                [](auto) { /* any other type of light */ }),
            scene.lights[i]);
    }

    // Draw a colored sphere at the location at which the trackball is looking/rotating around.
    glDisable(GL_LIGHTING);
    drawSphere(camera.lookAt(), 0.01f, glm::vec3(0.2f, 0.2f, 1.0f));
}

void drawSceneOpenGL(const Scene& scene) {
    // Activate the light in the legacy OpenGL mode.
    glEnable(GL_LIGHTING);

    // Tell OpenGL where the lights are (so it nows how to shade surfaces in the scene).
    // This is only used in the rasterization view. OpenGL only supports point lights so
    // we replace segment/parallelogram lights by point lights.
    int i = 0;
    const auto enableLight = [&](const glm::vec3& position, const glm::vec3 color) {
        const GLenum glLight = static_cast<GLenum>(GL_LIGHT0 + i);
        glEnable(glLight);
        const glm::vec4 position4 { position, 1 };
        glLightfv(glLight, GL_POSITION, glm::value_ptr(position4));
        const glm::vec4 color4 { glm::clamp(color, 0.0f, 1.0f), 1.0f };
        const glm::vec4 zero4 { 0.0f, 0.0f, 0.0f, 1.0f };
        glLightfv(glLight, GL_AMBIENT, glm::value_ptr(zero4));
        glLightfv(glLight, GL_DIFFUSE, glm::value_ptr(color4));
        glLightfv(glLight, GL_SPECULAR, glm::value_ptr(zero4));
        // NOTE: quadratic attenuation doesn't work like you think it would in legacy OpenGL.
        // The distance is not in world space but in NDC space!
        glLightf(glLight, GL_CONSTANT_ATTENUATION, 1.0f);
        glLightf(glLight, GL_LINEAR_ATTENUATION, 0.0f);
        glLightf(glLight, GL_QUADRATIC_ATTENUATION, 0.0f);
        i++;
    };
    for (const auto& light : scene.lights) {
        std::visit(
            make_visitor(
                [&](const PointLight& pointLight) {
                    enableLight(pointLight.position, pointLight.color);
                },
                [&](const SegmentLight& segmentLight) {
                    // Approximate with two point lights: one at each endpoint.
                    enableLight(segmentLight.endpoint0, 0.5f * segmentLight.color0);
                    enableLight(segmentLight.endpoint1, 0.5f * segmentLight.color1);
                },
                [&](const ParallelogramLight& parallelogramLight) {
                    enableLight(parallelogramLight.v0, 0.25f * parallelogramLight.color0);
                    enableLight(parallelogramLight.v0 + parallelogramLight.edge01, 0.25f * parallelogramLight.color1);
                    enableLight(parallelogramLight.v0 + parallelogramLight.edge02, 0.25f * parallelogramLight.color2);
                    enableLight(parallelogramLight.v0 + parallelogramLight.edge01 + parallelogramLight.edge02, 0.25f * parallelogramLight.color3);
                },
                [](auto) { /* any other type of light */ }),
            light);
    }

    // Draw the scene and the ray (if any).
    drawScene(scene);
}
