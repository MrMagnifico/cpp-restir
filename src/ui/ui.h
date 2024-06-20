#pragma once
#ifndef _UI_H_
#define _UI_H_

#include <ray_tracing/embree_interface.h>
#include <rendering/reservoir.h>
#include <rendering/screen.h>
#include <scene/scene.h>
#include <utils/common.h>
#include <utils/config.h>

#include <framework/trackball.h>


class UiManager {
public:
    UiManager(EmbreeInterface& embreeInterface, Trackball& camera, Config& config, std::optional<Ray>& optDebugRay,
              std::shared_ptr<ReservoirGrid>& previousFrameGrid, Scene& scene, SceneType& sceneType,
              Screen& screen, ViewMode& viewMode, Window& window,
              int& selectedLightIdx);

    void draw();

private:
    // External object handles
    EmbreeInterface& embreeInterface;
    Trackball& camera;
    Config& config;
    std::optional<Ray>& optDebugRay;
    std::shared_ptr<ReservoirGrid>& previousFrameGrid;
    Scene& scene;
    SceneType& sceneType;
    Screen& screen;
    ViewMode& viewMode;
    Window& window;

    // External primitive handles
    int& selectedLightIdx;

    // Final project template UI bits
    void drawProjectTab();
    void drawSceneSelection();
    void drawViewModeSelection();
    void drawFeaturesToggles();
    void drawCameraStats();
    void drawRenderToFile();
    void drawLightControls();

    // ReSTIR UI bits
    void drawRestirTab();
    void drawRestirFeaturesToggles();
    void drawRestirParams();

    // Misc UI bits
    void drawMiscTab();
    void drawToneMappingControls();
};

#endif
