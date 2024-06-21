
#include <framework/trackball.h>

#ifdef NDEBUG
#include <omp.h>
#endif

#include <post_processing/tone_mapping.h>
#include <rendering/render.h>
#include <rendering/screen.h>
#include <scene/light.h>
#include <utils/utils.h>

#include <array>
#include <iostream>
#include <random>
#include <vector>

PrimaryHitGrid genPrimaryRayHits(const Scene& scene, const Trackball& camera, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();
    PrimaryHitGrid primaryHits(windowResolution.y, std::vector<RayHit>(windowResolution.x));

    std::cout << "Primary rays computation..." << std::endl;
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            const glm::vec2 normalizedPixelPos { float(x) / float(windowResolution.x) * 2.0f - 1.0f,
                                                 float(y) / float(windowResolution.y) * 2.0f - 1.0f };
            primaryHits[y][x].ray = camera.generateRay(normalizedPixelPos); 
            embreeInterface.closestHit(primaryHits[y][x].ray, primaryHits[y][x].hit);
        }
    }
    return primaryHits;
}

glm::vec3 finalShading(const Reservoir& reservoir, const Ray& primaryRay, const EmbreeInterface& embreeInterface, const Features& features) {
    glm::vec3 finalColor(0.0f);
    for (const SampleData& sample : reservoir.outputSamples) {
        glm::vec3 sampleColor   = testVisibilityLightSample(sample.lightSample.position, embreeInterface, features, primaryRay, reservoir.hitInfo)  ?
                                  computeShading(sample.lightSample.position, sample.lightSample.color, features, primaryRay, reservoir.hitInfo)    :
                                  glm::vec3(0.0f);
        sampleColor             *= sample.outputWeight;
        finalColor              += sampleColor;
    }
    finalColor /= reservoir.outputSamples.size(); // Divide final shading value by number of samples
    return finalColor;
}

void combineToScreen(Screen& screen, const PixelGrid& finalPixelColors, const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();
    std::cout << "Iteration combination..." << std::endl;
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            glm::vec3 finalColor = finalPixelColors[y][x] / static_cast<float>(features.maxIterations);
            if (features.enableToneMapping) { finalColor = exposureToneMapping(finalColor, features); }
            screen.setPixel(x, y, finalColor);
        }
    }
}

ReservoirGrid genInitialSamples(const PrimaryHitGrid& primaryHits, const Scene& scene, const EmbreeInterface& embreeInterface, const Features& features, const glm::ivec2& windowResolution) {
    ReservoirGrid initialSamples(windowResolution.y, std::vector<Reservoir>(windowResolution.x, Reservoir(features.numSamplesInReservoir)));
    std::cout << "Initial light samples generation..." << std::endl;
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            initialSamples[y][x] = genCanonicalSamples(scene, embreeInterface, features, primaryHits[y][x]);
        }
    }
    return initialSamples;
}

void spatialReuse(ReservoirGrid& reservoirGrid, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features) {
    // Uniform selection of neighbours in N pixel Manhattan distance radius
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(-static_cast<int32_t>(features.spatialResampleRadius), features.spatialResampleRadius);

    std::cout << "Spatial reuse..." << std::endl;
    glm::ivec2 windowResolution = screen.resolution();
    ReservoirGrid prevIteration = reservoirGrid;
    for (uint32_t pass = 0U; pass < features.spatialResamplingPasses; pass++) {
        std::cout << "Pass " << pass + 1 << std::endl;
        #ifdef NDEBUG
        #pragma omp parallel for schedule(guided)
        #endif
        for (int y = 0; y < windowResolution.y; y++) {
            for (int x = 0; x != windowResolution.x; x++) {
                // Select candidates
                std::vector<Reservoir> selected;
                selected.reserve(features.numNeighboursToSample + 1U); // Reserve memory needed for maximum possible number of samples (neighbours + current)
                Reservoir& current = reservoirGrid[y][x];
                for (uint32_t neighbourCount = 0U; neighbourCount < features.numNeighboursToSample; neighbourCount++) {
                    int neighbourX              = std::clamp(x + distr(gen), 0, windowResolution.x - 1);
                    int neighbourY              = std::clamp(y + distr(gen), 0, windowResolution.y - 1);
                    Reservoir neighbour         = prevIteration[neighbourY][neighbourX]; // Create copy for local modification
                    
                    // Conduct heuristic check if biased combination is used
                    if (!features.unbiasedCombination) { 
                        float depthFracDiff     = std::abs(1.0f - (neighbour.cameraRay.t / current.cameraRay.t));   // Check depth difference (greater than 10% leads to rejection) 
                        float normalsDotProd    = glm::dot(neighbour.hitInfo.normal, current.hitInfo.normal);       // Check normal difference (greater than 25 degrees leads to rejection)
                        if (depthFracDiff > 0.1f || normalsDotProd < 0.90630778703f) { continue; } 
                    }
                    
                    selected.push_back(neighbour);
                }

                // Ensure pixel's own reservoir is also considered
                selected.push_back(current);

                // Combine to single reservoir (biased or unbiased depending on user selection)
                Reservoir combined(current.outputSamples.size());
                combined.cameraRay  = current.cameraRay;
                combined.hitInfo    = current.hitInfo;
                if (features.unbiasedCombination)   { Reservoir::combineUnbiased(selected, combined, embreeInterface, features); }
                else                                { Reservoir::combineBiased(selected, combined, features); }
                reservoirGrid[y][x] = combined;
            }
        }
        prevIteration = reservoirGrid;
    }
}

void temporalReuse(ReservoirGrid& reservoirGrid, ReservoirGrid& previousFrameGrid, const EmbreeInterface& embreeInterface,
                   Screen& screen, const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();

    std::cout << "Temporal reuse..." << std::endl;
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            // Clamp M and wSum values to a user-defined multiple of the current frame's to bound temporal creep
            Reservoir& current              = reservoirGrid[y][x];
            Reservoir& temporalPredecessor  = previousFrameGrid[y][x];
            size_t multipleCurrentM         = (features.temporalClampM * current.numSamples) + 1ULL;
            if (temporalPredecessor.numSamples > multipleCurrentM) {
                temporalPredecessor.wSum        *= multipleCurrentM / temporalPredecessor.numSamples;
                temporalPredecessor.numSamples  = multipleCurrentM;
            }

            // Combine to single reservoir
            Reservoir combined(current.outputSamples.size());
            combined.cameraRay                              = current.cameraRay;
            combined.hitInfo                                = current.hitInfo;
            std::array<Reservoir, 2ULL> pixelAndPredecessor = { current, temporalPredecessor };
            Reservoir::combineBiased(pixelAndPredecessor, combined, features); // Samples from temporal predecessor should be visible, no need to do unbiased combination
            reservoirGrid[y][x]                             = combined;
        }
    }
}

ReservoirGrid renderRayTracing(std::shared_ptr<ReservoirGrid> previousFrameGrid,
                               const Scene& scene, const Trackball& camera,
                               const EmbreeInterface& embreeInterface, Screen& screen,
                               const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();
    PrimaryHitGrid primaryHits  = genPrimaryRayHits(scene, camera, embreeInterface, screen, features);
    PixelGrid finalPixelColors(windowResolution.y,   std::vector<glm::vec3>(windowResolution.x, glm::vec3(0.0f)));
    ReservoirGrid currentGrid;

    for (uint32_t iteration = 0U; iteration < features.maxIterations; iteration++) {
        std::cout << "= Iteration " << iteration + 1 << std::endl;

        // Carry out ReSTIR steps
        currentGrid = genInitialSamples(primaryHits, scene, embreeInterface, features, screen.resolution());
        if (features.temporalReuse && previousFrameGrid)    { temporalReuse(currentGrid, *previousFrameGrid.get(), embreeInterface, screen, features); }
        if (features.spatialReuse)                          { spatialReuse(currentGrid, embreeInterface, screen, features); }

        // Final shading
        std::cout << "Shading final samples..." << std::endl;
        #ifdef NDEBUG
        #pragma omp parallel for schedule(guided)
        #endif
        for (int y = 0; y < windowResolution.y; y++) {
            for (int x = 0; x != windowResolution.x; x++) {
                // Accumulate shading from final sample(s) to running sum
                const Reservoir& reservoir  = currentGrid[y][x];
                finalPixelColors[y][x]      += finalShading(reservoir, reservoir.cameraRay, embreeInterface, features);
            }
        }

    }

    // Produce final color and return current frame's final grid for temporal reuse
    combineToScreen(screen, finalPixelColors, features);
    return currentGrid;
}
