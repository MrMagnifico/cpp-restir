
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


ReservoirGrid genInitialSamples(const Scene& scene, const Trackball& camera, const EmbreeInterface& embreeInterface, Screen& screen, const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();
    ReservoirGrid initialSamples(windowResolution.y, std::vector<Reservoir>(windowResolution.x, Reservoir(features.numSamplesInReservoir)));

    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            const glm::vec2 normalizedPixelPos { float(x) / float(windowResolution.x) * 2.0f - 1.0f,
                                                 float(y) / float(windowResolution.y) * 2.0f - 1.0f };
            const Ray cameraRay     = camera.generateRay(normalizedPixelPos);
            initialSamples[y][x]    = genCanonicalSamples(scene, embreeInterface, features, cameraRay);
        }
    }
    return initialSamples;
}

void spatialReuse(ReservoirGrid& reservoirGrid, const EmbreeInterface& embreeInterface, const Screen& screen, const Features& features) {
    // Uniform selection of neighbours in N pixel Manhattan distance radius
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(-features.spatialResampleRadius, features.spatialResampleRadius);

    glm::ivec2 windowResolution = screen.resolution();
    ReservoirGrid prevIteration = reservoirGrid;
    for (uint32_t pass = 0U; pass < features.spatialResamplingPasses; pass++) {
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
                   Screen& screen, const glm::vec2 motionVector, const Features& features) {
    glm::ivec2 windowResolution = screen.resolution();
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            // Clamp M and wSum values to a user-defined multiple of the current frame's to bound temporal creep
            // TODO: Add consideration of motion vectors
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
                               const glm::vec2 motionVector, const Features& features) {
    ReservoirGrid reservoirGrid = genInitialSamples(scene, camera, embreeInterface, screen, features);
    if (features.temporalReuse && previousFrameGrid)    { temporalReuse(reservoirGrid, *previousFrameGrid.get(), embreeInterface, screen, motionVector, features); }
    if (features.spatialReuse)                          { spatialReuse(reservoirGrid, embreeInterface, screen, features); }

    // Final shading
    glm::ivec2 windowResolution = screen.resolution();
    #ifdef NDEBUG
    #pragma omp parallel for schedule(guided)
    #endif
    for (int y = 0; y < windowResolution.y; y++) {
        for (int x = 0; x != windowResolution.x; x++) {
            // Compute shading from final sample(s)
            glm::vec3 finalColor(0.0f);
            const Reservoir& reservoir = reservoirGrid[y][x];
            for (const SampleData& sample : reservoir.outputSamples) {
                glm::vec3 sampleColor   = testVisibilityLightSample(sample.lightSample.position, embreeInterface, features, reservoir.cameraRay, reservoir.hitInfo)             ?
                                          computeShading(sample.lightSample.position, sample.lightSample.color, features, reservoir.cameraRay, reservoir.hitInfo)   :
                                          glm::vec3(0.0f);
                sampleColor             *= sample.outputWeight;
                finalColor              += sampleColor;
            }
            finalColor /= reservoir.outputSamples.size(); // Divide final shading value by number of samples

            // Apply tone mapping and set final pixel color
            if (features.enableToneMapping) { finalColor = exposureToneMapping(finalColor, features); }
            screen.setPixel(x, y, finalColor);
        }
    }

    // Return current frame's final grid for temporal reuse
    return reservoirGrid;
}
