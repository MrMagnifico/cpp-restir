#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/glm.hpp>
DISABLE_WARNINGS_POP()

#include <rendering/reservoir.h>
#include <rendering/shading.h>
#include <utils/utils.h>


void Reservoir::update(LightSample sample, float weight) {
    numSamples  += 1ULL;
    wSum        += weight;
    for (SampleData& outputSample : outputSamples) {
        float uniformRandom = linearMap(static_cast<float>(rand()), 0.0f, RAND_MAX, 0.0f, 1.0f);
        if (uniformRandom < (weight / wSum)) { 
            outputSample.lightSample.position   = sample.position;
            outputSample.lightSample.color      = sample.color;
        }
    }
}

void Reservoir::combineBiased(const std::span<Reservoir>& reservoirStream, Reservoir& finalReservoir, const Features& features) {
    size_t totalSampleCount = 0ULL;
    for (const Reservoir& reservoir : reservoirStream) {
        totalSampleCount += reservoir.numSamples * reservoir.outputSamples.size();
        for (const SampleData& reservoirSample : reservoir.outputSamples) {
            float pdfValue = targetPDF(reservoirSample.lightSample, finalReservoir.cameraRay, finalReservoir.hitInfo, features);
            finalReservoir.update(reservoirSample.lightSample, pdfValue * reservoirSample.outputWeight * reservoir.numSamples);
        }
    }

    finalReservoir.numSamples = totalSampleCount;
    for (SampleData& finalReservoirSample : finalReservoir.outputSamples) {
        float finalPdfValue = targetPDF(finalReservoirSample.lightSample, finalReservoir.cameraRay, finalReservoir.hitInfo, features);
        if (finalPdfValue == 0.0f)  { finalReservoirSample.outputWeight = 0.0f; }
        else                        { finalReservoirSample.outputWeight = (1.0f / finalPdfValue) * 
                                                                          (1.0f / finalReservoir.numSamples) *
                                                                          finalReservoir.wSum; }
    }
}

void Reservoir::combineUnbiased(const std::span<Reservoir>& reservoirStream, Reservoir& finalReservoir, const BvhInterface& bvh, const Features& features) {
    size_t totalSampleCount = 0ULL;
    for (const Reservoir& reservoir : reservoirStream) {
        totalSampleCount    += reservoir.numSamples * reservoir.outputSamples.size();
        for (const SampleData& reservoirSample : reservoir.outputSamples) {
            float pdfValue      = targetPDF(reservoirSample.lightSample, finalReservoir.cameraRay, finalReservoir.hitInfo, features);
            finalReservoir.update(reservoirSample.lightSample, pdfValue * reservoirSample.outputWeight * reservoir.numSamples);
        }
    }
    finalReservoir.numSamples = totalSampleCount;

    std::vector<size_t> numValidSamples(finalReservoir.outputSamples.size(), 0ULL);
    for (const Reservoir& reservoir : reservoirStream) {
        for (size_t outputSampleIdx = 0ULL; outputSampleIdx < finalReservoir.outputSamples.size(); outputSampleIdx++) {
            const SampleData& finalReservoirSample  = finalReservoir.outputSamples[outputSampleIdx]; 
            float pdfValue                          = targetPDF(finalReservoirSample.lightSample, reservoir.cameraRay, reservoir.hitInfo, features);
            if (features.spatialReuseVisibilityCheck)   { pdfValue *= testVisibilityLightSample(finalReservoirSample.lightSample.position, bvh, features, reservoir.cameraRay, reservoir.hitInfo); }
            if (pdfValue > 0.0f)                        { numValidSamples[outputSampleIdx] += reservoir.numSamples * reservoir.outputSamples.size(); }
        }
    }
    
    for (size_t outputSampleIdx = 0ULL; outputSampleIdx < finalReservoir.outputSamples.size(); outputSampleIdx++) {
        SampleData& finalReservoirSample    = finalReservoir.outputSamples[outputSampleIdx];
        float finalPdfValue                 = targetPDF(finalReservoirSample.lightSample, finalReservoir.cameraRay, finalReservoir.hitInfo, features);
        if (finalPdfValue == 0.0f || numValidSamples[outputSampleIdx] == 0ULL)  { finalReservoirSample.outputWeight = 0.0f; }
        else                                                                    { finalReservoirSample.outputWeight = (1.0f / finalPdfValue) * 
                                                                                                                      (1.0f / numValidSamples[outputSampleIdx]) *
                                                                                                                      finalReservoir.wSum; }
    }
}

float targetPDF(const LightSample& sample, const Ray& cameraRay, const HitInfo& hitInfo, const Features& features) {
    glm::vec3 bsdf = computeShading(sample.position, sample.color, features, cameraRay, hitInfo);
    return glm::length(bsdf);
}
