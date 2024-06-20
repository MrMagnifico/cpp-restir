#include "texture.h"
#include <framework/image.h>

glm::vec3 acquireTexel(const Image& image, const glm::vec2& texCoord, const Features& features) {
    float dummyIntegral;
    float fractionalX = std::modf(texCoord.x, &dummyIntegral);
    float fractionalY = std::modf(texCoord.y, &dummyIntegral);
    if (fractionalX < 0.0f) { fractionalX = 1.0f + fractionalX; }
    if (fractionalY < 0.0f) { fractionalY = 1.0f + fractionalY; }  
    size_t discreteX = fractionalX * (image.width    - 1);
    size_t discreteY = fractionalY * (image.height   - 1);
    size_t memoryLoc = (discreteY * image.width) + discreteX;
    return image.pixels[memoryLoc];
}
