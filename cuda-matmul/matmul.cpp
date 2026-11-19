#include <cstdio>

int main()
{
    constexpr int width = 20;
    constexpr int elements = width * width;
    float M[elements];
    float N[elements];
    float P[elements];

    for (int i = 0; i < elements; ++i) {
        M[i] = static_cast<float>(i + 1);
        N[i] = (i / width == i % width) ? 1.0f : 0.0f;
    }

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k) {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }

    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            std::printf("%5.0f", P[row * width + col]);
        }
        std::printf("\n");
    }
}
