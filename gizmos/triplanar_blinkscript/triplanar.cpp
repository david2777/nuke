kernel Triplanar : public ImageComputationKernel <ePixelWise> {
    Image<eRead> imBeauty;
    Image<eRead, eAccessRanged2D, eEdgeClamped> imWorldPos;
    Image<eRead> imWorldNormal;
    Image<eRead, eAccessRandom, eEdgeClamped> imTextureX;
    Image<eRead, eAccessRandom, eEdgeClamped> imTextureY;
    Image<eRead, eAccessRandom, eEdgeClamped> imTextureZ;
    Image<eWrite> dst;

    param:
        // Global params
        int axisOutput;
        bool premult;
        bool useTextureAlpha;

        // Global transform params
        float2 offsetGlobal;
        float scaleGlobal;
        float rotateAngleGlobal;

        // Blend and filtering params
        float blendExponent;
        int filterMode;
        int anisotropicSampleCount;

        // Per axis transform params
        float2 offsetX;
        float angleX;
        float scaleX;

        float2 offsetY;
        float angleY;
        float scaleY;

        float2 offsetZ;
        float angleZ;
        float scaleZ;

    local:
        // Constant lower weight threshold to remove tiny values
        float kWeightThreshold;

        // Cached texture sizes for UV to pixel space conversions
        int2 textureX_Size;
        int2 textureY_Size;
        int2 textureZ_Size;

        // Pre-calculated sin and cos values for rotations
        float rotateX_Sin;
        float rotateX_Cos;

        float rotateY_Sin;
        float rotateY_Cos;

        float rotateZ_Sin;
        float rotateZ_Cos;

    void define() {
        // Define parameters in node panel 
        defineParam(premult, "Premult", true);
        defineParam(useTextureAlpha, "Apply Texture Alpha", false);
        defineParam(axisOutput, "Output Axis", 3);

        defineParam(offsetGlobal, "Global Translate", float2(0.0f, 0.0f));
        defineParam(rotateAngleGlobal, "Global Rotate", 0.0f);
        defineParam(scaleGlobal, "Global Scale", 1.0f);

        defineParam(blendExponent, "Blend Exponent", 1.0f);
        defineParam(filterMode, "Filter Mode", 0);
        defineParam(anisotropicSampleCount, "Anisotropic Samples", 8);

        defineParam(offsetX, "X Axis Translate", float2(0.0f, 0.0f));
        defineParam(angleX, "X Axis Rotate", 0.0f);
        defineParam(scaleX, "X Axis Scale", 1.0f);

        defineParam(offsetY, "Y Axis Translate", float2(0.0f, 0.0f));
        defineParam(angleY, "Y Axis Rotate", 0.0f);
        defineParam(scaleY, "Y Axis Scale", 1.0f);

        defineParam(offsetZ, "Z Axis Translate", float2(0.0f, 0.0f));
        defineParam(angleZ, "Z Axis Rotate", 0.0f);
        defineParam(scaleZ, "Z Axis Scale", 1.0f);
    }

    // Convert degrees to radians
    float angleToRad(float angle) {
        return angle * 3.1415926535f / 180.0f;
    }

    void init() {
        // Define a lower threshold for the weights at which point it rounds down to 0
        kWeightThreshold = 0.0001f;

        // Ensure filter mode is between 0 and 2
        filterMode = max(filterMode, 0);
        filterMode = min(filterMode, 2);

        // Set our ranged access to one pixel down and one pixel right for anisotropic filtering
        imWorldPos.setRange(0, 1, 0, 1);

        // Get texture widths and heights to perform UV to pixel space conversions
        textureX_Size = int2(imTextureX.bounds.height(), imTextureX.bounds.width());
        textureY_Size = int2(imTextureY.bounds.height(), imTextureY.bounds.width());
        textureZ_Size = int2(imTextureZ.bounds.height(), imTextureZ.bounds.width());

        // Setup rotations, sin and cos are expecting radians
        float angleRad = angleToRad(angleX + rotateAngleGlobal);
        rotateX_Sin = sin(angleRad);
        rotateX_Cos = cos(angleRad);

        angleRad = angleToRad(angleY + rotateAngleGlobal);
        rotateY_Sin = sin(angleRad);
        rotateY_Cos = cos(angleRad);

        angleRad = angleToRad(angleZ + rotateAngleGlobal);
        rotateZ_Sin = sin(angleRad);
        rotateZ_Cos = cos(angleRad);

        // Calculate offsets in pixel space
        offsetX = (offsetX + offsetGlobal) / textureX_Size;
        offsetY = (offsetY + offsetGlobal) / textureY_Size;
        offsetZ = (offsetZ + offsetGlobal) / textureZ_Size;

        // Calculate scales
        scaleX = scaleX * scaleGlobal;
        scaleY = scaleY * scaleGlobal;
        scaleZ = scaleZ * scaleGlobal;
    }

    // Compute the weights based on the normal data
    float3 computeWeights(float4 normal) {
        // Get absolute value of the normal data
        float4 absNormal = fabs(normal);

        // Check that there is any data on this sample
        if ((absNormal.x == 0.0f) and (absNormal.y == 0.0f) and (absNormal.z == 0.0f)) {
            return float3(0.0f);
        }

        // Apply exponent
        float weightX = pow(absNormal.x, blendExponent);
        float weightY = pow(absNormal.y, blendExponent);
        float weightZ = pow(absNormal.z, blendExponent);

        // Normalize weights
        float total = weightX + weightY + weightZ;
        weightX = weightX / total;
        weightY = weightY / total;
        weightZ = weightZ / total;

        // Cull weights below the lower weight threshold, this is more of a preference but I
        // don't like seeing tiny values when inspecting the result.
        if (weightX < kWeightThreshold) {
            weightX = 0.0f;
        } if (weightY < kWeightThreshold) {
            weightY = 0.0f;
        } if (weightZ < kWeightThreshold) {
            weightZ = 0.0f;
        }

        // Return our weights in the form of a float3
        return float3(weightX, weightY, weightZ);
    }

    // Calculate UV coordinates from two World Pos values and an axis where 0 = X, 1 = Y, and 2 = Z.
    float2 computeUVs(float x, float y, int axis) {
        axis = max(axis, 0);
        axis = min(axis, 2);

        // Get the scale, offset, and rotation for the given axis
        float scaleLocal;
        float2 offsetLocal;
        float angleCosLocal;
        float angleSinLocal;
        if (axis == 0) {
            scaleLocal = scaleX;
            offsetLocal = offsetX;
            angleCosLocal = rotateX_Cos;
            angleSinLocal = rotateX_Sin;
        } if (axis == 1) {
            scaleLocal = scaleY;
            offsetLocal = offsetY;
            angleCosLocal = rotateY_Cos;
            angleSinLocal = rotateY_Sin;
        } if (axis == 2) {
            scaleLocal = scaleZ;
            offsetLocal = offsetZ;
            angleCosLocal = rotateZ_Cos;
            angleSinLocal = rotateZ_Sin;
        }

        // Pack our UVs into a float2 vector, applying the scale and offset
        float2 uv = float2((x * scaleLocal), (y * scaleLocal)) - offsetLocal; 

        // Rotate UVs
        uv = float2(uv.x * angleCosLocal - uv.y * angleSinLocal, uv.x * angleSinLocal + uv.y * angleCosLocal);

        // Extract the fractional range of each UV to allow for tiling using a basic fract function
        uv = float2((uv.x - floor(uv.x)), (uv.y - floor(uv.y)));

        return uv;
    }

    // Accounts for UV wrapping by adjusting differential values when they go beyond 0.5 in magnitude
    float wrappedDiff(float a, float b) {
        float diff = a - b;
        if (diff > 0.5f) {
            diff -= 1.0f;
        } if (diff < -0.5f) {
            diff += 1.0f;
        }
        return diff;
    }

    // Accounts for UV wrapping by adjusting differential values when they go beyond 0.5 in magnitude
    float2 wrappedDiff2(float2 uv1, float2 uv2) {
        return float2(wrappedDiff(uv1.x, uv2.x), wrappedDiff(uv1.y, uv2.y));
    }

    // Calculate the derivatives for a one pixel offset of the position data
    float2 computeUVDerivatives(float2 uv, float xOffset, float yOffset, int axis) {
        float2 uvOffset = computeUVs(xOffset, yOffset, axis);
        return wrappedDiff2(uv, uvOffset);
    }

    // Sample the texture using a linear or point method, fast but noisy
    float4 sampleLinear(float2 uv, int axis) {
        axis = max(axis, 0);
        axis = min(axis, 2);

        if (axis == 0) {
            uv = uv * textureX_Size;
            return imTextureX(uv.x, uv.y);
        } if (axis == 1) {
            uv = uv * textureY_Size;
            return imTextureY(uv.x, uv.y);
        } else {
            uv = uv * textureZ_Size;
            return imTextureZ(uv.x, uv.y);
        }

    }

    // Sample the texture using a bilinear method, slower but higher quality
    float4 sampleBilinear(float2 uv, int axis) {
        axis = max(axis, 0);
        axis = min(axis, 2);

        if (axis == 0) {
            uv = uv * textureX_Size;
            return bilinear(imTextureX, uv.x, uv.y);
        } if (axis == 1) {
            uv = uv * textureY_Size;
            return bilinear(imTextureY, uv.x, uv.y);
        } else {
            uv = uv * textureZ_Size;
            return bilinear(imTextureZ, uv.x, uv.y);
        }

    }

    // Sample the texture using anisotropic method, slowest but highest quality, especially on oblique angles
    float4 sampleAnisotropic(float2 uv, float4 posRight, float4 posDown, int axis) {
        axis = max(axis, 0);
        axis = min(axis, 2);
        
        // Compute the UV derivatives for the right and down vectors
        float2 uv_RightD = 0.0f;
        float2 uv_DownD = 0.0f;

        if (axis == 0) {
            uv_RightD = computeUVDerivatives(uv, posRight.y, posRight.z, 0);
            uv_DownD = computeUVDerivatives(uv, posDown.y, posDown.z, 0);
        } if (axis == 1) {
            uv_RightD = computeUVDerivatives(uv, posRight.x, posRight.z, 1);
            uv_DownD = computeUVDerivatives(uv, posDown.x, posDown.z, 1);
        } if (axis == 2) {
            uv_RightD = computeUVDerivatives(uv, posRight.x, posRight.y, 2);
            uv_DownD = computeUVDerivatives(uv, posDown.x, posDown.y, 2);
        }

        // Get the length of the derivative
        float len_uv_RightD = length(uv_RightD);
        float len_uv_DownD = length(uv_DownD);

        // Find the dominant derivative (the direction with the greater change between pixels)
        float2 majorDir;
        float maxDerivative;
        if (len_uv_RightD > len_uv_DownD) {
            majorDir = normalize(uv_RightD);
            maxDerivative = len_uv_RightD;
        } else {
            majorDir = normalize(uv_DownD);
            maxDerivative = len_uv_DownD;
        }

        // Calculate our step size, center offset, and initialize our output
        float step = maxDerivative / (anisotropicSampleCount - 1);
        float centerOffset = (anisotropicSampleCount - 1) * 0.5;
        float4 result = float4(0.0f);

        // Step through our sample count and accumulate linear samples on each iteration, slowly moving in the direction of the offset 
        for (int i = 0; i < anisotropicSampleCount; i++) {
            float offset = (i - centerOffset) * step;
            float2 sampleUV = uv + majorDir * offset;
            float4 sampleColor = float4(0.0f);
            if (axis == 0) {
                sampleUV = sampleUV * textureX_Size;
                sampleColor = imTextureX(sampleUV.x, sampleUV.y);
            } if (axis == 1) {
                sampleUV = sampleUV * textureY_Size;
                sampleColor = imTextureY(sampleUV.x, sampleUV.y);
            } if (axis == 2) {
                sampleUV = sampleUV * textureZ_Size;
                sampleColor = imTextureZ(sampleUV.x, sampleUV.y);
            }
            result = result + sampleColor;
        }

        // Normalize our result
        return result / anisotropicSampleCount;
    }

    // The actual kernel
    void process() {
        // Read image data inputs
        float4 pos = imWorldPos(0, 0);
        float4 normal = imWorldNormal();

        // Compute weights
        float3 weights = computeWeights(normal);

        // Calculate UVs
        float2 uvX = computeUVs(pos.y, pos.z, 0);
        float2 uvY = computeUVs(pos.x, pos.z, 1);
        float2 uvZ = computeUVs(pos.x, pos.y, 2);

        float4 texSampleX = float4(0.0f);
        float4 texSampleY = float4(0.0f);
        float4 texSampleZ = float4(0.0f);

        if (filterMode == 0) { // Linear Sampling
            texSampleX = sampleLinear(uvX, 0);
            texSampleY = sampleLinear(uvY, 1);
            texSampleZ = sampleLinear(uvZ, 2);
        } if (filterMode == 1) { // Bilinear Sampling
            texSampleX = sampleBilinear(uvX, 0);
            texSampleY = sampleBilinear(uvY, 1);
            texSampleZ = sampleBilinear(uvZ, 2);
        } if (filterMode == 2) { // Anisotropic Sampling
            // Get the position data for the right and down pixels
            float4 posRight = imWorldPos(1, 0);
            float4 posDown = imWorldPos(0, 1);

            // Gather the samples
            texSampleX = sampleAnisotropic(uvX, posRight, posDown, 0);
            texSampleY = sampleAnisotropic(uvY, posRight, posDown, 1);
            texSampleZ = sampleAnisotropic(uvZ, posRight, posDown, 2);
        }

        // Compute the output by multiplying the texture sample by the output weights
        float4 output = float4(0.0f);
        if (axisOutput <= 0) { // X Axis Only
            output = (texSampleX * weights.x);
            if (! useTextureAlpha) {
                output.w = weights.x;
            }
        } if (axisOutput == 1) { // Y Axis Only
            output = (texSampleY * weights.y);
            if (! useTextureAlpha) {
                output.w = weights.y;
            }
        } if (axisOutput == 2) { // Z Axis Only
            output = (texSampleZ * weights.z);
            if (! useTextureAlpha) {
                output.w = weights.z;
            }
        } if (axisOutput >= 3) { // All Axis
            output = (texSampleX * weights.x) + (texSampleY * weights.y) + (texSampleZ * weights.z);
            // Compute alpha, we're not fully normalizing weights so we just compute a rough alpha here
            // which we can multiply by below
            if ((! useTextureAlpha) && (output.x + output.y + output.z) > 0.0f) {
                output.w = 1.0f;
            }
        }

        // Premult
        if (premult) {
            // Read beauty and mult by original alpha
            float4 beauty = imBeauty();
            output = output * beauty.w;

            // If applying by the texture alpha, mult by that as well
            if (useTextureAlpha) {
                float alpha = output.w;
                output = output * alpha;
                output.w = alpha;
            }
        }

        // Write to the output
        dst() = output;
    }
};