#version 450

layout(set = 0, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec4 position_camera;
} camera_data;

layout(set = 0, binding = 1) uniform samplerCube texture_space;

layout(location = 0) in vec3 view_dir_in;
layout(location = 1) in vec3 camera_world_pos;

layout(location = 0) out vec4 color_out;

const float FADE_NEAR = 5.0;
const float FADE_FAR = 700.0;
const uint COLOR_X = 0xFF4444FF;
const uint COLOR_Y = 0xFF44FF44;
const uint COLOR_Z = 0xFFFF4444;

float drawGrid(vec2 coord, float gridSize) {
    vec2 gridCoord = coord / gridSize;
    vec2 fracPart = fract(gridCoord - 0.5) - 0.5;
    float minDist = min(abs(fracPart.x), abs(fracPart.y));
    return 1.0 - smoothstep(0.0, 0.1, minDist);
}

float intersectPlane(vec3 rayOrigin, vec3 rayDir, vec3 planeNormal) {
    float denom = dot(planeNormal, rayDir);
    if (abs(denom) < 1e-6) return -1.0;
    float t = -dot(planeNormal, rayOrigin) / denom;
    return (t > 0.001) ? t : -1.0;
}

void main() {
    vec3 rayDir = normalize(view_dir_in);
    vec3 rayOrigin = camera_world_pos;

    vec3 skyColor = texture(texture_space, rayDir).rgb;
    color_out = vec4(skyColor, 1.0);
    gl_FragDepth = 1.0;

    float closestDepth = 1.0;
    vec3 finalColor = skyColor;

    struct PlaneInfo {
        vec3 normal;
        bool isMajor;
    };

    PlaneInfo planes[3] = PlaneInfo[](
        PlaneInfo(vec3(1,0,0), false),
        PlaneInfo(vec3(0,0,1), false),
        PlaneInfo(vec3(0,1,0), true)
    );

    for (int i = 0; i < 3; ++i) {
        float t = intersectPlane(rayOrigin, rayDir, planes[i].normal);
        if (t <= 0.0) continue;

        vec3 hitPoint = rayOrigin + rayDir * t;

        if (!planes[i].isMajor) {
            bool valid = false;
            if (planes[i].normal == vec3(1,0,0) && abs(hitPoint.z) < 0.1) valid = true;
            if (planes[i].normal == vec3(0,0,1) && abs(hitPoint.x) < 0.1) valid = true;
            if (!valid) continue;
        }

        vec4 viewPos = camera_data.view * vec4(hitPoint, 1.0);
        vec4 clipPos = camera_data.projection * viewPos;
        float depth = clamp(clipPos.z / clipPos.w, 0.0, 1.0);;

        if (depth >= closestDepth) continue;

        if (planes[i].isMajor) {
            vec2 gridCoords = vec2(hitPoint.x, hitPoint.z);
            float grid1 = drawGrid(gridCoords, 1.0);
            float grid10 = drawGrid(gridCoords, 10.0);

            if (max(grid1, grid10) <= 0.01) {
                continue;
            }

            float fade = 1.0 - smoothstep(FADE_NEAR, FADE_FAR, t);
            float fadeDirs = 1.0 - smoothstep(FADE_NEAR, 300.0, t);
            float dx = abs(mod(hitPoint.x + 5.0, 10.0) - 5.0);
            float dz = abs(mod(hitPoint.z + 5.0, 10.0) - 5.0);
            const float lineWidth = 0.1;

            vec3 gridColor = skyColor;
            if (dx < lineWidth && dz < lineWidth) {
                gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Y).xyz, fadeDirs);
            } else if (dx < lineWidth) {
                gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Z).xyz, fadeDirs);
            } else if (dz < lineWidth) {
                gridColor = mix(skyColor, unpackUnorm4x8(COLOR_X).xyz, fadeDirs);
            } else {
                gridColor = mix(skyColor, vec3(0.5), fade);
            }

            finalColor = gridColor;
            closestDepth = depth;
        } else {
            float fade = 1.0 - smoothstep(FADE_NEAR, FADE_FAR, t);
            vec3 gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Y).xyz, fade);
            finalColor = gridColor;
            closestDepth = depth;
        }
    }

    color_out = vec4(finalColor, 1.0);
    gl_FragDepth = closestDepth;
}