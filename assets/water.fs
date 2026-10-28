#version 330
// SHARED_SKY

in vec3 worldPosition;
in vec3 normal;
uniform vec3 eyePosition;
uniform mat4 captureView;
uniform mat4 captureProjection;
uniform sampler2D captureColor;
uniform sampler2D captureDepth;
uniform vec2 drawableSize;
uniform vec2 clipPlanes;
uniform float simTime;
uniform float reflectionWeight;
uniform samplerBuffer boatPose;
uniform vec2 boatWaterline[16];
uniform int boatWaterlineCount;

bool insideBoat(vec2 world) {
    if (boatWaterlineCount < 3) return false;
    vec4 forward = texelFetch(boatPose,0);
    vec4 transverse = texelFetch(boatPose,2);
    vec2 delta = world - texelFetch(boatPose,3).xz;
    vec2 local = vec2(dot(delta,forward.xz),dot(delta,transverse.xz));
    for (int i=0; i<boatWaterlineCount; ++i) {
        vec2 a=boatWaterline[i], edge=boatWaterline[(i+1)%boatWaterlineCount]-a;
        vec2 point=local-a;
        if (edge.x*point.y-edge.y*point.x < 0.0) return false;
    }
    return true;
}
out vec4 finalColor;

float viewDepth(float depth) {
    return 2.0 * clipPlanes.x * clipPlanes.y /
           (clipPlanes.y + clipPlanes.x - (2.0*depth - 1.0)*(clipPlanes.y - clipPlanes.x));
}

bool validStone(vec2 uv, float waterDepth, out float basinDepth) {
    basinDepth = 0.0;
    vec2 inset = 1.0 / drawableSize;
    if (any(lessThan(uv, inset)) || any(greaterThan(uv, 1.0 - inset))) return false;
    float depth = texture(captureDepth, uv).r;
    basinDepth = viewDepth(depth);
    return texture(captureColor, uv).a >= 1.0 && depth < 1.0 && basinDepth > waterDepth + 0.005;
}

void main() {
    if (insideBoat(worldPosition.xz)) discard;
    vec3 n = normalize(normal);
    vec3 view = normalize(eyePosition - worldPosition);
    vec4 eye = captureView * vec4(worldPosition, 1.0);
    vec4 projected = captureProjection * eye;
    vec2 base = projected.xy / projected.w * 0.5 + 0.5;
    vec3 displaced = worldPosition + 0.75 * vec3(n.x, 0, n.z) /
                     max(0.25, abs(dot(n, view)));
    vec4 shifted = captureProjection * captureView * vec4(displaced, 1.0);
    vec2 offset = clamp(shifted.xy / shifted.w * 0.5 + 0.5 - base, vec2(-0.02), vec2(0.02));
    float baseDepth = 0.0;
    bool baseValid = validStone(base, -eye.z, baseDepth);
    bool candidateValid = baseValid;
    float candidateDepth = baseDepth;
    for (int i = 1; i <= 4; ++i) {
        float depth = baseDepth;
        bool valid = validStone(base + offset * (float(i)/4.0), -eye.z, depth);
        candidateValid = candidateValid && valid && abs(depth - baseDepth) <= max(0.2, 0.1*baseDepth);
        candidateDepth = depth;
    }
    vec3 stone = vec3(0.30, 0.34, 0.36);
    float travel = 1.0;
    if (baseValid) {
        vec2 uv = candidateValid ? base + offset : base;
        stone = texture(captureColor, uv).rgb;
        travel = max(0.0, (candidateValid ? candidateDepth : baseDepth) + eye.z);
    }
    vec3 transmission = exp(-vec3(0.16, 0.07, 0.035) * travel);
    vec3 refraction = stone * transmission + vec3(0.035, 0.12, 0.17)*(1.0 - transmission);
    vec3 reflection = skyRadiance(reflect(-view, n), simTime);
    float highlight = pow(max(dot(n, normalize(sunDirection + view)), 0.0), 90.0);
    reflection = clamp(reflection + vec3(0.65, 0.61, 0.49)*highlight*sunVisibility(simTime), 0.0, 1.0);
    finalColor = vec4(reflectionWeight * reflection + (1.0-reflectionWeight) * refraction, 1.0);
}
