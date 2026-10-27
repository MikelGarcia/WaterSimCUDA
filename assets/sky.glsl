const vec3 sunDirection = normalize(vec3(-0.35, 0.85, -0.4));

float cloudNoise(vec2 p) {
    vec2 cell = floor(p), f = fract(p);
    f = f*f*(3.0 - 2.0*f);
    vec4 h = fract(sin(vec4(dot(cell, vec2(127.1, 311.7)),
                              dot(cell + vec2(1, 0), vec2(127.1, 311.7)),
                              dot(cell + vec2(0, 1), vec2(127.1, 311.7)),
                              dot(cell + vec2(1, 1), vec2(127.1, 311.7)))) * 43758.5453);
    return mix(mix(h.x, h.y, f.x), mix(h.z, h.w, f.x), f.y);
}

float cloudCoverage(vec3 direction, float simTime) {
    float angle = 1.8 + 0.015 * simTime;
    vec2 p = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * direction.xz;
    p = 3.0 * p / (max(direction.y, 0.0) + 0.35) + vec2(8.3, 4.7);
    float field = 0.57*cloudNoise(p) + 0.28*cloudNoise(p*2.03) + 0.15*cloudNoise(p*4.07);
    return smoothstep(0.35, 0.69, field) * smoothstep(-0.08, 0.18, direction.y);
}

float sunVisibility(float simTime) {
    return 1.0 - 0.96 * cloudCoverage(sunDirection, simTime);
}

vec3 skyRadiance(vec3 worldDirection, float simTime) {
    vec3 d = normalize(worldDirection);
    vec3 atmosphere = mix(vec3(0.63, 0.74, 0.81), vec3(0.13, 0.36, 0.64),
                          pow(max(d.y, 0.0), 0.55));
    float cloud = cloudCoverage(d, simTime);
    vec3 cloudColor = mix(vec3(0.80, 0.84, 0.87), vec3(0.29, 0.34, 0.40), smoothstep(0.5, 1.0, cloud));
    vec3 color = mix(atmosphere, cloudColor, cloud * 0.94);
    float sun = smoothstep(cos(0.024), cos(0.009), dot(d, sunDirection));
    color = mix(color, vec3(1.0, 0.95, 0.79), sun * sunVisibility(simTime));
    return clamp(color, 0.0, 1.0);
}
