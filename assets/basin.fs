#version 330

in vec3 worldPosition;
in vec3 normal;
uniform int submergedOnly;
out vec4 finalColor;

void main() {
    if (submergedOnly != 0 && worldPosition.y > 0.0) discard;
    vec3 n = normalize(normal);
    vec3 face = abs(n);
    vec2 position = face.y > 0.5 ? worldPosition.xz :
                    (face.x > 0.5 ? worldPosition.zy : worldPosition.xy);
    vec2 stone = position / 0.8;
    stone.x += mod(floor(stone.y), 2.0) * 0.5;
    vec2 cell = floor(stone);
    vec2 edge = min(fract(stone), 1.0 - fract(stone));
    float variation = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
    float joint = smoothstep(0.012, 0.028, min(edge.x, edge.y));
    float grain = sin(dot(worldPosition, vec3(41.0, 53.0, 47.0))) * 0.008;
    vec3 color = vec3(0.44, 0.46, 0.45) * (0.91 + 0.18 * variation) + grain;
    color = mix(vec3(0.23, 0.25, 0.24), color, joint);
    float light = 0.56 + 0.44 * max(dot(n, normalize(vec3(-0.35, 0.85, -0.4))), 0.0);
    finalColor = vec4(color * light, 1.0);
}
