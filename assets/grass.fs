#version 330

in vec3 worldPosition;
in vec3 normal;
out vec4 finalColor;

void main() {
    vec2 p = worldPosition.xz;
    float patch = sin(p.x * 1.7 + sin(p.y * 0.8)) * sin(p.y * 1.3);
    float blades = sin(p.x * 127.0 + sin(p.y * 83.0)) * sin(p.y * 111.0);
    float detail = mix(blades, 0.0, clamp(length(fwidth(p)) * 20.0, 0.0, 1.0));
    vec3 color = vec3(0.13, 0.29, 0.075) * (1.0 + patch * 0.14 + detail * 0.10);
    float light = 0.56 + 0.44 * max(dot(normalize(normal), normalize(vec3(-0.35, 0.85, -0.4))), 0.0);
    finalColor = vec4(color * light, 1.0);
}
