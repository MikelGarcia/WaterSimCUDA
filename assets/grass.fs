#version 330

in vec3 worldPosition;
in vec3 normal;
out vec4 finalColor;
uniform sampler2D texture0;

void main() {
    vec3 color = texture(texture0, worldPosition.xz / 4.0).rgb;
    float light = 0.56 + 0.44 * max(dot(normalize(normal), normalize(vec3(-0.35, 0.85, -0.4))), 0.0);
    finalColor = vec4(color * light, 1.0);
}
