#version 330

in vec3 worldPosition;
in vec3 normal;
uniform vec3 eyePosition;
uniform float simTime;
out vec4 finalColor;

// SHARED_SKY

void main() {
    vec3 n = normalize(normal);
    vec3 view = normalize(eyePosition - worldPosition);
    vec3 environment = skyRadiance(reflect(-view, n), simTime);
    float light = 0.5 + 0.5 * max(dot(n, sunDirection), 0.0);
    float highlight = pow(max(dot(n, normalize(view + sunDirection)), 0.0), 90.0) * sunVisibility(simTime);
    vec3 body = vec3(0.05, 0.42, 0.73) * light;
    finalColor = vec4(clamp(0.62 * body + 0.38 * environment + highlight * vec3(0.9, 0.9, 0.8), 0.0, 1.0), 1.0);
}
