#version 330

in vec3 worldPosition;
in vec3 normal;
uniform vec3 eyePosition;
out vec4 finalColor;

void main() {
    vec3 n = normalize(normal);
    vec3 view = normalize(eyePosition - worldPosition);
    vec3 light = normalize(vec3(-0.35, 0.85, -0.4));
    vec3 halfway = normalize(light + view);
    float diffuse = max(dot(n, light), 0.0);
    float specular = pow(max(dot(n, halfway), 0.0), 180.0);
    float fresnel = pow(1.0 - max(dot(n, view), 0.0), 4.0);
    vec3 reflection = reflect(-view, n);
    float sky = smoothstep(0.0, 0.8, reflection.y);
    vec3 blue = mix(vec3(0.015, 0.19, 0.31), vec3(0.035, 0.32, 0.48), sky);
    vec3 color = blue * (0.5 + 0.5 * diffuse) + fresnel * vec3(0.15, 0.22, 0.25);
    color += specular * vec3(0.9, 0.96, 1.0);
    finalColor = vec4(color, 1.0);
}
