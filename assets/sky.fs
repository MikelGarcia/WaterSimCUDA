#version 330
// SHARED_SKY

uniform vec3 cameraForward;
uniform vec3 cameraRight;
uniform vec3 cameraUp;
uniform vec2 drawableSize;
uniform float logicalAspect;
uniform float tanHalfFov;
uniform float simTime;
out vec4 finalColor;

void main() {
    vec2 screen = 2.0 * gl_FragCoord.xy / drawableSize - 1.0;
    vec3 ray = cameraForward + tanHalfFov *
               (screen.x * logicalAspect * cameraRight + screen.y * cameraUp);
    finalColor = vec4(skyRadiance(ray, simTime), 1.0);
}
