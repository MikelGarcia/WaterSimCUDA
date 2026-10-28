#version 330
in vec3 worldPosition;
in vec3 normal;
in vec2 texCoord;
uniform sampler2D texture0;
uniform vec4 colDiffuse;
uniform vec3 eyePosition;
uniform int submergedOnly;
out vec4 finalColor;
void main() {
    if (submergedOnly != 0 && worldPosition.y > 0.0) discard;
    vec4 color = texture(texture0,texCoord) * colDiffuse;
    vec3 n = normalize(gl_FrontFacing ? normal : -normal);
    vec3 light = normalize(vec3(-0.35,0.85,-0.4));
    float diffuse = 0.50 + 0.50 * max(dot(n,light),0.0);
    vec3 halfway = normalize(light + normalize(eyePosition-worldPosition));
    float specular = pow(max(dot(n,halfway),0.0),48.0) * 0.22;
    finalColor = vec4(color.rgb * diffuse + specular,color.a);
}
