#version 330

in vec3 vertexPosition;
in vec3 vertexNormal;
uniform mat4 matModel;
uniform mat4 mvp;
out vec3 worldPosition;
out vec3 normal;

void main() {
    worldPosition = vec3(matModel * vec4(vertexPosition, 1.0));
    normal = transpose(inverse(mat3(matModel))) * vertexNormal;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
