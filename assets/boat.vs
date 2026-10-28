#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;
uniform mat4 mvp;
uniform samplerBuffer boatPose;
out vec3 worldPosition;
out vec3 normal;
out vec2 texCoord;
void main() {
    mat4 pose = mat4(texelFetch(boatPose,0),texelFetch(boatPose,1),texelFetch(boatPose,2),texelFetch(boatPose,3));
    vec4 world = pose * vec4(vertexPosition,1.0);
    worldPosition = world.xyz;
    normal = mat3(pose) * vertexNormal;
    texCoord = vertexTexCoord;
    gl_Position = mvp * world;
}
