#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 pos_texture;

out vec2 frag_tex_coords;


out float visibility;
const float density = 0.00007;
const float gradient = 1.5;

void main() {
    mat4 view_ = view;
    view_[3] = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 pos = projection * view_ * vec4(position, 1.0);

    //vec4 pos = projection * view * vec4(position, 1.0);

    gl_Position = vec4(pos.xy, pos.z, pos.w);

    //gl_Position = pos.xyww;

    frag_tex_coords = pos_texture;

    vec4 posRelToCam = view * model * vec4(position, 1.0); //FOG
    float distance = length(posRelToCam.xyz);
    visibility = exp(-pow((distance * density), gradient));
    visibility = clamp(visibility, 0.0, 1.0);

}
