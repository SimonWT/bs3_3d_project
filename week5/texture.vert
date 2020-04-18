#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 pos_texture;

out vec2 frag_tex_coords;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    frag_tex_coords = pos_texture;
}
