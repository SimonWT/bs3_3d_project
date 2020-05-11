#version 330 core

//uniform vec3 color;
uniform sampler2D diffuse_map;
in vec2 frag_tex_coords;
out vec4 outColor;

void main() {
    outColor = vec4(0.4, 0.6, 3, 1);
}
