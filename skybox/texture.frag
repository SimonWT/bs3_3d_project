#version 330 core

uniform sampler2D diffuse_map;
in vec2 frag_tex_coords;
out vec4 out_color;

in float visibility;

void main() {
    out_color = texture(diffuse_map, frag_tex_coords);
    out_color = mix(vec4(vec3(0.61,0.87,1.61),1.0), out_color, visibility);
}

