#version 330 core

// fragment position and normal of the fragment, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
in vec3 w_position, w_normal;   // in world coodinates

// light dir, in world coordinates
uniform vec3 light_dir;

// material properties
uniform vec3 k_d;

// world camera position
uniform vec3 w_camera_position;

out vec4 out_color;

void main() {
    // TODO: compute Lambert illumination
    out_color = vec4(k_d, 1);
}
