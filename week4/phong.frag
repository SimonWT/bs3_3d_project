#version 330 core

// fragment position and normal of the fragment, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)
in vec3 w_position, w_normal;   // in world coodinates

// light dir, in world coordinates
uniform vec3 light_dir;

// material properties
uniform vec3 k_d, k_a, k_s;
uniform float s;

// world camera position
uniform vec3 w_camera_position;

out vec4 out_color;

void main() {
    // TODO: compute Lambert illumination
   // vec3 light = reflect(light_dir, w_normal);
   // out_color = vec4(k_d * dot(w_normal / length(w_normal), light), 1);

    vec3 n = normalize(w_normal);
    vec3 l = normalize(-light_dir);
    vec3 v = normalize(w_camera_position - w_position);
    vec3 r = reflect(-l, n);

    vec3 diffuse_color = k_d * max(dot(n, l), 0);
    vec3 specular_color = k_s * pow(max(dot(r, v), 0), s);

    out_color = vec4(k_a, 1) + vec4(diffuse_color, 1) + vec4(specular_color, 1);

}
