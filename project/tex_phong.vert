#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 pos_texture;

uniform mat4 model, view, projection;

// position and normal for the fragment shader, in WORLD coordinates
// (you can also compute in VIEW coordinates, your choice! rename variables)

out vec3 w_position, w_normal;   // in world coordinates

//for texture
out vec2 frag_tex_coords;

//for FOG
out float visibility;
const float density = 0.007;
const float gradient = 1.5;

void main() {
    //gl_Position = projection * view * model * vec4(position, 1.0);

    vec4 w_position4 = model * vec4(position, 1.0);
    gl_Position = projection * view * w_position4;

    // fragment position in world coordinates
    w_position = w_position4.xyz / w_position4.w;  // dehomogenize

    // fragment normal in world coordinates
    mat3 nit_matrix = transpose(inverse(mat3(model)));
    w_normal = normalize(nit_matrix * normal);

    frag_tex_coords = pos_texture;

    //FOG
    vec4 posRelToCam = view * model * vec4(position, 1.0);
    float distance = length(posRelToCam.xyz);
    visibility = exp(-pow((distance * density), gradient));
    visibility = clamp(visibility, 0.0, 1.0);
}
