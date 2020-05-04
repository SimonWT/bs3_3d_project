#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    //Cancel translation
    mat4 view_ = view;
    view_[3] = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 pos = projection * view_ * vec4(aPos, 1.0);

    //gl_Position = vec4(pos.xy, pos.w, pos.w);
    gl_Position = pos.xyww;

    TexCoords = aPos;

    //gl_Position = projection * view * vec4(aPos, 1.0);
}  
