#version 450

layout(set = 0, binding = 0) uniform CameraData
{
    mat4 projection;
    mat4 view;

    vec4 position_camera;
} camera_data;

layout(location = 0) in vec3 position_in;
layout(location = 1) in vec3 texture_position_in;
layout(location = 2) in vec3 normal_in;
layout(location = 3) in vec3 color_in;

layout(location = 0) out vec3 view_dir_out;
layout(location = 1) out vec3 camera_world_pos;

void main()
{
    mat4 view_no_translation = mat4(mat3(camera_data.view));

    gl_Position = camera_data.projection * view_no_translation * vec4(position_in, 1.0);
    
    gl_Position.y = -gl_Position.y;
    
    view_dir_out = position_in;
    
    camera_world_pos = camera_data.position_camera.xyz;
}