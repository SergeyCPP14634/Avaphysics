#version 450

layout(set = 0, binding = 0) uniform CameraData
{
    mat4 projection;
    mat4 view;

    vec4 position_camera;
} camera_data;

layout(set = 0, binding = 1) uniform RenderBodyData
{
    mat4 model;

    vec4 position_render_body;

    float shininess;
} render_body_data;

layout(location = 0) in vec3 position_in;
layout(location = 1) in vec3 texture_position_in;
layout(location = 2) in vec3 normal_in;
layout(location = 3) in vec3 color_in;

layout(location = 0) out vec3 position_out;
layout(location = 1) out vec3 texture_position_out;
layout(location = 2) out vec3 normal_out;
layout(location = 3) out vec3 color_out;

void main()
{
    gl_Position = camera_data.projection * camera_data.view * render_body_data.model * vec4(position_in, 1.0);

    gl_Position.y = -gl_Position.y;

    position_out = vec3(render_body_data.model * vec4(position_in, 1.0));

    texture_position_out = texture_position_in;

    normal_out = normalize(transpose(inverse(mat3(render_body_data.model))) * normal_in);

    color_out = color_in;
}