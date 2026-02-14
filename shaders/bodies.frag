#version 450

layout(set = 0, binding = 0) uniform CameraData
{
    mat4 projection;
    mat4 view;

    vec4 position_camera;
} camera_data;

layout(set = 0, binding = 2) uniform sampler2D texture_body;

layout(location = 0) in vec3 position_in;
layout(location = 1) in vec3 texture_position_in;
layout(location = 2) in vec3 normal_in;
layout(location = 3) in vec3 color_in;

layout(location = 0) out vec4 color_out;

const vec3 accent = vec3(0.5);

void main()
{
    const vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));

    vec3 normal = normalize(normal_in);

    float diffuse = max(dot(normal, lightDir), 0.0);

    vec3 viewDir = normalize(camera_data.position_camera.xyz - position_in);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 8.0);
    vec3 specular = vec3(1.0) * spec;

    vec3 color_object = vec3(1.0, 1.0, 1.0);

    if (texture_position_in.x >= 0.0 && texture_position_in.y >= 0.0)
    {
        color_object = texture(texture_body, vec2(texture_position_in.x, 1.0 + texture_position_in.y * -1.0)).rgb;
    }
    else if (color_in.r >= 0.0 && color_in.g >= 0.0 && color_in.b >= 0.0)
    {
        color_object = color_in.rgb;
    }

    vec3 final_color = color_object * diffuse + specular + color_object * accent;

    color_out = vec4(final_color, 1.0);
}