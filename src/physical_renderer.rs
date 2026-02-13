extern crate nalgebra_glm as glm;
extern crate sdl3;

use sdl3::event::Event;
use std::mem::*;

use crate::body::*;
use crate::camera::*;
use crate::gui::*;
use crate::vulkan_renderer::vulkan_logic::*;
use crate::vulkan_renderer::*;

mod vs_space {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
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
        "
    }
}

mod fs_space {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(set = 0, binding = 0) uniform CameraData {
                mat4 projection;
                mat4 view;
                vec4 position_camera;
            } camera_data;

            layout(set = 0, binding = 1) uniform samplerCube texture_space;

            layout(location = 0) in vec3 view_dir_in;
            layout(location = 1) in vec3 camera_world_pos;

            layout(location = 0) out vec4 color_out;

            const float FADE_NEAR = 5.0;
            const float FADE_FAR = 700.0;
            const uint COLOR_X = 0xFF4444FF;
            const uint COLOR_Y = 0xFF44FF44;
            const uint COLOR_Z = 0xFFFF4444;

            float drawGrid(vec2 coord, float gridSize) {
                vec2 gridCoord = coord / gridSize;
                vec2 fracPart = fract(gridCoord - 0.5) - 0.5;
                float minDist = min(abs(fracPart.x), abs(fracPart.y));
                return 1.0 - smoothstep(0.0, 0.1, minDist);
            }

            float intersectPlane(vec3 rayOrigin, vec3 rayDir, vec3 planeNormal) {
                float denom = dot(planeNormal, rayDir);
                if (abs(denom) < 1e-6) return -1.0;
                float t = -dot(planeNormal, rayOrigin) / denom;
                return (t > 0.001) ? t : -1.0;
            }

            void main() {
                vec3 rayDir = normalize(view_dir_in);
                vec3 rayOrigin = camera_world_pos;

                vec3 skyColor = texture(texture_space, rayDir).rgb;
                color_out = vec4(skyColor, 1.0);
                gl_FragDepth = 1.0;

                float closestDepth = 1.0;
                vec3 finalColor = skyColor;

                struct PlaneInfo {
                    vec3 normal;
                    bool isMajor;
                };

                PlaneInfo planes[3] = PlaneInfo[](
                    PlaneInfo(vec3(1,0,0), false),
                    PlaneInfo(vec3(0,0,1), false),
                    PlaneInfo(vec3(0,1,0), true)
                );

                for (int i = 0; i < 3; ++i) {
                    float t = intersectPlane(rayOrigin, rayDir, planes[i].normal);
                    if (t <= 0.0) continue;

                    vec3 hitPoint = rayOrigin + rayDir * t;

                    if (!planes[i].isMajor) {
                        bool valid = false;
                        if (planes[i].normal == vec3(1,0,0) && abs(hitPoint.z) < 0.1) valid = true;
                        if (planes[i].normal == vec3(0,0,1) && abs(hitPoint.x) < 0.1) valid = true;
                        if (!valid) continue;
                    }

                    vec4 viewPos = camera_data.view * vec4(hitPoint, 1.0);
                    vec4 clipPos = camera_data.projection * viewPos;
                    float depth = clamp(clipPos.z / clipPos.w, 0.0, 1.0);;

                    if (depth >= closestDepth) continue;

                    if (planes[i].isMajor) {
                        vec2 gridCoords = vec2(hitPoint.x, hitPoint.z);
                        float grid1 = drawGrid(gridCoords, 1.0);
                        float grid10 = drawGrid(gridCoords, 10.0);

                        if (max(grid1, grid10) <= 0.01) {
                            continue;
                        }

                        float fade = 1.0 - smoothstep(FADE_NEAR, FADE_FAR, t);
                        float fadeDirs = 1.0 - smoothstep(FADE_NEAR, 300.0, t);
                        float dx = abs(mod(hitPoint.x + 5.0, 10.0) - 5.0);
                        float dz = abs(mod(hitPoint.z + 5.0, 10.0) - 5.0);
                        const float lineWidth = 0.1;

                        vec3 gridColor = skyColor;
                        if (dx < lineWidth && dz < lineWidth) {
                            gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Y).xyz, fadeDirs);
                        } else if (dx < lineWidth) {
                            gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Z).xyz, fadeDirs);
                        } else if (dz < lineWidth) {
                            gridColor = mix(skyColor, unpackUnorm4x8(COLOR_X).xyz, fadeDirs);
                        } else {
                            gridColor = mix(skyColor, vec3(0.5), fade);
                        }

                        finalColor = gridColor;
                        closestDepth = depth;
                    } else {
                        float fade = 1.0 - smoothstep(FADE_NEAR, FADE_FAR, t);
                        vec3 gridColor = mix(skyColor, unpackUnorm4x8(COLOR_Y).xyz, fade);
                        finalColor = gridColor;
                        closestDepth = depth;
                    }
                }

                color_out = vec4(finalColor, 1.0);
                gl_FragDepth = closestDepth;
            }
        "
    }
}

mod vs_bodies {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
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
        "
    }
}

mod fs_bodies {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
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
        "
    }
}

struct SpaceRenderer {
    vertex_buffer_space: Buffer,
    texture_space: Image,
    texture_view_space: ImageView,
    sampler_space: Sampler,
    descriptor_set_layout_space: DescriptorSetLayout,
    descriptor_set_space: DescriptorSet,
    pipeline_layout_space: PipelineLayout,
    pipeline_space: Pipeline,
    space_box_vertexes: Vec<Vertex>,
}

struct BodyRenderer {
    uniform_buffer_body: Buffer,
    texture_body: Image,
    texture_view_body: ImageView,
    descriptor_set_body: DescriptorSet,
}

pub struct PhysicalRenderer {
    pub sdl_context: sdl3::Sdl,
    pub window: sdl3::video::Window,
    device: Device,
    swapchain: Swapchain,
    render_pass: RenderPass,
    framebuffers: Vec<Framebuffer>,
    renderer: Renderer,
    imgui_context: ImGuiContext,
    uniform_buffer_camera: Buffer,

    vertex_buffers_bodies: Vec<Buffer>,
    index_buffers_bodies: Vec<Buffer>,
    indices_counts_bodies: Vec<u32>,
    sampler_bodies: Sampler,
    descriptor_set_layout_bodies: DescriptorSetLayout,
    pipeline_layout_bodies: PipelineLayout,
    pipeline_bodies: Pipeline,

    space_renderer: SpaceRenderer,
    bodies_renderer: Vec<BodyRenderer>,
}

impl PhysicalRenderer {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let space_box_vertexes = vec![
            Vertex {
                position: glm::vec3(-1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, 1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, 1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
            Vertex {
                position: glm::vec3(-1.0, -1.0, -1.0),
                texture_position: glm::vec3(0.0, 0.0, 0.0),
                normal: glm::vec3(0.0, 0.0, 0.0),
                color: glm::vec3(0.0, 0.0, 0.0),
            },
        ];

        let sdl_context = sdl3::init().map_err(|_| "Failed to create sdl context")?;
        let video_subsystem = sdl_context
            .video()
            .map_err(|err| format!("Failed to create sdl video subsystem {}", err))?;

        let window = video_subsystem
            .window("Physical", width, height)
            .resizable()
            .vulkan()
            .build()
            .map_err(|e| e.to_string())?;

        video_subsystem.text_input().start(&window);

        let instance = Instance::new(InstanceConfig {
            app_name: "Physical".to_string(),
            engine_name: "PhysicalEngine".to_string(),
        })?;

        let surface = Surface::new(SurfaceConfig {
            instance: instance.clone(),
            window: Some(window.clone()),
        })?;

        let physical_device = PhysicalDevice::new(PhysicalDeviceConfig {
            instance: instance.clone(),
            surface: surface.clone(),
            priority_gpu: PhysicalDevicePriority::Nvidia,
        })?;

        let device = Device::new(DeviceConfig {
            physical_device: physical_device.clone(),
        })?;

        let swapchain = Swapchain::new(SwapchainConfig {
            surface: surface.clone(),
            physical_device: physical_device.clone(),
            device: device.clone(),
            present_mode: PresentMode::Fifo,
            width,
            height,
        })?;

        let swapchain_image_views: Vec<ImageView> = swapchain
            .images()
            .iter()
            .map(|image| {
                ImageView::new(ImageViewConfig {
                    image: image.clone(),
                    image_view_type: ImageViewType::TwoD,
                    image_aspect: ImageAspect::Color,
                    index_layer: 0,
                    count_layers: 1,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let depth_image = Image::new(ImageConfig {
            device: device.clone(),
            image_flag: ImageFlag::TwoD,
            image_format: ImageFormat::D32Sfloat,
            image_usages: vec![ImageUsage::Depth],
            count_layers: 1,
            width: swapchain.width(),
            height: swapchain.height(),
        })?;

        let depth_image_view = ImageView::new(ImageViewConfig {
            image: depth_image.clone(),
            image_view_type: ImageViewType::TwoD,
            image_aspect: ImageAspect::Depth,
            index_layer: 0,
            count_layers: 1,
        })?;

        let render_pass = RenderPass::new(RenderPassConfig {
            device: device.clone(),
            attachment_infos: vec![
                AttachmentInfo {
                    image_format: swapchain_image_views[0]
                        .config()
                        .image
                        .config()
                        .image_format,
                    image_layout_final: ImageLayout::Present,
                },
                AttachmentInfo {
                    image_format: ImageFormat::D32Sfloat,
                    image_layout_final: ImageLayout::DepthAttachment,
                },
            ],
            subpass_infos: vec![SubpassInfo {
                color_attachment_index: Some(0),
                depth_attachment_index: Some(1),
            }],
        })?;

        let framebuffers: Vec<Framebuffer> = swapchain_image_views
            .iter()
            .map(|image_view| {
                Framebuffer::new(FramebufferConfig {
                    render_pass: render_pass.clone(),
                    image_views: vec![image_view.clone(), depth_image_view.clone()],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let renderer = Renderer::new(RendererConfig {
            window: Some(window.clone()),
            device: device.clone(),
            swapchain: swapchain.clone(),
            framebuffers: framebuffers.clone(),
        })?;

        let imgui_context = ImGuiContext::new(ImGuiContextConfig {
            instance: instance.clone(),
            physical_device: physical_device.clone(),
            device: device.clone(),
            swapchain: swapchain.clone(),
            render_pass: render_pass.clone(),
        })?;

        let uniform_buffer_camera = Buffer::new(BufferConfig {
            device: device.clone(),
            buffer_usages: vec![BufferUsage::Uniform],
        })?;

        let mut vertex_buffer_space = Buffer::new(BufferConfig {
            device: device.clone(),
            buffer_usages: vec![BufferUsage::Vertex],
        })?;
        vertex_buffer_space.load_data(Buffer::slice_as_bytes(&space_box_vertexes))?;

        let texture_space = create_texture(
            device.clone(),
            true,
            vec![
                "textures/nx.png".to_string(),
                "textures/px.png".to_string(),
                "textures/py.png".to_string(),
                "textures/ny.png".to_string(),
                "textures/nz.png".to_string(),
                "textures/pz.png".to_string(),
            ],
        )?;

        let texture_view_space = ImageView::new(ImageViewConfig {
            image: texture_space.clone(),
            image_view_type: ImageViewType::Cube,
            image_aspect: ImageAspect::Color,
            index_layer: 0,
            count_layers: 6,
        })?;

        let sampler_space = Sampler::new(SamplerConfig {
            physical_device: physical_device.clone(),
            device: device.clone(),
            sampler_anisotropy: 16,
        })?;

        let descriptor_set_layout_space = DescriptorSetLayout::new(DescriptorSetLayoutConfig {
            device: device.clone(),
            bindings: vec![
                BindingSet {
                    binding: 0,
                    descriptor_type: DescriptorType::Uniform,
                    shader_stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
                },
                BindingSet {
                    binding: 1,
                    descriptor_type: DescriptorType::Sampler,
                    shader_stages: vec![ShaderStage::Fragment],
                },
            ],
        })?;

        let descriptor_set_space = DescriptorSet::new(DescriptorSetConfig {
            device: device.clone(),
            descriptor_set_layout: descriptor_set_layout_space.clone(),
            descriptor_infos: Vec::new(),
        })?;

        let pipeline_layout_space = PipelineLayout::new(PipelineLayoutConfig {
            device: device.clone(),
            descriptor_set_layouts: vec![descriptor_set_layout_space.clone()],
        })?;

        let pipeline_space = Pipeline::new(PipelineConfig {
            device: device.clone(),
            render_pass: render_pass.clone(),
            pipeline_layout: pipeline_layout_space.clone(),
            vertex_shader_module: Some(
                device
                    .load_shader(vs_space::load)
                    .map_err(|_| "Failed to load vs space")?,
            ),
            fragment_shader_module: Some(
                device
                    .load_shader(fs_space::load)
                    .map_err(|_| "Failed to load fs space")?,
            ),
            width: swapchain.width(),
            height: swapchain.height(),
            depth_test: true,
            depth_write: true,
            blending: false,
            cullface: Cullface::None,
            subpass_index: 0,
            bindings: vec![BindingVertex {
                binding: 0,
                size: std::mem::size_of::<Vertex>() as u64,
                attributes: vec![
                    AttributeBindingVertex {
                        attribute: 0,
                        offset: offset_of!(Vertex, position) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 1,
                        offset: offset_of!(Vertex, texture_position) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 2,
                        offset: offset_of!(Vertex, normal) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 3,
                        offset: offset_of!(Vertex, color) as u32,
                        format: VertexFormat::Float32x3,
                    },
                ],
            }],
        })?;

        let sphere = load_model("objects/sphere.obj".to_string())?;
        let cube = load_model("objects/cube.obj".to_string())?;

        let mut vertex_buffers_bodies = vec![
            Buffer::new(BufferConfig {
                device: device.clone(),
                buffer_usages: vec![BufferUsage::Vertex],
            })?,
            Buffer::new(BufferConfig {
                device: device.clone(),
                buffer_usages: vec![BufferUsage::Vertex],
            })?,
        ];

        vertex_buffers_bodies[0].load_data(Buffer::slice_as_bytes(&sphere.meshes_vertices[0]))?;
        vertex_buffers_bodies[1].load_data(Buffer::slice_as_bytes(&cube.meshes_vertices[0]))?;

        let mut index_buffers_bodies = vec![
            Buffer::new(BufferConfig {
                device: device.clone(),
                buffer_usages: vec![BufferUsage::Index],
            })?,
            Buffer::new(BufferConfig {
                device: device.clone(),
                buffer_usages: vec![BufferUsage::Index],
            })?,
        ];

        index_buffers_bodies[0].load_data(Buffer::slice_as_bytes(&sphere.meshes_indices[0]))?;
        index_buffers_bodies[1].load_data(Buffer::slice_as_bytes(&cube.meshes_indices[0]))?;

        let indices_counts_bodies = vec![sphere.indices_counts[0], cube.indices_counts[0]];

        let sampler_bodies = Sampler::new(SamplerConfig {
            physical_device: physical_device.clone(),
            device: device.clone(),
            sampler_anisotropy: 16,
        })?;

        let descriptor_set_layout_bodies = DescriptorSetLayout::new(DescriptorSetLayoutConfig {
            device: device.clone(),
            bindings: vec![
                BindingSet {
                    binding: 0,
                    descriptor_type: DescriptorType::Uniform,
                    shader_stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
                },
                BindingSet {
                    binding: 1,
                    descriptor_type: DescriptorType::Uniform,
                    shader_stages: vec![ShaderStage::Vertex],
                },
                BindingSet {
                    binding: 2,
                    descriptor_type: DescriptorType::Sampler,
                    shader_stages: vec![ShaderStage::Fragment],
                },
            ],
        })?;

        let pipeline_layout_bodies = PipelineLayout::new(PipelineLayoutConfig {
            device: device.clone(),
            descriptor_set_layouts: vec![descriptor_set_layout_bodies.clone()],
        })?;

        let pipeline_bodies = Pipeline::new(PipelineConfig {
            device: device.clone(),
            render_pass: render_pass.clone(),
            pipeline_layout: pipeline_layout_bodies.clone(),
            vertex_shader_module: Some(
                device
                    .load_shader(vs_bodies::load)
                    .map_err(|_| "Failed to load vs bodies")?,
            ),
            fragment_shader_module: Some(
                device
                    .load_shader(fs_bodies::load)
                    .map_err(|_| "Failed to load fs bodies")?,
            ),
            width: swapchain.width(),
            height: swapchain.height(),
            depth_test: true,
            depth_write: true,
            blending: false,
            cullface: Cullface::None,
            subpass_index: 0,
            bindings: vec![BindingVertex {
                binding: 0,
                size: std::mem::size_of::<Vertex>() as u64,
                attributes: vec![
                    AttributeBindingVertex {
                        attribute: 0,
                        offset: offset_of!(Vertex, position) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 1,
                        offset: offset_of!(Vertex, texture_position) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 2,
                        offset: offset_of!(Vertex, normal) as u32,
                        format: VertexFormat::Float32x3,
                    },
                    AttributeBindingVertex {
                        attribute: 3,
                        offset: offset_of!(Vertex, color) as u32,
                        format: VertexFormat::Float32x3,
                    },
                ],
            }],
        })?;

        Ok(Self {
            sdl_context,
            window,
            device,
            swapchain,
            render_pass,
            framebuffers,
            renderer,
            imgui_context,
            uniform_buffer_camera,

            vertex_buffers_bodies,
            index_buffers_bodies,
            indices_counts_bodies,
            sampler_bodies,
            descriptor_set_layout_bodies,
            pipeline_layout_bodies,
            pipeline_bodies,

            space_renderer: SpaceRenderer {
                vertex_buffer_space,
                texture_space,
                texture_view_space,
                sampler_space,
                descriptor_set_layout_space,
                descriptor_set_space,
                pipeline_layout_space,
                pipeline_space,
                space_box_vertexes,
            },

            bodies_renderer: Vec::new(),
        })
    }

    pub fn update_event(&mut self, event: &sdl3::event::Event) {
        if let Event::Window {
            win_event: sdl3::event::WindowEvent::Resized(..),
            ..
        } = event
        {
            self.renderer.recreate_swapchain().unwrap();
        }
    }

    pub fn update_event_imgui(&mut self, event: Option<&sdl3::event::Event>) {
        if event.is_none() {
            self.imgui_context.reset_event();
            return;
        }
        self.imgui_context.update_event(event.unwrap());
    }

    pub fn update(&mut self, camera: &Camera, gui: &mut Gui) -> Result<(), String> {
        self.uniform_buffer_camera
            .load_data(Buffer::as_bytes(&camera.camera_data))?;

        self.space_renderer.descriptor_set_space.update(vec![
            DescriptorUpdateInfo::buffer(0, &self.uniform_buffer_camera)?,
            DescriptorUpdateInfo::sampler(
                1,
                &self.space_renderer.texture_view_space,
                &self.space_renderer.sampler_space,
            )?,
        ])?;

        if self.bodies_renderer.len() < gui.area.count_bodies() {
            let body = gui.area.body(gui.area.count_bodies() - 1).unwrap();

            let uniform_buffer_body = Buffer::new(BufferConfig {
                device: self.device.clone(),
                buffer_usages: vec![BufferUsage::Uniform],
            })?;

            let texture_body = match create_texture(
                self.device.clone(),
                false,
                vec![body.render_body.texture_path.clone()],
            ) {
                Ok(texture) => texture,
                Err(_) => create_texture(
                    self.device.clone(),
                    false,
                    vec!["textures/earth.jpg".to_string()],
                )?,
            };

            let texture_view_body = ImageView::new(ImageViewConfig {
                image: texture_body.clone(),
                image_view_type: ImageViewType::TwoD,
                image_aspect: ImageAspect::Color,
                index_layer: 0,
                count_layers: 1,
            })?;

            let descriptor_set_body = DescriptorSet::new(DescriptorSetConfig {
                device: self.device.clone(),
                descriptor_set_layout: self.descriptor_set_layout_bodies.clone(),
                descriptor_infos: Vec::new(),
            })?;

            self.bodies_renderer.push(BodyRenderer {
                uniform_buffer_body,
                texture_body,
                texture_view_body,
                descriptor_set_body,
            });
        } else if self.bodies_renderer.len() > gui.area.count_bodies() {
            self.bodies_renderer.pop();
        }

        for index_body in 0..gui.area.count_bodies() {
            let body = gui.area.body_mut(index_body).unwrap();

            self.bodies_renderer[index_body]
                .uniform_buffer_body
                .load_data(Buffer::as_bytes(&body.render_body.render_body_data))?;

            if body.render_body.texture_is_update {
                self.bodies_renderer[index_body].texture_body = match create_texture(
                    self.device.clone(),
                    false,
                    vec![body.render_body.texture_path.clone()],
                ) {
                    Ok(texture) => texture,
                    Err(_) => create_texture(
                        self.device.clone(),
                        false,
                        vec!["textures/earth.jpg".to_string()],
                    )?,
                };

                self.bodies_renderer[index_body].texture_view_body =
                    ImageView::new(ImageViewConfig {
                        image: self.bodies_renderer[index_body].texture_body.clone(),
                        image_view_type: ImageViewType::TwoD,
                        image_aspect: ImageAspect::Color,
                        index_layer: 0,
                        count_layers: 1,
                    })?;

                body.render_body.texture_is_update = false;
            }

            let descriptor_update_info_body = vec![
                DescriptorUpdateInfo::buffer(0, &self.uniform_buffer_camera)?,
                DescriptorUpdateInfo::buffer(
                    1,
                    &self.bodies_renderer[index_body].uniform_buffer_body,
                )?,
                DescriptorUpdateInfo::sampler(
                    2,
                    &self.bodies_renderer[index_body].texture_view_body,
                    &self.sampler_bodies,
                )?,
            ];

            self.bodies_renderer[index_body]
                .descriptor_set_body
                .update(descriptor_update_info_body)?;
        }

        Ok(())
    }

    pub fn render(
        &mut self,
        gui: &mut Gui,
        camera: &Camera,
        delta_time: f32,
    ) -> Result<(), String> {
        match self.renderer.next_image()? {
            RendererResult::SwapchainOutOfDate => {
                return Ok(());
            }
            RendererResult::Ok => {}
        }

        self.renderer.begin_command_buffer()?;
        self.renderer.begin_render_pass(&self.render_pass)?;
        self.renderer
            .bind_pipeline(&self.space_renderer.pipeline_space)?;
        self.renderer
            .bind_vertex_buffers(vec![(0, &self.space_renderer.vertex_buffer_space)])?;
        self.renderer.bind_descriptor_sets(
            &self.space_renderer.pipeline_layout_space,
            vec![(0, &self.space_renderer.descriptor_set_space)],
        )?;
        self.renderer
            .draw(self.space_renderer.space_box_vertexes.len() as u32)?;
        self.renderer.bind_pipeline(&self.pipeline_bodies)?;
        self.renderer
            .bind_vertex_buffers(vec![(0, &self.vertex_buffers_bodies[0])])?;
        self.renderer
            .bind_index_buffer(&self.index_buffers_bodies[0], IndexType::Uint32)?;

        for index_body in 0..gui.area.count_bodies() {
            let body = gui.area.body(index_body).unwrap();

            if body.render_body.body_type == BodyType::Sphere {
                self.renderer.bind_descriptor_sets(
                    &self.pipeline_layout_bodies,
                    vec![(0, &self.bodies_renderer[index_body].descriptor_set_body)],
                )?;

                self.renderer.draw_indexed(self.indices_counts_bodies[0])?;
            }
        }

        self.renderer
            .bind_vertex_buffers(vec![(0, &self.vertex_buffers_bodies[1])])?;
        self.renderer
            .bind_index_buffer(&self.index_buffers_bodies[1], IndexType::Uint32)?;

        for index_body in 0..gui.area.count_bodies() {
            let body = gui.area.body(index_body).unwrap();

            if body.render_body.body_type == BodyType::Rectangle {
                self.renderer.bind_descriptor_sets(
                    &self.pipeline_layout_bodies,
                    vec![(0, &self.bodies_renderer[index_body].descriptor_set_body)],
                )?;

                self.renderer.draw_indexed(self.indices_counts_bodies[1])?;
            }
        }

        gui.update(self.imgui_context.new_frame(delta_time)?, camera)?;

        self.renderer.render_imgui(&mut self.imgui_context)?;
        self.renderer.end_render_pass()?;
        self.renderer.end_command_buffer()?;
        self.renderer.submit()?;

        match self.renderer.present()? {
            RendererResult::SwapchainOutOfDate => {
                return Ok(());
            }
            RendererResult::Ok => {}
        }

        Ok(())
    }
}
