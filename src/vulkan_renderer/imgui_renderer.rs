use crate::vulkan_renderer::*;
use imgui::{DrawCmd, DrawCmdParams, DrawVert, TextureId, Textures};

use std::mem::*;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
        #version 450 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;
        layout(location = 2) in vec4 aColor;

        layout(set=0, binding=0) uniform Transform {
            vec2 uScale;
            vec2 uTranslate;
        } trans;

        out gl_PerVertex {
            vec4 gl_Position;
        };

        layout(location = 0) out struct {
            vec4 Color;
            vec2 UV;
        } Out;

        void main()
        {
            Out.Color = aColor;
            Out.UV = aUV;
            gl_Position = vec4(aPos * trans.uScale + trans.uTranslate, 0, 1);
        }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 450 core
        layout(location = 0) out vec4 fColor;

        layout(set=0, binding=1) uniform sampler2D sTexture;

        layout(location = 0) in struct {
            vec4 Color;
            vec2 UV;
        } In;

        void main()
        {
            fColor = In.Color * texture(sTexture, In.UV.st);
        }
        ",
    }
}

pub type ImGuiTexture = (ImageView, Sampler);

#[derive(Debug)]
pub enum ImGuiRendererError {
    BadTexture(TextureId),
}

impl std::fmt::Display for ImGuiRendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImGuiRendererError::BadTexture(id) => write!(f, "Bad texture ID: {:?}", id),
        }
    }
}

impl std::error::Error for ImGuiRendererError {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Transform {
    scale: [f32; 2],
    translate: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ImGuiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    col: [f32; 4],
}

impl From<DrawVert> for ImGuiVertex {
    fn from(v: DrawVert) -> Self {
        ImGuiVertex {
            pos: [v.pos[0], v.pos[1]],
            uv: [v.uv[0], v.uv[1]],
            col: [
                v.col[0] as f32 / 255.0,
                v.col[1] as f32 / 255.0,
                v.col[2] as f32 / 255.0,
                v.col[3] as f32 / 255.0,
            ],
        }
    }
}

pub struct ImGuiRenderer {
    physical_device: PhysicalDevice,
    device: Device,
    render_pass: RenderPass,
    pipeline: Pipeline,
    font_texture: ImGuiTexture,
    textures: Textures<ImGuiTexture>,
    matrix_buffer: Buffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    descriptor_set_layout: DescriptorSetLayout,
    descriptor_set: DescriptorSet,
}

impl ImGuiRenderer {
    pub fn new(
        imgui_context: &mut imgui::Context,
        physical_device: PhysicalDevice,
        device: Device,
        render_pass: RenderPass,
        subpass_index: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let matrix_buffer = Buffer::new(BufferConfig {
            device: device.clone(),
            buffer_usages: vec![BufferUsage::Uniform],
        })?;

        let vertex_buffer = Buffer::new(BufferConfig {
            device: device.clone(),
            buffer_usages: vec![BufferUsage::Vertex],
        })?;

        let index_buffer = Buffer::new(BufferConfig {
            device: device.clone(),
            buffer_usages: vec![BufferUsage::Index],
        })?;

        let vs = vs::load(device.device.clone()).unwrap();
        let fs = fs::load(device.device.clone()).unwrap();

        let descriptor_set_layout = DescriptorSetLayout::new(DescriptorSetLayoutConfig {
            device: device.clone(),
            bindings: vec![
                BindingSet {
                    binding: 0,
                    descriptor_type: DescriptorType::Uniform,
                    shader_stages: vec![ShaderStage::Vertex],
                },
                BindingSet {
                    binding: 1,
                    descriptor_type: DescriptorType::Sampler,
                    shader_stages: vec![ShaderStage::Fragment],
                },
            ],
        })?;

        let descriptor_set = DescriptorSet::new(DescriptorSetConfig {
            device: device.clone(),
            descriptor_set_layout: descriptor_set_layout.clone(),
            descriptor_infos: Vec::new(),
        })?;

        let pipeline_layout = PipelineLayout::new(PipelineLayoutConfig {
            device: device.clone(),
            descriptor_set_layouts: vec![descriptor_set_layout.clone()],
        })?;

        let pipeline = Pipeline::new(PipelineConfig {
            device: device.clone(),
            render_pass: render_pass.clone(),
            pipeline_layout,
            vertex_shader_module: Some(vs),
            fragment_shader_module: Some(fs),
            width: 1,
            height: 1,
            depth_test: false,
            depth_write: false,
            blending: true,
            cullface: Cullface::None,
            subpass_index,
            bindings: vec![BindingVertex {
                binding: 0,
                size: std::mem::size_of::<ImGuiVertex>() as u64,
                attributes: vec![
                    AttributeBindingVertex {
                        attribute: 0,
                        offset: offset_of!(ImGuiVertex, pos) as u32,
                        format: VertexFormat::Float32x2,
                    },
                    AttributeBindingVertex {
                        attribute: 1,
                        offset: offset_of!(ImGuiVertex, uv) as u32,
                        format: VertexFormat::Float32x2,
                    },
                    AttributeBindingVertex {
                        attribute: 2,
                        offset: offset_of!(ImGuiVertex, col) as u32,
                        format: VertexFormat::Float32x4,
                    },
                ],
            }],
        })?;

        let font_texture =
            Self::upload_font_texture(imgui_context.fonts(), &physical_device, &device)?;
        imgui_context.fonts().tex_id = TextureId::from(usize::MAX);

        Ok(Self {
            physical_device,
            device,
            render_pass,
            pipeline,
            font_texture,
            textures: Textures::new(),
            matrix_buffer,
            vertex_buffer,
            index_buffer,
            descriptor_set_layout,
            descriptor_set,
        })
    }

    fn upload_font_texture(
        fonts: &mut imgui::FontAtlas,
        physical_device: &PhysicalDevice,
        device: &Device,
    ) -> Result<ImGuiTexture, Box<dyn std::error::Error>> {
        let texture = fonts.build_rgba32_texture();

        let image = vulkan_logic::create_image_from_data(
            device.clone(),
            ImageFormat::Rgba8Srgb,
            texture.width,
            texture.height,
            texture.data,
        )?;

        let view = ImageView::new(ImageViewConfig {
            image: image.clone(),
            image_view_type: ImageViewType::TwoD,
            image_aspect: ImageAspect::Color,
            index_layer: 0,
            count_layers: 1,
        })?;

        let sampler = Sampler::new(SamplerConfig {
            physical_device: physical_device.clone(),
            device: device.clone(),
            sampler_anisotropy: 1,
        })?;

        Ok((view, sampler))
    }

    pub fn lookup_texture(&self, id: TextureId) -> Result<&ImGuiTexture, ImGuiRendererError> {
        if id.id() == usize::MAX {
            Ok(&self.font_texture)
        } else if let Some(tex) = self.textures.get(id) {
            Ok(tex)
        } else {
            Err(ImGuiRendererError::BadTexture(id))
        }
    }

    pub fn textures(&mut self) -> &mut Textures<ImGuiTexture> {
        &mut self.textures
    }

    pub fn draw(
        &mut self,
        renderer: &mut Renderer,
        draw_data: &imgui::DrawData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if draw_data.draw_lists_count() == 0 {
            return Ok(());
        }

        renderer.bind_pipeline(&self.pipeline)?;

        let mut all_vertices: Vec<ImGuiVertex> = Vec::new();
        let mut all_indices: Vec<imgui::DrawIdx> = Vec::new();
        let mut list_offsets: Vec<(usize, usize)> = Vec::new();

        for draw_list in draw_data.draw_lists() {
            list_offsets.push((all_vertices.len(), all_indices.len()));
            all_vertices.extend(
                draw_list
                    .vtx_buffer()
                    .iter()
                    .map(|&vtx| ImGuiVertex::from(vtx)),
            );
            all_indices.extend(draw_list.idx_buffer().iter());
        }

        let trans = Transform {
            scale: [
                2.0 / draw_data.display_size[0],
                2.0 / draw_data.display_size[1],
            ],
            translate: [
                -1.0 - draw_data.display_pos[0] * (2.0 / draw_data.display_size[0]),
                -1.0 - draw_data.display_pos[1] * (2.0 / draw_data.display_size[1]),
            ],
        };

        self.matrix_buffer.load_data(Buffer::as_bytes(&trans))?;
        self.vertex_buffer
            .load_data(Buffer::slice_as_bytes(&all_vertices))?;
        self.index_buffer
            .load_data(Buffer::slice_as_bytes(&all_indices))?;

        renderer.bind_vertex_buffers(vec![(0, &self.vertex_buffer)])?;
        renderer.bind_index_buffer(&self.index_buffer, IndexType::Uint16)?;

        for (i, draw_list) in draw_data.draw_lists().enumerate() {
            let (vtx_base, idx_base) = list_offsets[i];

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                texture_id,
                                idx_offset,
                                ..
                            },
                    } => {
                        let clip_off = draw_data.display_pos;
                        let clip_scale = draw_data.framebuffer_scale;

                        let x = (clip_rect[0] - clip_off[0]) * clip_scale[0];
                        let y = (clip_rect[1] - clip_off[1]) * clip_scale[1];
                        let w = (clip_rect[2] - clip_rect[0]) * clip_scale[0];
                        let h = (clip_rect[3] - clip_rect[1]) * clip_scale[1];

                        if w <= 0.0 || h <= 0.0 {
                            continue;
                        }

                        let scissor = vulkano::pipeline::graphics::viewport::Scissor {
                            offset: [x.max(0.0).floor() as u32, y.max(0.0).floor() as u32],
                            extent: [w.ceil() as u32, h.ceil() as u32],
                        };

                        renderer
                            .current_builder
                            .as_mut()
                            .unwrap()
                            .set_scissor(0, [scissor].into_iter().collect())?;

                        let tex = self.lookup_texture(texture_id)?;

                        self.descriptor_set.update(vec![
                            DescriptorUpdateInfo::buffer(0, &self.matrix_buffer)?,
                            DescriptorUpdateInfo::sampler(1, &tex.0, &tex.1)?,
                        ])?;

                        renderer.bind_descriptor_sets(
                            &self.pipeline.config().pipeline_layout,
                            vec![(0, &self.descriptor_set)],
                        )?;

                        let first_index = (idx_base + idx_offset) as u32;

                        unsafe {
                            renderer.current_builder.as_mut().unwrap().draw_indexed(
                                count as u32,
                                1,
                                first_index,
                                vtx_base as i32,
                                0,
                            )?;
                        }
                    }
                    DrawCmd::ResetRenderState => (),
                    DrawCmd::RawCallback { .. } => (),
                }
            }
        }

        Ok(())
    }
}
