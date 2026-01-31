use super::*;

use crate::body::Vertex;
use russimp::scene::{PostProcess, Scene};

pub type VulkanLogicResult<T> = Result<T, String>;

#[derive(Clone)]
pub struct ModelData {
    pub meshes_vertices: Vec<Vec<Vertex>>,
    pub meshes_indices: Vec<Vec<u32>>,
    pub indices_counts: Vec<u32>,
}

pub fn create_texture(
    device: Device,
    is_cube_texture: bool,
    paths: Vec<String>,
) -> VulkanLogicResult<Image> {
    let error_object = String::from("CreateTexture");

    if paths.is_empty() {
        return Err(format!("{}: Undefined paths", error_object));
    }
    if is_cube_texture && !paths.len().is_multiple_of(6) {
        return Err(format!(
            "{}: Not enough textures for cube texture",
            error_object
        ));
    }

    let mut pixels = Vec::with_capacity(paths.len());
    let (mut width, mut height, mut channels) = (0usize, 0usize, 0usize);

    for (i, path) in paths.iter().enumerate() {
        let image = match stb_image::image::load_with_depth(path, 4, false) {
            stb_image::image::LoadResult::ImageU8(image) => image,
            _ => return Err(format!("Failed to load texture at index {}", i)),
        };

        let (w, h, c) = (image.width, image.height, image.depth);

        if i == 0 {
            (width, height, channels) = (w, h, c);
        } else if w != width || h != height || c != channels {
            return Err(format!("Texture parameters do not match at index {}", i));
        }

        pixels.push(image.data);
    }

    let mut data = Vec::with_capacity(width * height * 4 * pixels.len());

    for pixel_data in &pixels {
        data.extend_from_slice(pixel_data);
    }

    let mut staging_buffer = Buffer::new(BufferConfig {
        device: device.clone(),
        buffer_usages: vec![BufferUsage::Src],
    })?;

    staging_buffer.load_data(&data)?;

    let texture_image_flag = match (is_cube_texture, pixels.len()) {
        (true, 6) => ImageFlag::Cube,
        (true, _) => ImageFlag::CubeArray,
        (false, 2..) => ImageFlag::TwoDArray,
        _ => ImageFlag::TwoD,
    };

    let texture_image = Image::new(ImageConfig {
        device: device.clone(),
        image_flag: texture_image_flag,
        image_format: ImageFormat::Rgba8Srgb,
        image_usages: vec![ImageUsage::Dst, ImageUsage::Sampler],
        count_layers: pixels.len() as u32,
        width: width as u32,
        height: height as u32,
    })?;

    let copy_regions: Vec<command_buffer::BufferImageCopy> = (0..paths.len())
        .map(|i| command_buffer::BufferImageCopy {
            buffer_offset: (width * height * 4 * i) as u64,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: image::ImageSubresourceLayers {
                aspects: image::ImageAspects::COLOR,
                mip_level: 0,
                array_layers: i as u32..(i as u32 + 1),
            },
            image_offset: [0, 0, 0],
            image_extent: [width as u32, height as u32, 1],
            ..Default::default()
        })
        .collect();

    let mut builder = command_buffer::AutoCommandBufferBuilder::primary(
        device.command_buffer_allocator.clone(),
        device.queue_graphics.queue_family_index(),
        command_buffer::CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(|_| format!("{}: Failed to create command buffer", error_object))?;

    builder
        .copy_buffer_to_image(command_buffer::CopyBufferToImageInfo {
            regions: copy_regions.into_iter().collect(),
            ..command_buffer::CopyBufferToImageInfo::buffer_image(
                staging_buffer
                    .subbuffer
                    .clone()
                    .ok_or_else(|| format!("{}: Undefined StagingBuffer", error_object))?,
                texture_image.image.clone(),
            )
        })
        .map_err(|err| format!("{}: Failed to copy buffer to image {}", error_object, err))?;

    let cmd = builder
        .build()
        .map_err(|_| format!("{}: Failed to build command buffer", error_object))?;

    let future = sync::now(device.device);

    let exec_future = future
        .then_execute(device.queue_graphics.clone(), cmd)
        .map_err(|_| format!("{}: Failed to execute command buffer", error_object))?
        .then_signal_fence_and_flush()
        .map_err(|_| format!("{}: Failed to flush fence", error_object))?;

    exec_future.boxed().cleanup_finished();

    Ok(texture_image)
}

pub fn create_image_from_data(
    device: Device,
    image_format: ImageFormat,
    width: u32,
    height: u32,
    data: &[u8],
) -> VulkanLogicResult<Image> {
    let error_object = String::from("CreateTexture");

    let mut staging_buffer = Buffer::new(BufferConfig {
        device: device.clone(),
        buffer_usages: vec![BufferUsage::Src],
    })?;

    staging_buffer.load_data(data)?;

    let texture_image = Image::new(ImageConfig {
        device: device.clone(),
        image_flag: ImageFlag::TwoD,
        image_format,
        image_usages: vec![ImageUsage::Dst, ImageUsage::Sampler],
        count_layers: 1,
        width,
        height,
    })?;

    let mut builder = command_buffer::AutoCommandBufferBuilder::primary(
        device.command_buffer_allocator.clone(),
        device.queue_graphics.queue_family_index(),
        command_buffer::CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(|_| format!("{}: Failed to create command buffer", error_object))?;

    builder
        .copy_buffer_to_image(command_buffer::CopyBufferToImageInfo::buffer_image(
            staging_buffer
                .subbuffer
                .clone()
                .ok_or_else(|| format!("{}: Undefined StagingBuffer", error_object))?,
            texture_image.image.clone(),
        ))
        .map_err(|err| format!("{}: Failed to copy buffer to image {}", error_object, err))?;

    let cmd = builder
        .build()
        .map_err(|_| format!("{}: Failed to build command buffer", error_object))?;

    let future = sync::now(device.device);

    let exec_future = future
        .then_execute(device.queue_graphics.clone(), cmd)
        .map_err(|_| format!("{}: Failed to execute command buffer", error_object))?
        .then_signal_fence_and_flush()
        .map_err(|_| format!("{}: Failed to flush fence", error_object))?;

    exec_future.boxed().cleanup_finished();

    Ok(texture_image)
}

pub fn load_model(model_path: &str) -> Result<ModelData, String> {
    let scene = Scene::from_file(
        model_path,
        vec![PostProcess::Triangulate, PostProcess::CalculateTangentSpace],
    )
    .map_err(|_| format!("Failed to load model: {}", model_path))?;

    let mut meshes_vertices = Vec::new();
    let mut meshes_indices = Vec::new();
    let mut indices_counts = Vec::new();

    for mesh in scene.meshes.iter() {
        let mut vertexes = Vec::new();
        let mut indexes = Vec::new();

        let has_texture_coords = !mesh.texture_coords.is_empty();
        let has_normals = !mesh.normals.is_empty();
        let has_colors = !mesh.colors.is_empty();

        if !has_normals {
            return Err("Failed to find normals in mesh".to_string());
        }

        for face in &mesh.faces {
            for &index in &face.0 {
                let vertex_index = index as usize;

                let position = mesh.vertices[vertex_index];
                let texture_position = if has_texture_coords && mesh.texture_coords[0].is_some() {
                    let tc = mesh.texture_coords[0].as_ref().unwrap()[vertex_index];
                    glm::vec3(tc.x, tc.y, 1.0)
                } else {
                    glm::vec3(-1.0, -1.0, -1.0)
                };
                let normal = mesh.normals[vertex_index];
                let color = if has_colors && mesh.colors[0].is_some() {
                    let c = mesh.colors[0].as_ref().unwrap()[vertex_index];
                    glm::vec3(c.r, c.g, c.b)
                } else {
                    glm::vec3(-1.0, -1.0, -1.0)
                };

                vertexes.push(Vertex {
                    position: glm::vec3(position.x, position.y, position.z),
                    texture_position,
                    normal: glm::vec3(normal.x, normal.y, normal.z),
                    color,
                });

                indexes.push(vertexes.len() as u32 - 1);
            }
        }

        meshes_vertices.push(vertexes);
        meshes_indices.push(indexes.clone());
        indices_counts.push(indexes.len() as u32);
    }

    Ok(ModelData {
        meshes_vertices,
        meshes_indices,
        indices_counts,
    })
}
