extern crate sdl3;

mod imgui_renderer;
pub mod vulkan_logic;

use std::any::*;
use std::cell::*;
use std::collections::HashMap;
use std::rc::*;
use std::sync::*;
use vulkano::sync::GpuFuture;
use vulkano::*;

type ShaderLoadFn =
    fn(Arc<device::Device>) -> Result<Arc<shader::ShaderModule>, Validated<VulkanError>>;

pub type VulkanRendererResult<T> = Result<T, String>;

pub trait VulkanRendererObject {
    type Config;

    fn new(config: Self::Config) -> VulkanRendererResult<Self>
    where
        Self: Sized;

    fn config(&self) -> Self::Config;
}

#[derive(Clone)]
pub struct InstanceConfig {
    pub app_name: String,
    pub engine_name: String,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        Self {
            app_name: "app".to_string(),
            engine_name: "engine".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Instance {
    instance: Arc<instance::Instance>,
    config: InstanceConfig,
    error_object: String,
}

impl VulkanRendererObject for Instance {
    type Config = InstanceConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let library =
            VulkanLibrary::new().map_err(|_| format!("{0}: Failed to load {0}", error_object))?;

        let extensions = Self::required_extensions(&library).map_err(|err| {
            format!(
                "{0}: Failed to get required extensions: {1}",
                error_object, err
            )
        })?;

        let instance = instance::Instance::new(
            library.clone(),
            instance::InstanceCreateInfo {
                application_name: Some(config.app_name.clone()),
                engine_name: Some(config.engine_name.clone()),
                enabled_extensions: extensions,
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            instance,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Instance {
    fn required_extensions(
        library: &Arc<VulkanLibrary>,
    ) -> VulkanRendererResult<instance::InstanceExtensions> {
        let supported = library.supported_extensions();

        let mut extensions = instance::InstanceExtensions::empty();

        if !supported.khr_surface {
            return Err("KHR_surface extension not supported".to_string());
        }
        extensions.khr_surface = true;

        #[cfg(target_os = "windows")]
        {
            if !supported.khr_win32_surface {
                return Err("KHR_win32_surface extension not supported".to_string());
            }
            extensions.khr_win32_surface = true;
        }

        #[cfg(target_os = "android")]
        {
            if !supported.khr_android_surface {
                return Err("KHR_android_surface extension not supported".to_string());
            }
            extensions.khr_android_surface = true;
        }

        #[cfg(target_os = "macos")]
        {
            if !supported.mvk_macos_surface {
                return Err("MVK_macos_surface extension not supported".to_string());
            }
            extensions.mvk_macos_surface = true;
        }

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        {
            if std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland") {
                if !supported.khr_wayland_surface {
                    return Err("KHR_wayland_surface extension not supported".to_string());
                }
                extensions.khr_wayland_surface = true;
            } else {
                if !supported.khr_xcb_surface {
                    return Err("KHR_xcb_surface extension not supported".to_string());
                }
                extensions.khr_xcb_surface = true;
            }
        }

        Ok(extensions)
    }
}

#[derive(Clone)]
pub struct SurfaceConfig {
    pub instance: Instance,
    pub window: Option<sdl3::video::Window>,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            instance: Instance::new(InstanceConfig::default()).unwrap(),
            window: None,
        }
    }
}

#[derive(Clone)]
pub struct Surface {
    surface: Arc<swapchain::Surface>,
    config: SurfaceConfig,
    error_object: String,
}

impl VulkanRendererObject for Surface {
    type Config = SurfaceConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let instance = &config.instance.instance;

        let window = config
            .window
            .clone()
            .ok_or_else(|| format!("{}: Window not created", error_object))?;

        let surface = Arc::new(unsafe {
            swapchain::Surface::from_handle(
                instance.clone(),
                ash::vk::SurfaceKHR::from_raw(
                    window
                        .vulkan_create_surface(
                            config.instance.instance.handle().as_raw() as sdl3::video::VkInstance
                        )
                        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?
                        as u64,
                ),
                Self::surface_api(),
                None,
            )
        });

        Ok(Self {
            surface,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Surface {
    fn surface_api() -> swapchain::SurfaceApi {
        #[cfg(target_os = "windows")]
        return swapchain::SurfaceApi::Win32;

        #[cfg(target_os = "android")]
        return swapchain::SurfaceApi::Android;

        #[cfg(target_os = "macos")]
        return swapchain::SurfaceApi::MacOs;

        #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
        {
            if std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland") {
                return swapchain::SurfaceApi::Wayland;
            }
            swapchain::SurfaceApi::Xcb
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PhysicalDevicePriority {
    Nvidia,
    Intel,
    Radeon,
}

#[derive(Clone)]
pub struct PhysicalDeviceConfig {
    pub instance: Instance,
    pub surface: Surface,
    pub priority_gpu: PhysicalDevicePriority,
}

impl Default for PhysicalDeviceConfig {
    fn default() -> Self {
        Self {
            instance: Instance::new(InstanceConfig::default()).unwrap(),
            surface: Surface::new(SurfaceConfig::default()).unwrap(),
            priority_gpu: PhysicalDevicePriority::Intel,
        }
    }
}

#[derive(Clone)]
pub struct PhysicalDevice {
    physical_device: Arc<device::physical::PhysicalDevice>,
    config: PhysicalDeviceConfig,
    queue_family_index: u32,
    error_object: String,
}

impl VulkanRendererObject for PhysicalDevice {
    type Config = PhysicalDeviceConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let instance = &config.instance.instance;

        let physical_devices: Vec<Arc<device::physical::PhysicalDevice>> = instance
            .enumerate_physical_devices()
            .map_err(|_| format!("{0}: Failed to enumerate {0}", error_object))?
            .collect();

        let vendor_id = match config.priority_gpu {
            PhysicalDevicePriority::Nvidia => 0x10DE,
            PhysicalDevicePriority::Intel => 0x8086,
            PhysicalDevicePriority::Radeon => 0x1002,
        };

        let physical_device = physical_devices
            .iter()
            .find(|physical_device| physical_device.properties().vendor_id == vendor_id)
            .or_else(|| physical_devices.first())
            .cloned()
            .ok_or_else(|| format!("{0}: No {0} found", error_object))?;

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, queue_family)| {
                queue_family.queue_count > 0
                    && queue_family
                        .queue_flags
                        .intersects(device::QueueFlags::GRAPHICS)
                    && physical_device
                        .surface_support(i as u32, &config.surface.surface)
                        .unwrap_or(false)
            })
            .ok_or_else(|| format!("{}: No suitable queue family found", error_object))?
            as u32;

        Ok(Self {
            physical_device,
            config,
            queue_family_index,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl PhysicalDevice {
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}

#[derive(Clone)]
pub struct DeviceConfig {
    pub physical_device: PhysicalDevice,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            physical_device: PhysicalDevice::new(PhysicalDeviceConfig::default()).unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct Device {
    device: Arc<device::Device>,
    config: DeviceConfig,
    memory_allocator: Arc<memory::allocator::StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<descriptor_set::allocator::StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<command_buffer::allocator::StandardCommandBufferAllocator>,
    subbuffer_allocators:
        Rc<RefCell<HashMap<SubbufferAllocatorKey, Rc<buffer::allocator::SubbufferAllocator>>>>,
    queue_graphics: Arc<device::Queue>,
    error_object: String,
}

impl VulkanRendererObject for Device {
    type Config = DeviceConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let physical_device = &config.physical_device.physical_device;

        let physical_device_features_supported = physical_device.supported_features();
        let mut physical_device_features = device::DeviceFeatures::empty();

        physical_device_features.multi_draw_indirect =
            physical_device_features_supported.multi_draw_indirect;
        physical_device_features.sampler_anisotropy =
            physical_device_features_supported.sampler_anisotropy;
        physical_device_features.image_cube_array =
            physical_device_features_supported.image_cube_array;

        if !physical_device_features.multi_draw_indirect {
            return Err(format!("{}: MultiDrawIndirect not supported", error_object));
        }
        if !physical_device_features.sampler_anisotropy {
            return Err(format!("{}: SamplerAnisotropy not supported", error_object));
        }
        if !physical_device_features.image_cube_array {
            return Err(format!("{}: ImageCubeArray not supported", error_object));
        }

        let queue_create_infos = vec![device::QueueCreateInfo {
            queue_family_index: config.physical_device.queue_family_index,
            queues: vec![1.0],
            ..Default::default()
        }];

        let swapchain_supported = physical_device.supported_extensions().khr_swapchain;
        if !swapchain_supported {
            return Err(format!("{}: Swapchain not supported", error_object));
        }

        let (device, mut queues) = device::Device::new(
            physical_device.clone(),
            device::DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: device::DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                enabled_features: physical_device_features,
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        let memory_allocator = Arc::new(memory::allocator::StandardMemoryAllocator::new_default(
            device.clone(),
        ));
        let descriptor_set_allocator = Arc::new(
            descriptor_set::allocator::StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );
        let command_buffer_allocator = Arc::new(
            command_buffer::allocator::StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );

        let queue_graphics = queues
            .next()
            .ok_or_else(|| format!("{}: No graphics queue available", error_object))?;

        Ok(Self {
            device,
            config,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            subbuffer_allocators: Rc::new(RefCell::new(HashMap::new())),
            queue_graphics,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Device {
    pub fn load_shader(
        &self,
        load_fn: ShaderLoadFn,
    ) -> Result<Arc<shader::ShaderModule>, Validated<VulkanError>> {
        load_fn(self.device.clone())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    Src,
    Dst,
    Uniform,
    Vertex,
    Index,
}

impl From<BufferUsage> for buffer::BufferUsage {
    fn from(usage: BufferUsage) -> Self {
        match usage {
            BufferUsage::Src => buffer::BufferUsage::TRANSFER_SRC,
            BufferUsage::Dst => buffer::BufferUsage::TRANSFER_DST,
            BufferUsage::Uniform => buffer::BufferUsage::UNIFORM_BUFFER,
            BufferUsage::Vertex => buffer::BufferUsage::VERTEX_BUFFER,
            BufferUsage::Index => buffer::BufferUsage::INDEX_BUFFER,
        }
    }
}

#[derive(Clone)]
pub struct BufferConfig {
    pub device: Device,
    pub buffer_usages: Vec<BufferUsage>,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            buffer_usages: Vec::new(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SubbufferAllocatorKey {
    buffer_usages: Vec<BufferUsage>,
}

#[derive(Clone)]
pub struct Buffer {
    subbuffer: Option<buffer::Subbuffer<[u8]>>,
    subbuffer_allocator: Rc<buffer::allocator::SubbufferAllocator>,
    config: BufferConfig,
    error_object: String,
}

impl VulkanRendererObject for Buffer {
    type Config = BufferConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let memory_allocator = &config.device.memory_allocator;

        if config.buffer_usages.is_empty() {
            return Err(format!("{}: Undefined BufferUsages", error_object));
        }

        let subbuffer_allocators = &config.device.subbuffer_allocators;

        let subbuffer_allocator_key = SubbufferAllocatorKey {
            buffer_usages: config.buffer_usages.clone(),
        };

        let usage = config
            .buffer_usages
            .iter()
            .fold(buffer::BufferUsage::empty(), |acc, &usage| {
                acc | usage.into()
            });

        let subbuffer_allocator = subbuffer_allocators
            .borrow_mut()
            .entry(subbuffer_allocator_key)
            .or_insert_with(|| {
                Rc::new(buffer::allocator::SubbufferAllocator::new(
                    memory_allocator.clone(),
                    buffer::allocator::SubbufferAllocatorCreateInfo {
                        buffer_usage: usage,
                        memory_type_filter: memory::allocator::MemoryTypeFilter::PREFER_HOST
                            | memory::allocator::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                ))
            })
            .clone();

        Ok(Self {
            subbuffer: None,
            subbuffer_allocator,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Buffer {
    pub fn load_data(&mut self, data: &[u8]) -> VulkanRendererResult<()> {
        let layout = memory::allocator::DeviceLayout::from_size_alignment(data.len() as u64, 1)
            .ok_or_else(|| format!("{0}: Failed to create DeviceLayout", self.error_object))?;

        self.subbuffer = Some(
            self.subbuffer_allocator
                .allocate(layout)
                .map_err(|_| format!("{0}: Failed to allocate Subbuffer", self.error_object))?,
        );

        let mut mapping = self
            .subbuffer
            .as_mut()
            .unwrap()
            .write()
            .map_err(|_| format!("{}: Failed to map memory", self.error_object))?;

        mapping[..layout.size() as usize].copy_from_slice(data);

        Ok(())
    }

    pub fn as_bytes<T>(p: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts(p as *const T as *const u8, std::mem::size_of::<T>()) }
    }

    pub fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageFlag {
    TwoD,
    TwoDArray,
    Cube,
    CubeArray,
}

impl From<ImageFlag> for image::ImageCreateFlags {
    fn from(flag: ImageFlag) -> Self {
        match flag {
            ImageFlag::TwoD => image::ImageCreateFlags::empty(),
            ImageFlag::TwoDArray => image::ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE,
            ImageFlag::Cube => image::ImageCreateFlags::CUBE_COMPATIBLE,
            ImageFlag::CubeArray => {
                image::ImageCreateFlags::CUBE_COMPATIBLE
                    | image::ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Rgb8Srgb,
    Bgr8Srgb,
    Rgb8Unorm,
    Bgr8Unorm,
    Rgba8Srgb,
    Bgra8Srgb,
    Rgba8Unorm,
    Bgra8Unorm,
    R8Srgb,
    R8Unorm,
    D32Sfloat,
}

impl From<ImageFormat> for format::Format {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Rgb8Srgb => format::Format::R8G8B8_SRGB,
            ImageFormat::Bgr8Srgb => format::Format::B8G8R8_SRGB,
            ImageFormat::Rgb8Unorm => format::Format::R8G8B8_UNORM,
            ImageFormat::Bgr8Unorm => format::Format::B8G8R8_UNORM,
            ImageFormat::Rgba8Srgb => format::Format::R8G8B8A8_SRGB,
            ImageFormat::Bgra8Srgb => format::Format::B8G8R8A8_SRGB,
            ImageFormat::Rgba8Unorm => format::Format::R8G8B8A8_UNORM,
            ImageFormat::Bgra8Unorm => format::Format::B8G8R8A8_UNORM,
            ImageFormat::R8Srgb => format::Format::R8_SRGB,
            ImageFormat::R8Unorm => format::Format::R8_UNORM,
            ImageFormat::D32Sfloat => format::Format::D32_SFLOAT,
        }
    }
}

impl TryFrom<format::Format> for ImageFormat {
    type Error = ();

    fn try_from(format: format::Format) -> Result<Self, Self::Error> {
        match format {
            format::Format::R8G8B8_SRGB => Ok(ImageFormat::Rgb8Srgb),
            format::Format::B8G8R8_SRGB => Ok(ImageFormat::Bgr8Srgb),
            format::Format::R8G8B8_UNORM => Ok(ImageFormat::Rgb8Unorm),
            format::Format::B8G8R8_UNORM => Ok(ImageFormat::Bgr8Unorm),
            format::Format::R8G8B8A8_SRGB => Ok(ImageFormat::Rgba8Srgb),
            format::Format::B8G8R8A8_SRGB => Ok(ImageFormat::Bgra8Srgb),
            format::Format::R8G8B8A8_UNORM => Ok(ImageFormat::Rgba8Unorm),
            format::Format::B8G8R8A8_UNORM => Ok(ImageFormat::Bgra8Unorm),
            format::Format::R8_SRGB => Ok(ImageFormat::R8Srgb),
            format::Format::R8_UNORM => Ok(ImageFormat::R8Unorm),
            format::Format::D32_SFLOAT => Ok(ImageFormat::D32Sfloat),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageUsage {
    Src,
    Dst,
    Sampler,
    Color,
    Depth,
}

impl From<ImageUsage> for image::ImageUsage {
    fn from(usage: ImageUsage) -> Self {
        match usage {
            ImageUsage::Src => image::ImageUsage::TRANSFER_SRC,
            ImageUsage::Dst => image::ImageUsage::TRANSFER_DST,
            ImageUsage::Sampler => image::ImageUsage::SAMPLED,
            ImageUsage::Color => image::ImageUsage::COLOR_ATTACHMENT,
            ImageUsage::Depth => image::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        }
    }
}

#[derive(Clone)]
pub struct ImageConfig {
    pub device: Device,
    pub image_flag: ImageFlag,
    pub image_format: ImageFormat,
    pub image_usages: Vec<ImageUsage>,
    pub count_layers: u32,
    pub width: u32,
    pub height: u32,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            image_flag: ImageFlag::TwoD,
            image_format: ImageFormat::Rgba8Srgb,
            image_usages: Vec::new(),
            count_layers: 0,
            width: 0,
            height: 0,
        }
    }
}

#[derive(Clone)]
pub struct Image {
    image: Arc<image::Image>,
    config: ImageConfig,
    error_object: String,
}

impl VulkanRendererObject for Image {
    type Config = ImageConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let memory_allocator = &config.device.memory_allocator;

        if config.image_usages.is_empty() {
            return Err(format!("{}: Undefined ImageUsages", error_object));
        }
        if config.count_layers == 0 {
            return Err(format!("{}: Undefined size", error_object));
        }
        if config.width == 0 {
            return Err(format!("{}: Undefined width", error_object));
        }
        if config.height == 0 {
            return Err(format!("{}: Undefined height", error_object));
        }

        let usage = config
            .image_usages
            .iter()
            .fold(image::ImageUsage::empty(), |acc, &usage| acc | usage.into());

        let image = image::Image::new(
            memory_allocator.clone(),
            image::ImageCreateInfo {
                flags: config.image_flag.into(),
                image_type: image::ImageType::Dim2d,
                format: config.image_format.into(),
                extent: [config.width, config.height, 1],
                array_layers: config.count_layers,
                usage,
                ..Default::default()
            },
            memory::allocator::AllocationCreateInfo {
                memory_type_filter: memory::allocator::MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            image,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageViewType {
    TwoD,
    TwoDArray,
    Cube,
    CubeArray,
}

impl From<ImageViewType> for image::view::ImageViewType {
    fn from(type_: ImageViewType) -> Self {
        match type_ {
            ImageViewType::TwoD => image::view::ImageViewType::Dim2d,
            ImageViewType::TwoDArray => image::view::ImageViewType::Dim2dArray,
            ImageViewType::Cube => image::view::ImageViewType::Cube,
            ImageViewType::CubeArray => image::view::ImageViewType::CubeArray,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageAspect {
    Color,
    Depth,
}

impl From<ImageAspect> for image::ImageAspects {
    fn from(aspect: ImageAspect) -> Self {
        match aspect {
            ImageAspect::Color => image::ImageAspects::COLOR,
            ImageAspect::Depth => image::ImageAspects::DEPTH,
        }
    }
}

#[derive(Clone)]
pub struct ImageViewConfig {
    pub image: Image,
    pub image_view_type: ImageViewType,
    pub image_aspect: ImageAspect,
    pub index_layer: u32,
    pub count_layers: u32,
}

impl Default for ImageViewConfig {
    fn default() -> Self {
        Self {
            image: Image::new(ImageConfig::default()).unwrap(),
            image_view_type: ImageViewType::TwoD,
            image_aspect: ImageAspect::Color,
            index_layer: 0,
            count_layers: 0,
        }
    }
}

#[derive(Clone)]
pub struct ImageView {
    image_view: Arc<image::view::ImageView>,
    config: ImageViewConfig,
    error_object: String,
}

impl VulkanRendererObject for ImageView {
    type Config = ImageViewConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let image = &config.image.image;

        if config.count_layers == 0 {
            return Err(format!("{}: Undefined count layers", error_object));
        }
        if config.index_layer + config.count_layers > image.array_layers() {
            return Err(format!("{}: Undefined index layer", error_object));
        }
        if (config.image_view_type == ImageViewType::Cube
            || config.image_view_type == ImageViewType::CubeArray)
            && config.count_layers % 6 != 0
        {
            return Err(format!(
                "{}: Not enough count layers for cube image view",
                error_object
            ));
        }

        let image_view = image::view::ImageView::new(
            image.clone(),
            image::view::ImageViewCreateInfo {
                view_type: config.image_view_type.into(),
                format: image.format(),
                subresource_range: image::ImageSubresourceRange {
                    aspects: config.image_aspect.into(),
                    mip_levels: 0..image.mip_levels(),
                    array_layers: config.index_layer..config.count_layers,
                },
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            image_view,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone)]
pub struct SamplerConfig {
    pub physical_device: PhysicalDevice,
    pub device: Device,
    pub sampler_anisotropy: u32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            physical_device: PhysicalDevice::new(PhysicalDeviceConfig::default()).unwrap(),
            device: Device::new(DeviceConfig::default()).unwrap(),
            sampler_anisotropy: 0,
        }
    }
}

#[derive(Clone)]
pub struct Sampler {
    sampler: Arc<image::sampler::Sampler>,
    config: SamplerConfig,
    error_object: String,
}

impl VulkanRendererObject for Sampler {
    type Config = SamplerConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let physical_device = &config.physical_device.physical_device;
        let device = &config.device.device;

        let mut current_sampler_anisotropy =
            physical_device.properties().max_sampler_anisotropy as u32;

        if config.sampler_anisotropy > current_sampler_anisotropy {
            println!(
                "{}: Sampler anisotropy exceeds sampler anisotropy {} limit",
                error_object, current_sampler_anisotropy
            );
        } else {
            current_sampler_anisotropy = config.sampler_anisotropy;
        }

        let sampler = image::sampler::Sampler::new(
            device.clone(),
            image::sampler::SamplerCreateInfo {
                mag_filter: image::sampler::Filter::Linear,
                min_filter: image::sampler::Filter::Linear,
                mipmap_mode: image::sampler::SamplerMipmapMode::Linear,
                address_mode: [
                    image::sampler::SamplerAddressMode::ClampToEdge,
                    image::sampler::SamplerAddressMode::ClampToEdge,
                    image::sampler::SamplerAddressMode::ClampToEdge,
                ],
                mip_lod_bias: 0.0,
                anisotropy: Some(current_sampler_anisotropy as f32),
                compare: None,
                border_color: image::sampler::BorderColor::FloatOpaqueWhite,
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            sampler,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DescriptorType {
    Uniform,
    Sampler,
}

impl From<DescriptorType> for descriptor_set::layout::DescriptorType {
    fn from(descriptor_type: DescriptorType) -> Self {
        match descriptor_type {
            DescriptorType::Uniform => descriptor_set::layout::DescriptorType::UniformBuffer,
            DescriptorType::Sampler => descriptor_set::layout::DescriptorType::CombinedImageSampler,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl From<ShaderStage> for shader::ShaderStages {
    fn from(shader_stage: ShaderStage) -> Self {
        match shader_stage {
            ShaderStage::Vertex => shader::ShaderStages::VERTEX,
            ShaderStage::Fragment => shader::ShaderStages::FRAGMENT,
        }
    }
}

#[derive(Clone)]
pub struct BindingSet {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub shader_stages: Vec<ShaderStage>,
}

impl Default for BindingSet {
    fn default() -> Self {
        Self {
            binding: 0,
            descriptor_type: DescriptorType::Uniform,
            shader_stages: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSetLayoutConfig {
    pub device: Device,
    pub bindings: Vec<BindingSet>,
}

impl Default for DescriptorSetLayoutConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            bindings: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSetLayout {
    descriptor_set_layout: Arc<descriptor_set::layout::DescriptorSetLayout>,
    config: DescriptorSetLayoutConfig,
    error_object: String,
}

impl VulkanRendererObject for DescriptorSetLayout {
    type Config = DescriptorSetLayoutConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let device = &config.device.device;

        if config.bindings.is_empty() {
            return Err(format!("{}: Undefined Bindings", error_object));
        }

        let descriptor_set_layout = descriptor_set::layout::DescriptorSetLayout::new(
            device.clone(),
            descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: config
                    .bindings
                    .iter()
                    .map(|binding| {
                        (binding.binding, descriptor_set::layout::DescriptorSetLayoutBinding {
                            stages: binding
                                .shader_stages
                                .iter()
                                .fold(shader::ShaderStages::empty(), |acc, &stage| {
                                    acc | stage.into()
                                }),
                            ..descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                                binding.descriptor_type.into(),
                            )
                        })
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            descriptor_set_layout,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone)]
pub struct DescriptorInfo {
    write_descriptor_set: descriptor_set::WriteDescriptorSet,
}

impl Default for DescriptorInfo {
    fn default() -> Self {
        Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::none(0),
        }
    }
}

impl DescriptorInfo {
    pub fn buffer(binding: u32, buffer: Buffer) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        Ok(Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::buffer(
                binding,
                buffer
                    .subbuffer
                    .clone()
                    .ok_or_else(|| format!("{}: Undefined Buffer", error_object))?,
            ),
        })
    }

    pub fn sampler(
        binding: u32,
        image_view: ImageView,
        sampler: Sampler,
    ) -> VulkanRendererResult<Self> {
        Ok(Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::image_view_sampler(
                binding,
                image_view.image_view.clone(),
                sampler.sampler.clone(),
            ),
        })
    }
}

#[derive(Clone)]
pub struct DescriptorUpdateInfo {
    write_descriptor_set: descriptor_set::WriteDescriptorSet,
}

impl Default for DescriptorUpdateInfo {
    fn default() -> Self {
        Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::none(0),
        }
    }
}

impl DescriptorUpdateInfo {
    pub fn buffer(binding: u32, buffer: &Buffer) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        Ok(Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::buffer(
                binding,
                buffer
                    .subbuffer
                    .clone()
                    .ok_or_else(|| format!("{}: Undefined Buffer", error_object))?,
            ),
        })
    }

    pub fn sampler(
        binding: u32,
        image_view: &ImageView,
        sampler: &Sampler,
    ) -> VulkanRendererResult<Self> {
        Ok(Self {
            write_descriptor_set: descriptor_set::WriteDescriptorSet::image_view_sampler(
                binding,
                image_view.image_view.clone(),
                sampler.sampler.clone(),
            ),
        })
    }
}

#[derive(Clone)]
pub struct DescriptorSetConfig {
    pub device: Device,
    pub descriptor_set_layout: DescriptorSetLayout,
    pub descriptor_infos: Vec<DescriptorInfo>,
}

impl Default for DescriptorSetConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            descriptor_set_layout: DescriptorSetLayout::new(DescriptorSetLayoutConfig::default())
                .unwrap(),
            descriptor_infos: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSet {
    descriptor_set: Arc<descriptor_set::DescriptorSet>,
    config: DescriptorSetConfig,
    error_object: String,
}

impl VulkanRendererObject for DescriptorSet {
    type Config = DescriptorSetConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let descriptor_set_allocator = &config.device.descriptor_set_allocator;
        let descriptor_set_layout = &config.descriptor_set_layout.descriptor_set_layout;

        let descriptor_set = descriptor_set::DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            config
                .descriptor_infos
                .iter()
                .map(|descriptor_info| descriptor_info.write_descriptor_set.clone()),
            [],
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            descriptor_set,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl DescriptorSet {
    pub fn update(
        &mut self,
        descriptor_update_infos: Vec<DescriptorUpdateInfo>,
    ) -> VulkanRendererResult<()> {
        let descriptor_set_allocator = &self.config.device.descriptor_set_allocator;
        let descriptor_set_layout = &self.config.descriptor_set_layout.descriptor_set_layout;

        if descriptor_update_infos.is_empty() {
            return Err(format!(
                "{}: Undefined DescriptorUpdateInfos",
                self.error_object
            ));
        }

        self.descriptor_set = descriptor_set::DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            descriptor_update_infos
                .iter()
                .map(|descriptor_update_info| descriptor_update_info.write_descriptor_set.clone()),
            [],
        )
        .map_err(|_| format!("{0}: Failed to update {0}", self.error_object))?;

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    Fifo,
    Mailbox,
}

impl From<PresentMode> for swapchain::PresentMode {
    fn from(present_mode: PresentMode) -> Self {
        match present_mode {
            PresentMode::Fifo => swapchain::PresentMode::Fifo,
            PresentMode::Mailbox => swapchain::PresentMode::Mailbox,
        }
    }
}

#[derive(Clone)]
pub struct SwapchainConfig {
    pub surface: Surface,
    pub physical_device: PhysicalDevice,
    pub device: Device,
    pub present_mode: PresentMode,
    pub width: u32,
    pub height: u32,
}

impl Default for SwapchainConfig {
    fn default() -> Self {
        Self {
            surface: Surface::new(SurfaceConfig::default()).unwrap(),
            physical_device: PhysicalDevice::new(PhysicalDeviceConfig::default()).unwrap(),
            device: Device::new(DeviceConfig::default()).unwrap(),
            present_mode: PresentMode::Fifo,
            width: 0,
            height: 0,
        }
    }
}

#[derive(Clone)]
pub struct Swapchain {
    swapchain: Arc<swapchain::Swapchain>,
    swapchain_images: Vec<Image>,
    swapchain_width: Arc<Mutex<u32>>,
    swapchain_height: Arc<Mutex<u32>>,
    config: SwapchainConfig,
    error_object: String,
}

impl VulkanRendererObject for Swapchain {
    type Config = SwapchainConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let surface = &config.surface.surface;
        let physical_device = &config.physical_device.physical_device;
        let device = &config.device.device;

        if config.width == 0 {
            return Err(format!("{}: Undefined width", error_object));
        }
        if config.height == 0 {
            return Err(format!("{}: Undefined height", error_object));
        }

        let format = Self::surface_format(physical_device, surface, &error_object)?;

        let extent = Self::surface_extent(
            physical_device,
            surface,
            config.width,
            config.height,
            &error_object,
        )?;

        let capabilities = Self::surface_capabilities(physical_device, surface, &error_object)?;

        let swapchain = swapchain::Swapchain::new(
            device.clone(),
            surface.clone(),
            swapchain::SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count,
                image_format: format.0,
                image_color_space: format.1,
                image_extent: extent,
                image_usage: image::ImageUsage::COLOR_ATTACHMENT,
                present_mode: config.present_mode.into(),
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        let swapchain_images = swapchain
            .1
            .iter()
            .map(|image| Image {
                image: image.clone(),
                config: ImageConfig {
                    device: config.device.clone(),
                    image_flag: ImageFlag::TwoD,
                    image_format: format.0.try_into().unwrap(),
                    image_usages: vec![ImageUsage::Color],
                    count_layers: 1,
                    width: extent[0],
                    height: extent[1],
                },
                error_object: String::from(type_name::<Image>().rsplit("::").next().unwrap()),
            })
            .collect();

        Ok(Self {
            swapchain: swapchain.0,
            swapchain_images,
            swapchain_width: Arc::new(Mutex::new(extent[0])),
            swapchain_height: Arc::new(Mutex::new(extent[1])),
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Swapchain {
    fn surface_format(
        physical_device: &Arc<device::physical::PhysicalDevice>,
        surface: &Arc<swapchain::Surface>,
        error_object: &String,
    ) -> VulkanRendererResult<(format::Format, swapchain::ColorSpace)> {
        let formats = physical_device
            .surface_formats(surface, swapchain::SurfaceInfo::default())
            .map_err(|_| format!("{}: Failed to get surface formats", error_object))?;

        Ok(formats
            .iter()
            .find(|format| {
                (format.0 == format::Format::R8G8B8A8_SRGB
                    || format.0 == format::Format::B8G8R8A8_SRGB)
                    && format.1 == swapchain::ColorSpace::SrgbNonLinear
            })
            .cloned()
            .unwrap_or(formats[0]))
    }

    fn surface_extent(
        physical_device: &Arc<device::physical::PhysicalDevice>,
        surface: &Arc<swapchain::Surface>,
        width: u32,
        height: u32,
        error_object: &String,
    ) -> VulkanRendererResult<[u32; 2]> {
        let surface_capabilities = Self::surface_capabilities(physical_device, surface, error_object)?;

        Ok(match surface_capabilities.current_extent {
            Some(extent) if extent[0] == u32::MAX => [
                width.clamp(
                    surface_capabilities.min_image_extent[0],
                    surface_capabilities.max_image_extent[0],
                ),
                height.clamp(
                    surface_capabilities.min_image_extent[1],
                    surface_capabilities.max_image_extent[1],
                ),
            ],
            Some(extent) => extent,
            None => [
                width.clamp(
                    surface_capabilities.min_image_extent[0],
                    surface_capabilities.max_image_extent[0],
                ),
                height.clamp(
                    surface_capabilities.min_image_extent[1],
                    surface_capabilities.max_image_extent[1],
                ),
            ],
        })
    }

    fn surface_capabilities(
        physical_device: &Arc<device::physical::PhysicalDevice>,
        surface: &Arc<swapchain::Surface>,
        error_object: &String,
    ) -> VulkanRendererResult<swapchain::SurfaceCapabilities> {
        physical_device
            .surface_capabilities(surface, swapchain::SurfaceInfo::default())
            .map_err(|_| format!("{}: Failed to get surface capabilities", error_object))
    }

    pub fn images(&self) -> Vec<Image> {
        self.swapchain_images.clone()
    }

    pub fn width(&self) -> u32 {
        *self.swapchain_width.lock().unwrap()
    }

    pub fn height(&self) -> u32 {
        *self.swapchain_height.lock().unwrap()
    }

    fn recreate_swapchain(
        &mut self,
        window: Option<sdl3::video::Window>,
    ) -> VulkanRendererResult<()> {
        let surface = &self.config.surface.surface;
        let physical_device = &self.config.physical_device.physical_device;

        let format = Self::surface_format(physical_device, surface, &self.error_object)?;

        let size = window
            .as_ref()
            .ok_or_else(|| format!("{}: Window not created", self.error_object))?
            .size_in_pixels();

        let extent =
            Self::surface_extent(physical_device, surface, size.0, size.1, &self.error_object)?;

        let swapchain = self
            .swapchain
            .recreate(swapchain::SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .map_err(|_| format!("{0}: Failed to recreate {0}", self.error_object))?;

        let swapchain_images = swapchain
            .1
            .iter()
            .map(|image| Image {
                image: image.clone(),
                config: ImageConfig {
                    device: self.config.device.clone(),
                    image_flag: ImageFlag::TwoD,
                    image_format: format.0.try_into().unwrap(),
                    image_usages: vec![ImageUsage::Color],
                    count_layers: 1,
                    width: extent[0],
                    height: extent[1],
                },
                error_object: String::from(type_name::<Image>().rsplit("::").next().unwrap()),
            })
            .collect();

        self.swapchain = swapchain.0;
        self.swapchain_images = swapchain_images;
        *self.swapchain_width.lock().unwrap() = extent[0];
        *self.swapchain_height.lock().unwrap() = extent[1];
        Ok(())
    }
}

#[derive(Clone)]
pub struct PipelineLayoutConfig {
    pub device: Device,
    pub descriptor_set_layouts: Vec<DescriptorSetLayout>,
}

impl Default for PipelineLayoutConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            descriptor_set_layouts: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    pipeline_layout: Arc<pipeline::layout::PipelineLayout>,
    config: PipelineLayoutConfig,
    error_object: String,
}

impl VulkanRendererObject for PipelineLayout {
    type Config = PipelineLayoutConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let device = &config.device.device;

        if config.descriptor_set_layouts.is_empty() {
            return Err(format!("{}: Undefined DescriptorSetLayouts", error_object));
        }

        let pipeline_layout = pipeline::layout::PipelineLayout::new(
            device.clone(),
            pipeline::layout::PipelineLayoutCreateInfo {
                set_layouts: config
                    .descriptor_set_layouts
                    .iter()
                    .map(|descriptor_set_layout| {
                        descriptor_set_layout.descriptor_set_layout.clone()
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            pipeline_layout,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    ColorAttachment,
    DepthAttachment,
    Present,
    ColorSampler,
    DepthSampler,
    Undefined,
}

impl From<ImageLayout> for image::ImageLayout {
    fn from(image_layout: ImageLayout) -> Self {
        match image_layout {
            ImageLayout::ColorAttachment => image::ImageLayout::ColorAttachmentOptimal,
            ImageLayout::DepthAttachment => image::ImageLayout::DepthStencilAttachmentOptimal,
            ImageLayout::Present => image::ImageLayout::PresentSrc,
            ImageLayout::ColorSampler => image::ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::DepthSampler => image::ImageLayout::DepthStencilReadOnlyOptimal,
            ImageLayout::Undefined => image::ImageLayout::Undefined,
        }
    }
}

#[derive(Clone)]
pub struct AttachmentInfo {
    pub image_format: ImageFormat,
    pub image_layout_final: ImageLayout,
}

impl Default for AttachmentInfo {
    fn default() -> Self {
        Self {
            image_format: ImageFormat::Rgba8Srgb,
            image_layout_final: ImageLayout::Undefined,
        }
    }
}

#[derive(Clone, Default)]
pub struct SubpassInfo {
    pub color_attachment_index: Option<u32>,
    pub depth_attachment_index: Option<u32>,
}

#[derive(Clone)]
pub struct RenderPassConfig {
    pub device: Device,
    pub attachment_infos: Vec<AttachmentInfo>,
    pub subpass_infos: Vec<SubpassInfo>,
}

impl Default for RenderPassConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            attachment_infos: Vec::new(),
            subpass_infos: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct RenderPass {
    render_pass: Arc<render_pass::RenderPass>,
    config: RenderPassConfig,
    error_object: String,
}

impl VulkanRendererObject for RenderPass {
    type Config = RenderPassConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let device = &config.device.device;

        if config.attachment_infos.is_empty() {
            return Err(format!("{}: Undefined AttachmentInfos", error_object));
        }
        if config.subpass_infos.is_empty() {
            return Err(format!("{}: Undefined SubpassInfos", error_object));
        }

        for subpass_info in &config.subpass_infos {
            if subpass_info
                .color_attachment_index
                .is_some_and(|index| index >= config.attachment_infos.len() as u32)
                || subpass_info
                    .depth_attachment_index
                    .is_some_and(|index| index >= config.attachment_infos.len() as u32)
            {
                return Err(format!("{}: Index attachment out of range", error_object));
            }
        }

        let render_pass = render_pass::RenderPass::new(
            device.clone(),
            render_pass::RenderPassCreateInfo {
                attachments: config
                    .attachment_infos
                    .iter()
                    .map(|attachment_info| render_pass::AttachmentDescription {
                        format: attachment_info.image_format.into(),
                        samples: image::SampleCount::Sample1,
                        load_op: render_pass::AttachmentLoadOp::Clear,
                        store_op: render_pass::AttachmentStoreOp::Store,
                        stencil_load_op: Some(render_pass::AttachmentLoadOp::DontCare),
                        stencil_store_op: Some(render_pass::AttachmentStoreOp::DontCare),
                        final_layout: attachment_info.image_layout_final.into(),
                        ..Default::default()
                    })
                    .collect(),
                subpasses: config
                    .subpass_infos
                    .iter()
                    .map(|subpass_info| render_pass::SubpassDescription {
                        color_attachments: vec![subpass_info.color_attachment_index.map(|index| {
                            render_pass::AttachmentReference {
                                attachment: index,
                                layout: image::ImageLayout::ColorAttachmentOptimal,
                                ..Default::default()
                            }
                        })],
                        depth_stencil_attachment: subpass_info.depth_attachment_index.map(
                            |index| render_pass::AttachmentReference {
                                attachment: index,
                                layout: image::ImageLayout::DepthStencilAttachmentOptimal,
                                ..Default::default()
                            },
                        ),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            render_pass,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone)]
pub struct FramebufferConfig {
    pub render_pass: RenderPass,
    pub image_views: Vec<ImageView>,
}

impl Default for FramebufferConfig {
    fn default() -> Self {
        Self {
            render_pass: RenderPass::new(RenderPassConfig::default()).unwrap(),
            image_views: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct Framebuffer {
    framebuffer: Arc<render_pass::Framebuffer>,
    config: FramebufferConfig,
    error_object: String,
}

impl VulkanRendererObject for Framebuffer {
    type Config = FramebufferConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let render_pass = &config.render_pass.render_pass;

        if config.image_views.is_empty() {
            return Err(format!("{}: Undefined ImageViews", error_object));
        }

        let framebuffer = render_pass::Framebuffer::new(
            render_pass.clone(),
            render_pass::FramebufferCreateInfo {
                attachments: config
                    .image_views
                    .iter()
                    .map(|image_view| image_view.image_view.clone())
                    .collect(),
                ..Default::default()
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            framebuffer,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cullface {
    Front,
    Back,
    None,
}

impl From<Cullface> for pipeline::graphics::rasterization::CullMode {
    fn from(cullface: Cullface) -> Self {
        match cullface {
            Cullface::Front => pipeline::graphics::rasterization::CullMode::Front,
            Cullface::Back => pipeline::graphics::rasterization::CullMode::Back,
            Cullface::None => pipeline::graphics::rasterization::CullMode::None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint8,
    Uint8x2,
    Uint8x3,
    Uint8x4,
    Uint16,
    Uint16x2,
    Uint16x3,
    Uint16x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Unorm8,
    Unorm8x2,
    Unorm8x3,
    Unorm8x4,
    Unorm16,
    Unorm16x2,
    Unorm16x3,
    Unorm16x4,
}

impl From<VertexFormat> for format::Format {
    fn from(format_vertex: VertexFormat) -> Self {
        match format_vertex {
            VertexFormat::Float32 => format::Format::R32_SFLOAT,
            VertexFormat::Float32x2 => format::Format::R32G32_SFLOAT,
            VertexFormat::Float32x3 => format::Format::R32G32B32_SFLOAT,
            VertexFormat::Float32x4 => format::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Uint8 => format::Format::R8_UINT,
            VertexFormat::Uint8x2 => format::Format::R8G8_UINT,
            VertexFormat::Uint8x3 => format::Format::R8G8B8_UINT,
            VertexFormat::Uint8x4 => format::Format::R8G8B8A8_UINT,
            VertexFormat::Uint16 => format::Format::R16_UINT,
            VertexFormat::Uint16x2 => format::Format::R16G16_UINT,
            VertexFormat::Uint16x3 => format::Format::R16G16B16_UINT,
            VertexFormat::Uint16x4 => format::Format::R16G16B16A16_UINT,
            VertexFormat::Uint32 => format::Format::R32_UINT,
            VertexFormat::Uint32x2 => format::Format::R32G32_UINT,
            VertexFormat::Uint32x3 => format::Format::R32G32B32_UINT,
            VertexFormat::Uint32x4 => format::Format::R32G32B32A32_UINT,
            VertexFormat::Unorm8 => format::Format::R8_UNORM,
            VertexFormat::Unorm8x2 => format::Format::R8G8_UNORM,
            VertexFormat::Unorm8x3 => format::Format::R8G8B8_UNORM,
            VertexFormat::Unorm8x4 => format::Format::R8G8B8A8_UNORM,
            VertexFormat::Unorm16 => format::Format::R16_UNORM,
            VertexFormat::Unorm16x2 => format::Format::R16G16_UNORM,
            VertexFormat::Unorm16x3 => format::Format::R16G16B16_UNORM,
            VertexFormat::Unorm16x4 => format::Format::R16G16B16A16_UNORM,
        }
    }
}

#[derive(Clone)]
pub struct AttributeBindingVertex {
    pub attribute: u32,
    pub offset: u32,
    pub format: VertexFormat,
}

impl Default for AttributeBindingVertex {
    fn default() -> Self {
        Self {
            attribute: 0,
            offset: 0,
            format: VertexFormat::Float32x3,
        }
    }
}

#[derive(Clone, Default)]
pub struct BindingVertex {
    pub binding: u32,
    pub size: u64,
    pub attributes: Vec<AttributeBindingVertex>,
}

#[derive(Clone)]
pub struct PipelineConfig {
    pub device: Device,
    pub render_pass: RenderPass,
    pub pipeline_layout: PipelineLayout,
    pub vertex_shader_module: Option<Arc<shader::ShaderModule>>,
    pub fragment_shader_module: Option<Arc<shader::ShaderModule>>,
    pub width: u32,
    pub height: u32,
    pub depth_test: bool,
    pub depth_write: bool,
    pub blending: bool,
    pub cullface: Cullface,
    pub subpass_index: u32,
    pub bindings: Vec<BindingVertex>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            device: Device::new(DeviceConfig::default()).unwrap(),
            render_pass: RenderPass::new(RenderPassConfig::default()).unwrap(),
            pipeline_layout: PipelineLayout::new(PipelineLayoutConfig::default()).unwrap(),
            vertex_shader_module: None,
            fragment_shader_module: None,
            width: 0,
            height: 0,
            depth_test: false,
            depth_write: false,
            blending: false,
            cullface: Cullface::None,
            subpass_index: 0,
            bindings: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct Pipeline {
    pipeline: Arc<pipeline::graphics::GraphicsPipeline>,
    config: PipelineConfig,
    error_object: String,
}

impl VulkanRendererObject for Pipeline {
    type Config = PipelineConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let device = &config.device.device;
        let render_pass = &config.render_pass.render_pass;
        let pipeline_layout = &config.pipeline_layout.pipeline_layout;

        if config.bindings.is_empty() {
            return Err(format!("{}: Undefined Bindings", error_object));
        }
        if config.width == 0 {
            return Err(format!("{}: Undefined width", error_object));
        }
        if config.height == 0 {
            return Err(format!("{}: Undefined height", error_object));
        }
        if config.vertex_shader_module.is_none() {
            return Err(format!("{}: Undefined vertex shader module", error_object));
        }
        if config.fragment_shader_module.is_none() {
            return Err(format!(
                "{}: Undefined fragment shader module",
                error_object
            ));
        }

        let vertex_entry = config
            .vertex_shader_module
            .clone()
            .unwrap()
            .entry_point("main")
            .ok_or_else(|| format!("{}: Vertex shader entry point not found", error_object))?;

        let fragment_entry = config
            .fragment_shader_module
            .clone()
            .unwrap()
            .entry_point("main")
            .ok_or_else(|| format!("{}: Fragment shader entry point not found", error_object))?;

        let stages = [
            pipeline::PipelineShaderStageCreateInfo::new(vertex_entry),
            pipeline::PipelineShaderStageCreateInfo::new(fragment_entry),
        ];

        let vertex_input_state = pipeline::graphics::vertex_input::VertexInputState::new()
            .bindings(config.bindings.iter().map(|binding| {
                (
                    binding.binding,
                    pipeline::graphics::vertex_input::VertexInputBindingDescription {
                        stride: binding.size as u32,
                        input_rate: pipeline::graphics::vertex_input::VertexInputRate::Vertex,
                        ..Default::default()
                    },
                )
            }))
            .attributes(config.bindings.iter().flat_map(|binding| {
                binding.attributes.iter().map(move |attribute| {
                    (
                        attribute.attribute,
                        pipeline::graphics::vertex_input::VertexInputAttributeDescription {
                            binding: binding.binding,
                            format: attribute.format.into(),
                            offset: attribute.offset,
                            ..Default::default()
                        },
                    )
                })
            }));

        let viewport_state = pipeline::graphics::viewport::ViewportState {
            viewports: [pipeline::graphics::viewport::Viewport {
                offset: [0.0, 0.0],
                extent: [config.width as f32, config.height as f32],
                depth_range: 0.0..=1.0,
            }]
            .into_iter()
            .collect(),
            scissors: [pipeline::graphics::viewport::Scissor {
                offset: [0, 0],
                extent: [config.width, config.height],
            }]
            .into_iter()
            .collect(),
            ..Default::default()
        };

        let rasterization_state = pipeline::graphics::rasterization::RasterizationState {
            cull_mode: config.cullface.into(),
            ..Default::default()
        };

        let depth_stencil_state = if config.depth_test {
            Some(pipeline::graphics::depth_stencil::DepthStencilState {
                depth: Some(pipeline::graphics::depth_stencil::DepthState {
                    write_enable: config.depth_write,
                    compare_op: pipeline::graphics::depth_stencil::CompareOp::LessOrEqual,
                }),
                ..Default::default()
            })
        } else if config.render_pass.config.subpass_infos[0]
            .depth_attachment_index
            .is_some()
        {
            Some(pipeline::graphics::depth_stencil::DepthStencilState {
                depth: Some(pipeline::graphics::depth_stencil::DepthState {
                    write_enable: false,
                    compare_op: pipeline::graphics::depth_stencil::CompareOp::Always,
                }),
                ..Default::default()
            })
        } else {
            None
        };

        let color_blend_state = if config.blending {
            pipeline::graphics::color_blend::ColorBlendState::with_attachment_states(
                1,
                pipeline::graphics::color_blend::ColorBlendAttachmentState {
                    blend: Some(pipeline::graphics::color_blend::AttachmentBlend {
                        src_color_blend_factor:
                            pipeline::graphics::color_blend::BlendFactor::SrcAlpha,
                        dst_color_blend_factor:
                            pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                        color_blend_op: pipeline::graphics::color_blend::BlendOp::Add,
                        src_alpha_blend_factor: pipeline::graphics::color_blend::BlendFactor::One,
                        dst_alpha_blend_factor:
                            pipeline::graphics::color_blend::BlendFactor::OneMinusSrcAlpha,
                        alpha_blend_op: pipeline::graphics::color_blend::BlendOp::Add,
                    }),
                    color_write_mask: pipeline::graphics::color_blend::ColorComponents::all(),
                    color_write_enable: true,
                },
            )
        } else {
            pipeline::graphics::color_blend::ColorBlendState::with_attachment_states(
                1,
                pipeline::graphics::color_blend::ColorBlendAttachmentState::default(),
            )
        };

        let subpass = render_pass::Subpass::from(render_pass.clone(), config.subpass_index)
            .ok_or_else(|| format!("{}: Invalid subpass index", error_object))?;

        let pipeline = pipeline::graphics::GraphicsPipeline::new(
            device.clone(),
            None,
            pipeline::graphics::GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(
                    pipeline::graphics::input_assembly::InputAssemblyState::default(),
                ),
                viewport_state: Some(viewport_state),
                rasterization_state: Some(rasterization_state),
                multisample_state: Some(
                    pipeline::graphics::multisample::MultisampleState::default(),
                ),
                depth_stencil_state,
                color_blend_state: Some(color_blend_state),
                dynamic_state: [
                    pipeline::DynamicState::Viewport,
                    pipeline::DynamicState::Scissor,
                ]
                .into_iter()
                .collect(),
                subpass: Some(
                    pipeline::graphics::subpass::PipelineSubpassType::BeginRenderPass(subpass),
                ),
                ..pipeline::graphics::GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
            },
        )
        .map_err(|_| format!("{0}: Failed to create {0}", error_object))?;

        Ok(Self {
            pipeline,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

#[derive(Clone)]
pub struct RendererConfig {
    pub window: Option<sdl3::video::Window>,
    pub device: Device,
    pub swapchain: Swapchain,
    pub framebuffers: Vec<Framebuffer>,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            window: None,
            device: Device::new(DeviceConfig::default()).unwrap(),
            swapchain: Swapchain::new(SwapchainConfig::default()).unwrap(),
            framebuffers: Vec::new(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum RendererResult {
    Ok,
    SwapchainOutOfDate,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    Uint16,
    Uint32,
}

pub struct Renderer {
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    current_builder:
        Option<command_buffer::AutoCommandBufferBuilder<command_buffer::PrimaryAutoCommandBuffer>>,
    cmd: Option<Arc<command_buffer::PrimaryAutoCommandBuffer>>,
    current_image_index: u32,
    acquire_future: Option<Box<dyn GpuFuture>>,
    exec_future: Option<Box<dyn GpuFuture>>,
    config: RendererConfig,
    error_object: String,
}

impl VulkanRendererObject for Renderer {
    type Config = RendererConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let device = &config.device.device;

        Ok(Self {
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
            current_builder: None,
            cmd: None,
            current_image_index: 0,
            acquire_future: None,
            exec_future: None,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl Renderer {
    pub fn next_image(&mut self) -> VulkanRendererResult<RendererResult> {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let (image_index, _, acquire_future) =
            match swapchain::acquire_next_image(self.config.swapchain.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    return Ok(RendererResult::SwapchainOutOfDate);
                }
                Err(_) => {
                    return Err(format!(
                        "{}: Failed to acquire next image",
                        self.error_object
                    ));
                }
            };

        self.current_image_index = image_index;
        self.acquire_future = Some(acquire_future.boxed());
        Ok(RendererResult::Ok)
    }

    pub fn begin_command_buffer(&mut self) -> VulkanRendererResult<()> {
        self.current_builder = Some(
            command_buffer::AutoCommandBufferBuilder::primary(
                self.config.device.command_buffer_allocator.clone(),
                self.config.device.queue_graphics.queue_family_index(),
                command_buffer::CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(|_| format!("{}: Failed to create command buffer", self.error_object))?,
        );
        Ok(())
    }

    pub fn end_command_buffer(&mut self) -> VulkanRendererResult<()> {
        self.cmd = Some(
            self.current_builder
                .take()
                .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                .build()
                .map_err(|_| format!("{}: Failed to build command buffer", self.error_object))?,
        );
        Ok(())
    }

    pub fn submit(&mut self) -> VulkanRendererResult<()> {
        let acquire_future = self
            .acquire_future
            .take()
            .ok_or_else(|| format!("{}: No acquire future", self.error_object))?;
        let cmd = self
            .cmd
            .take()
            .ok_or_else(|| format!("{}: No command buffer", self.error_object))?;

        let exec_future = self
            .previous_frame_end
            .take()
            .ok_or_else(|| format!("{}: No previous frame end", self.error_object))?
            .join(acquire_future)
            .then_execute(self.config.device.queue_graphics.clone(), cmd)
            .map_err(|_| format!("{}: Failed to execute command buffer", self.error_object))?;

        self.exec_future = Some(exec_future.boxed());
        Ok(())
    }

    pub fn present(&mut self) -> VulkanRendererResult<RendererResult> {
        let exec_future = self
            .exec_future
            .take()
            .ok_or_else(|| format!("{}: No exec future", self.error_object))?;

        let final_future = exec_future
            .then_swapchain_present(
                self.config.device.queue_graphics.clone(),
                swapchain::SwapchainPresentInfo::swapchain_image_index(
                    self.config.swapchain.swapchain.clone(),
                    self.current_image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match final_future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                return Ok(RendererResult::SwapchainOutOfDate);
            }
            Err(_) => return Err(format!("{}: Failed to flush future", self.error_object)),
        }
        Ok(RendererResult::Ok)
    }

    pub fn begin_render_pass(&mut self, render_pass: &RenderPass) -> VulkanRendererResult<()> {
        let framebuffer = &self.config.framebuffers[self.current_image_index as usize].framebuffer;

        let render_pass = &render_pass.render_pass;

        self.current_builder
            .as_mut()
            .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
            .begin_render_pass(
                vulkano::command_buffer::RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0.into())],
                    render_pass: render_pass.clone(),
                    ..vulkano::command_buffer::RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                vulkano::command_buffer::SubpassBeginInfo {
                    contents: vulkano::command_buffer::SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|_| format!("{}: Failed to begin render pass", self.error_object))?;
        Ok(())
    }

    pub fn end_render_pass(&mut self) -> VulkanRendererResult<()> {
        self.current_builder
            .as_mut()
            .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
            .end_render_pass(vulkano::command_buffer::SubpassEndInfo {
                ..Default::default()
            })
            .map_err(|_| format!("{}: Failed to end render pass", self.error_object))?;
        Ok(())
    }

    pub fn next_subpass(&mut self) -> VulkanRendererResult<()> {
        self.current_builder
            .as_mut()
            .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
            .next_subpass(
                vulkano::command_buffer::SubpassEndInfo::default(),
                vulkano::command_buffer::SubpassBeginInfo {
                    contents: vulkano::command_buffer::SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .map_err(|_| format!("{}: Failed to go to next subpass", self.error_object))?;
        Ok(())
    }

    pub fn bind_pipeline(&mut self, pipeline: &Pipeline) -> VulkanRendererResult<()> {
        self.current_builder
            .as_mut()
            .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
            .bind_pipeline_graphics(pipeline.pipeline.clone())
            .map_err(|_| format!("{}: Failed to bind pipeline", self.error_object))?
            .set_viewport(
                0,
                [vulkano::pipeline::graphics::viewport::Viewport {
                    offset: [0.0, 0.0],
                    extent: [
                        self.config.swapchain.width() as f32,
                        self.config.swapchain.height() as f32,
                    ],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .map_err(|_| format!("{}: Failed to set viewport", self.error_object))?
            .set_scissor(
                0,
                [vulkano::pipeline::graphics::viewport::Scissor {
                    offset: [0, 0],
                    extent: [
                        self.config.swapchain.width(),
                        self.config.swapchain.height(),
                    ],
                }]
                .into_iter()
                .collect(),
            )
            .map_err(|_| format!("{}: Failed to set scissor", self.error_object))?;
        Ok(())
    }

    pub fn bind_vertex_buffers(
        &mut self,
        vertex_buffers: Vec<(u32, &Buffer)>,
    ) -> VulkanRendererResult<()> {
        for vertex_buffer in &vertex_buffers {
            self.current_builder
                .as_mut()
                .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                .bind_vertex_buffers(
                    vertex_buffer.0,
                    vertex_buffer
                        .1
                        .subbuffer
                        .clone()
                        .ok_or_else(|| format!("{}: Undefined VertexBuffer", self.error_object))?,
                )
                .map_err(|_| format!("{}: Failed to bind vertex buffer", self.error_object))?;
        }
        Ok(())
    }

    pub fn bind_index_buffer(
        &mut self,
        index_buffer: &Buffer,
        index_type: IndexType,
    ) -> VulkanRendererResult<()> {
        match index_type {
            IndexType::Uint16 => {
                self.current_builder
                    .as_mut()
                    .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                    .bind_index_buffer(
                        index_buffer
                            .subbuffer
                            .clone()
                            .ok_or_else(|| format!("{}: Undefined IndexBuffer", self.error_object))?
                            .reinterpret() as buffer::Subbuffer<[u16]>,
                    )
                    .map_err(|_| format!("{}: Failed to bind index buffer", self.error_object))?;
            }
            IndexType::Uint32 => {
                self.current_builder
                    .as_mut()
                    .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                    .bind_index_buffer(
                        index_buffer
                            .subbuffer
                            .clone()
                            .ok_or_else(|| format!("{}: Undefined IndexBuffer", self.error_object))?
                            .reinterpret() as buffer::Subbuffer<[u32]>,
                    )
                    .map_err(|_| format!("{}: Failed to bind index buffer", self.error_object))?;
            }
        }
        Ok(())
    }

    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_layout: &PipelineLayout,
        descriptor_sets: Vec<(u32, &DescriptorSet)>,
    ) -> VulkanRendererResult<()> {
        for descriptor_set in &descriptor_sets {
            self.current_builder
                .as_mut()
                .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
                    pipeline_layout.pipeline_layout.clone(),
                    descriptor_set.0,
                    descriptor_set.1.descriptor_set.clone(),
                )
                .map_err(|_| format!("{}: Failed to bind descriptor set", self.error_object))?;
        }
        Ok(())
    }

    pub fn draw(&mut self, count_draw_vertexes: u32) -> VulkanRendererResult<()> {
        unsafe {
            self.current_builder
                .as_mut()
                .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                .draw(count_draw_vertexes, 1, 0, 0)
                .map_err(|_| format!("{}: Failed to draw", self.error_object))?;
        }
        Ok(())
    }

    pub fn draw_indexed(&mut self, count_draw_indexes: u32) -> VulkanRendererResult<()> {
        unsafe {
            self.current_builder
                .as_mut()
                .ok_or_else(|| format!("{}: Command buffer not started", self.error_object))?
                .draw_indexed(count_draw_indexes, 1, 0, 0, 0)
                .map_err(|_| format!("{}: Failed to draw indexed", self.error_object))?;
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> VulkanRendererResult<()> {
        self.previous_frame_end = Some(sync::now(self.config.device.device.clone()).boxed());

        self.config
            .swapchain
            .recreate_swapchain(self.config.window.clone())?;

        let swapchain_image_views: Vec<ImageView> = self
            .config
            .swapchain
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
            device: self.config.device.clone(),
            image_flag: ImageFlag::TwoD,
            image_format: ImageFormat::D32Sfloat,
            image_usages: vec![ImageUsage::Depth],
            count_layers: 1,
            width: self.config.swapchain.width(),
            height: self.config.swapchain.height(),
        })?;

        let depth_image_view = ImageView::new(ImageViewConfig {
            image: depth_image.clone(),
            image_view_type: ImageViewType::TwoD,
            image_aspect: ImageAspect::Depth,
            index_layer: 0,
            count_layers: 1,
        })?;

        self.config.framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                Framebuffer::new(FramebufferConfig {
                    render_pass: self.config.framebuffers[0].config.render_pass.clone(),
                    image_views: vec![image_view.clone(), depth_image_view.clone()],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    pub fn render_imgui(&mut self, imgui_context: &mut ImGuiContext) -> VulkanRendererResult<()> {
        imgui_context
            .imgui_renderer
            .draw(self, imgui_context.imgui_context.render())
            .map_err(|err| format!("{}: Failed to draw ImGui {:?}", self.error_object, err))?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct ImGuiContextConfig {
    pub instance: Instance,
    pub physical_device: PhysicalDevice,
    pub device: Device,
    pub swapchain: Swapchain,
    pub render_pass: RenderPass,
}

impl Default for ImGuiContextConfig {
    fn default() -> Self {
        Self {
            instance: Instance::new(InstanceConfig::default()).unwrap(),
            physical_device: PhysicalDevice::new(PhysicalDeviceConfig::default()).unwrap(),
            device: Device::new(DeviceConfig::default()).unwrap(),
            swapchain: Swapchain::new(SwapchainConfig::default()).unwrap(),
            render_pass: RenderPass::new(RenderPassConfig::default()).unwrap(),
        }
    }
}

pub struct ImGuiContext {
    imgui_context: imgui::Context,
    imgui_renderer: imgui_renderer::ImGuiRenderer,
    config: ImGuiContextConfig,
    error_object: String,
}

impl VulkanRendererObject for ImGuiContext {
    type Config = ImGuiContextConfig;

    fn new(config: Self::Config) -> VulkanRendererResult<Self> {
        let error_object = String::from(type_name::<Self>().rsplit("::").next().unwrap());

        let mut imgui_context = imgui::Context::create();

        imgui_context
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);

        let io = imgui_context.io_mut();

        io.display_size = [
            config.swapchain.width() as f32,
            config.swapchain.height() as f32,
        ];

        let imgui_renderer = imgui_renderer::ImGuiRenderer::new(
            &mut imgui_context,
            config.physical_device.clone(),
            config.device.clone(),
            config.render_pass.clone(),
            0,
        )
        .map_err(|_| format!("{}: Failed to create ImGuiRenderer", error_object))?;

        Ok(Self {
            imgui_context,
            imgui_renderer,
            config,
            error_object,
        })
    }

    fn config(&self) -> Self::Config {
        self.config.clone()
    }
}

impl ImGuiContext {
    pub fn new_frame(&mut self, delta_time: f32) -> VulkanRendererResult<&mut imgui::Ui> {
        let io = self.imgui_context.io_mut();

        io.display_size = [
            self.config.swapchain.width() as f32,
            self.config.swapchain.height() as f32,
        ];

        io.delta_time = delta_time;

        Ok(self.imgui_context.new_frame())
    }

    pub fn reset_event(&mut self) {
        let io = self.imgui_context.io_mut();

        for key in &mut io.keys_down {
            *key = false;
        }

        io.key_ctrl = false;
        io.key_shift = false;
        io.key_alt = false;
        io.key_super = false;

        io.clear_input_characters();

        io.nav_inputs = [0.0; imgui::NavInput::COUNT];

        io.app_focus_lost = true;
    }

    pub fn update_event(&mut self, event: &sdl3::event::Event) -> bool {
        let io = self.imgui_context.io_mut();

        match *event {
            sdl3::event::Event::MouseMotion { x, y, .. } => {
                io.add_mouse_pos_event([x, y]);
                true
            }
            sdl3::event::Event::MouseWheel { x, y, .. } => {
                io.add_mouse_wheel_event([x, y]);
                true
            }

            sdl3::event::Event::MouseButtonDown { mouse_btn, .. } => {
                Self::handle_mouse_button(io, &mouse_btn, true);
                true
            }

            sdl3::event::Event::MouseButtonUp { mouse_btn, .. } => {
                Self::handle_mouse_button(io, &mouse_btn, false);
                true
            }

            sdl3::event::Event::TextInput { ref text, .. } => {
                text.chars().for_each(|c| io.add_input_character(c));
                true
            }

            sdl3::event::Event::KeyDown {
                scancode: Some(key),
                keymod,
                ..
            } => {
                Self::handle_key_modifier(io, &keymod);
                Self::handle_key(io, &key, true);
                true
            }

            sdl3::event::Event::KeyUp {
                scancode: Some(key),
                keymod,
                ..
            } => {
                Self::handle_key_modifier(io, &keymod);
                Self::handle_key(io, &key, false);
                true
            }

            _ => false,
        }
    }

    fn handle_mouse_button(io: &mut imgui::Io, button: &sdl3::mouse::MouseButton, pressed: bool) {
        match button {
            sdl3::mouse::MouseButton::Left => {
                io.add_mouse_button_event(imgui::MouseButton::Left, pressed)
            }
            sdl3::mouse::MouseButton::Right => {
                io.add_mouse_button_event(imgui::MouseButton::Right, pressed)
            }
            sdl3::mouse::MouseButton::Middle => {
                io.add_mouse_button_event(imgui::MouseButton::Middle, pressed)
            }
            sdl3::mouse::MouseButton::X1 => {
                io.add_mouse_button_event(imgui::MouseButton::Extra1, pressed)
            }
            sdl3::mouse::MouseButton::X2 => {
                io.add_mouse_button_event(imgui::MouseButton::Extra2, pressed)
            }
            _ => {}
        }
    }

    fn handle_key_modifier(io: &mut imgui::Io, keymod: &sdl3::keyboard::Mod) {
        io.add_key_event(
            imgui::Key::ModShift,
            keymod.intersects(sdl3::keyboard::Mod::LSHIFTMOD | sdl3::keyboard::Mod::RSHIFTMOD),
        );
        io.add_key_event(
            imgui::Key::ModCtrl,
            keymod.intersects(sdl3::keyboard::Mod::LCTRLMOD | sdl3::keyboard::Mod::RCTRLMOD),
        );
        io.add_key_event(
            imgui::Key::ModAlt,
            keymod.intersects(sdl3::keyboard::Mod::LALTMOD | sdl3::keyboard::Mod::RALTMOD),
        );
        io.add_key_event(
            imgui::Key::ModSuper,
            keymod.intersects(sdl3::keyboard::Mod::LGUIMOD | sdl3::keyboard::Mod::RGUIMOD),
        );
    }

    fn handle_key(io: &mut imgui::Io, key: &sdl3::keyboard::Scancode, pressed: bool) {
        let igkey = match key {
            sdl3::keyboard::Scancode::A => imgui::Key::A,
            sdl3::keyboard::Scancode::B => imgui::Key::B,
            sdl3::keyboard::Scancode::C => imgui::Key::C,
            sdl3::keyboard::Scancode::D => imgui::Key::D,
            sdl3::keyboard::Scancode::E => imgui::Key::E,
            sdl3::keyboard::Scancode::F => imgui::Key::F,
            sdl3::keyboard::Scancode::G => imgui::Key::G,
            sdl3::keyboard::Scancode::H => imgui::Key::H,
            sdl3::keyboard::Scancode::I => imgui::Key::I,
            sdl3::keyboard::Scancode::J => imgui::Key::J,
            sdl3::keyboard::Scancode::K => imgui::Key::K,
            sdl3::keyboard::Scancode::L => imgui::Key::L,
            sdl3::keyboard::Scancode::M => imgui::Key::M,
            sdl3::keyboard::Scancode::N => imgui::Key::N,
            sdl3::keyboard::Scancode::O => imgui::Key::O,
            sdl3::keyboard::Scancode::P => imgui::Key::P,
            sdl3::keyboard::Scancode::Q => imgui::Key::Q,
            sdl3::keyboard::Scancode::R => imgui::Key::R,
            sdl3::keyboard::Scancode::S => imgui::Key::S,
            sdl3::keyboard::Scancode::T => imgui::Key::T,
            sdl3::keyboard::Scancode::U => imgui::Key::U,
            sdl3::keyboard::Scancode::V => imgui::Key::V,
            sdl3::keyboard::Scancode::W => imgui::Key::W,
            sdl3::keyboard::Scancode::X => imgui::Key::X,
            sdl3::keyboard::Scancode::Y => imgui::Key::Y,
            sdl3::keyboard::Scancode::Z => imgui::Key::Z,
            sdl3::keyboard::Scancode::_1 => imgui::Key::Keypad1,
            sdl3::keyboard::Scancode::_2 => imgui::Key::Keypad2,
            sdl3::keyboard::Scancode::_3 => imgui::Key::Keypad3,
            sdl3::keyboard::Scancode::_4 => imgui::Key::Keypad4,
            sdl3::keyboard::Scancode::_5 => imgui::Key::Keypad5,
            sdl3::keyboard::Scancode::_6 => imgui::Key::Keypad6,
            sdl3::keyboard::Scancode::_7 => imgui::Key::Keypad7,
            sdl3::keyboard::Scancode::_8 => imgui::Key::Keypad8,
            sdl3::keyboard::Scancode::_9 => imgui::Key::Keypad9,
            sdl3::keyboard::Scancode::_0 => imgui::Key::Keypad0,
            sdl3::keyboard::Scancode::Return => imgui::Key::Enter,
            sdl3::keyboard::Scancode::Escape => imgui::Key::Escape,
            sdl3::keyboard::Scancode::Backspace => imgui::Key::Backspace,
            sdl3::keyboard::Scancode::Tab => imgui::Key::Tab,
            sdl3::keyboard::Scancode::Space => imgui::Key::Space,
            sdl3::keyboard::Scancode::Minus => imgui::Key::Minus,
            sdl3::keyboard::Scancode::Equals => imgui::Key::Equal,
            sdl3::keyboard::Scancode::LeftBracket => imgui::Key::LeftBracket,
            sdl3::keyboard::Scancode::RightBracket => imgui::Key::RightBracket,
            sdl3::keyboard::Scancode::Backslash => imgui::Key::Backslash,
            sdl3::keyboard::Scancode::Semicolon => imgui::Key::Semicolon,
            sdl3::keyboard::Scancode::Apostrophe => imgui::Key::Apostrophe,
            sdl3::keyboard::Scancode::Grave => imgui::Key::GraveAccent,
            sdl3::keyboard::Scancode::Comma => imgui::Key::Comma,
            sdl3::keyboard::Scancode::Period => imgui::Key::Period,
            sdl3::keyboard::Scancode::Slash => imgui::Key::Slash,
            sdl3::keyboard::Scancode::CapsLock => imgui::Key::CapsLock,
            sdl3::keyboard::Scancode::F1 => imgui::Key::F1,
            sdl3::keyboard::Scancode::F2 => imgui::Key::F2,
            sdl3::keyboard::Scancode::F3 => imgui::Key::F3,
            sdl3::keyboard::Scancode::F4 => imgui::Key::F4,
            sdl3::keyboard::Scancode::F5 => imgui::Key::F5,
            sdl3::keyboard::Scancode::F6 => imgui::Key::F6,
            sdl3::keyboard::Scancode::F7 => imgui::Key::F7,
            sdl3::keyboard::Scancode::F8 => imgui::Key::F8,
            sdl3::keyboard::Scancode::F9 => imgui::Key::F9,
            sdl3::keyboard::Scancode::F10 => imgui::Key::F10,
            sdl3::keyboard::Scancode::F11 => imgui::Key::F11,
            sdl3::keyboard::Scancode::F12 => imgui::Key::F12,
            sdl3::keyboard::Scancode::PrintScreen => imgui::Key::PrintScreen,
            sdl3::keyboard::Scancode::ScrollLock => imgui::Key::ScrollLock,
            sdl3::keyboard::Scancode::Pause => imgui::Key::Pause,
            sdl3::keyboard::Scancode::Insert => imgui::Key::Insert,
            sdl3::keyboard::Scancode::Home => imgui::Key::Home,
            sdl3::keyboard::Scancode::PageUp => imgui::Key::PageUp,
            sdl3::keyboard::Scancode::Delete => imgui::Key::Delete,
            sdl3::keyboard::Scancode::End => imgui::Key::End,
            sdl3::keyboard::Scancode::PageDown => imgui::Key::PageDown,
            sdl3::keyboard::Scancode::Right => imgui::Key::RightArrow,
            sdl3::keyboard::Scancode::Left => imgui::Key::LeftArrow,
            sdl3::keyboard::Scancode::Down => imgui::Key::DownArrow,
            sdl3::keyboard::Scancode::Up => imgui::Key::UpArrow,
            sdl3::keyboard::Scancode::KpDivide => imgui::Key::KeypadDivide,
            sdl3::keyboard::Scancode::KpMultiply => imgui::Key::KeypadMultiply,
            sdl3::keyboard::Scancode::KpMinus => imgui::Key::KeypadSubtract,
            sdl3::keyboard::Scancode::KpPlus => imgui::Key::KeypadAdd,
            sdl3::keyboard::Scancode::KpEnter => imgui::Key::KeypadEnter,
            sdl3::keyboard::Scancode::Kp1 => imgui::Key::Keypad1,
            sdl3::keyboard::Scancode::Kp2 => imgui::Key::Keypad2,
            sdl3::keyboard::Scancode::Kp3 => imgui::Key::Keypad3,
            sdl3::keyboard::Scancode::Kp4 => imgui::Key::Keypad4,
            sdl3::keyboard::Scancode::Kp5 => imgui::Key::Keypad5,
            sdl3::keyboard::Scancode::Kp6 => imgui::Key::Keypad6,
            sdl3::keyboard::Scancode::Kp7 => imgui::Key::Keypad7,
            sdl3::keyboard::Scancode::Kp8 => imgui::Key::Keypad8,
            sdl3::keyboard::Scancode::Kp9 => imgui::Key::Keypad9,
            sdl3::keyboard::Scancode::Kp0 => imgui::Key::Keypad0,
            sdl3::keyboard::Scancode::KpPeriod => imgui::Key::KeypadDecimal,
            sdl3::keyboard::Scancode::Application => imgui::Key::Menu,
            sdl3::keyboard::Scancode::KpEquals => imgui::Key::KeypadEqual,
            sdl3::keyboard::Scancode::Menu => imgui::Key::Menu,
            sdl3::keyboard::Scancode::LCtrl => imgui::Key::LeftCtrl,
            sdl3::keyboard::Scancode::LShift => imgui::Key::LeftShift,
            sdl3::keyboard::Scancode::LAlt => imgui::Key::LeftAlt,
            sdl3::keyboard::Scancode::LGui => imgui::Key::LeftSuper,
            sdl3::keyboard::Scancode::RCtrl => imgui::Key::RightCtrl,
            sdl3::keyboard::Scancode::RShift => imgui::Key::RightShift,
            sdl3::keyboard::Scancode::RAlt => imgui::Key::RightAlt,
            sdl3::keyboard::Scancode::RGui => imgui::Key::RightSuper,
            _ => {
                return;
            }
        };

        io.add_key_event(igkey, pressed);
    }
}
