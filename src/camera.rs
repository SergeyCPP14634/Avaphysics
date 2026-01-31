#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct CameraData {
    pub projection: glm::Mat4,
    pub view: glm::Mat4,
    pub position_camera: glm::Vec4,
}

impl Default for CameraData {
    fn default() -> Self {
        Self {
            projection: glm::Mat4::identity(),
            view: glm::Mat4::identity(),
            position_camera: glm::vec4(0.0, 0.0, 0.0, 1.0),
        }
    }
}

pub struct Camera {
    pub width: u32,
    pub height: u32,
    pub camera_data: CameraData,
    angle_x: f32,
    angle_y: f32,
    pub sensitivity: f32,
    pub speed: f32,
    front: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    select_camera: bool,
    accumulated_dx: f32,
    accumulated_dy: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            camera_data: CameraData::default(),
            angle_x: 0.0,
            angle_y: 0.0,
            sensitivity: 0.0015,
            speed: 5.0,
            front: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            select_camera: false,
            accumulated_dx: 0.0,
            accumulated_dy: 0.0,
        }
    }
}

impl Camera {
    pub fn new(position: glm::Vec3, width: u32, height: u32, sensitivity: f32, speed: f32) -> Self {
        let mut cam = Self {
            select_camera: true,
            width,
            height,
            sensitivity,
            speed,
            camera_data: CameraData {
                projection: glm::perspective(
                    width as f32 / height as f32,
                    45_f32.to_radians(),
                    0.1,
                    10000.0,
                ),
                view: glm::Mat4::identity(),
                position_camera: glm::vec4(position.x, position.y, position.z, 1.0),
            },
            angle_x: 0.0,
            angle_y: 0.0,
            front: false,
            back: false,
            left: false,
            right: false,
            up: false,
            down: false,
            accumulated_dx: 0.0,
            accumulated_dy: 0.0,
        };
        cam.update_view_matrix();
        cam
    }

    pub fn update(&mut self, delta_time: f32) {
        self.camera_data.projection = glm::perspective(
            self.width as f32 / self.height as f32,
            45_f32.to_radians(),
            0.1,
            10000.0,
        );

        if !self.select_camera {
            return;
        }

        self.angle_x += self.accumulated_dx * self.sensitivity;
        self.angle_y += self.accumulated_dy * self.sensitivity;
        self.angle_y = self.angle_y.clamp(-89.0, 89.0);

        self.accumulated_dx = 0.0;
        self.accumulated_dy = 0.0;

        let actual_speed = self.speed * delta_time;
        let mut delta = glm::vec3(0.0, 0.0, 0.0);

        let yaw = self.angle_x.to_radians();
        let direction = glm::vec3(yaw.sin(), 0.0, -yaw.cos());

        if self.front ^ self.back {
            let sign = if self.front { 1.0 } else { -1.0 };
            delta += direction * sign * actual_speed;
        }

        if self.left ^ self.right {
            let sign = if self.left { -1.0 } else { 1.0 };
            let left = glm::vec3(-direction.z, 0.0, direction.x);
            delta += left * sign * actual_speed;
        }

        if self.up ^ self.down {
            let sign = if self.up { 1.0 } else { -1.0 };
            delta.y += sign * actual_speed;
        }

        self.camera_data.position_camera.x += delta.x;
        self.camera_data.position_camera.y += delta.y;
        self.camera_data.position_camera.z += delta.z;

        self.update_view_matrix();
    }

    pub fn update_event(&mut self, event: &sdl3::event::Event) {
        use sdl3::event::Event;
        use sdl3::keyboard::Keycode;

        match event {
            Event::KeyDown {
                keycode: Some(keycode),
                ..
            } => match keycode {
                Keycode::W => self.front = true,
                Keycode::S => self.back = true,
                Keycode::A => self.left = true,
                Keycode::D => self.right = true,
                Keycode::Space => self.up = true,
                Keycode::LShift => self.down = true,
                _ => {}
            },
            Event::KeyUp {
                keycode: Some(keycode),
                ..
            } => match keycode {
                Keycode::W => self.front = false,
                Keycode::S => self.back = false,
                Keycode::A => self.left = false,
                Keycode::D => self.right = false,
                Keycode::Space => self.up = false,
                Keycode::LShift => self.down = false,
                _ => {}
            },
            _ => {}
        }

        self.handle_mouse_motion(event);
    }

    fn handle_mouse_motion(&mut self, event: &sdl3::event::Event) {
        if let sdl3::event::Event::MouseMotion { xrel, yrel, .. } = event
            && self.select_camera
        {
            self.accumulated_dx += *xrel;
            self.accumulated_dy += *yrel;
        }
    }

    pub fn toggle_cursor_mode(&mut self, sdl: &sdl3::Sdl, window: &sdl3::video::Window) {
        self.select_camera = !self.select_camera;
        if self.select_camera {
            sdl.mouse().show_cursor(false);
            sdl.mouse().set_relative_mouse_mode(window, true);
        } else {
            sdl.mouse().show_cursor(true);
            sdl.mouse().set_relative_mouse_mode(window, false);
        }
    }

    fn update_view_matrix(&mut self) {
        let yaw = self.angle_x.to_radians();
        let pitch = self.angle_y.to_radians();

        let direction = glm::vec3(
            yaw.sin() * pitch.cos(),
            -pitch.sin(),
            -yaw.cos() * pitch.cos(),
        );

        let position = glm::vec3(
            self.camera_data.position_camera.x,
            self.camera_data.position_camera.y,
            self.camera_data.position_camera.z,
        );

        self.camera_data.view = glm::look_at(
            &position,
            &(position + direction),
            &glm::vec3(0.0, 1.0, 0.0),
        );
    }

    pub fn is_selected(&self) -> bool {
        self.select_camera
    }
}
