extern crate nalgebra_glm as glm;

use crate::area::Area;
use crate::body::{Body, BodyType, PhysicalBody, RenderBody};
use crate::camera::*;
use imgui::Drag;
use imgui::*;

pub struct MaxLengthEnforcer {
    pub max_length: usize,
}

impl InputTextCallbackHandler for MaxLengthEnforcer {
    fn on_edit(&mut self, mut data: TextCallbackData) {
        let current = data.str();
        if current.len() > self.max_length {
            let truncated: String = current.chars().take(self.max_length).collect();
            data.clear();
            data.push_str(&truncated);
        }
    }
}

pub struct Gui {
    pub area: Area,
    selected_body_index: i32,
    show_create_window: bool,
    new_body: Body,
    time: f32,
    name_body_buffer: String,
    texture_buffer: String,
}

impl Gui {
    pub fn new(area: Area) -> Self {
        let physical_body = PhysicalBody::default();
        let render_body = RenderBody::new(
            "New Body".to_string(),
            "textures/earth.jpg".to_string(),
            BodyType::Sphere,
            glm::vec3(1.0, 1.0, 1.0),
        );

        let mut new_body = Body::new(physical_body, render_body);
        new_body.physical_body.sync_edit_from_runtime();

        Self {
            area,
            selected_body_index: -1,
            show_create_window: false,
            new_body,
            time: 0.0,
            name_body_buffer: String::new(),
            texture_buffer: String::new(),
        }
    }

    pub fn update(&mut self, ui: &Ui, camera: &Camera) -> Result<(), String> {
        let _colors = [
            ui.push_style_color(StyleColor::WindowBg, [0.1, 0.1, 0.1, 0.95]),
            ui.push_style_color(StyleColor::Header, [0.2, 0.4, 0.8, 0.8]),
            ui.push_style_color(StyleColor::HeaderHovered, [0.3, 0.5, 0.9, 0.8]),
            ui.push_style_color(StyleColor::HeaderActive, [0.15, 0.35, 0.75, 0.8]),
            ui.push_style_color(StyleColor::Button, [0.2, 0.2, 0.2, 1.0]),
            ui.push_style_color(StyleColor::ButtonHovered, [0.3, 0.3, 0.3, 1.0]),
            ui.push_style_color(StyleColor::ButtonActive, [0.15, 0.15, 0.15, 1.0]),
        ];

        let _vars = [
            ui.push_style_var(StyleVar::WindowRounding(5.0)),
            ui.push_style_var(StyleVar::FrameRounding(3.0)),
            ui.push_style_var(StyleVar::ItemSpacing([8.0, 4.0])),
            ui.push_style_var(StyleVar::FramePadding([8.0, 4.0])),
        ];

        self.render_main_menu(ui)?;
        self.render_area_editor(ui)?;
        self.render_body_list(ui)?;
        self.render_body_editor(ui)?;
        self.render_simulation_controls(ui)?;
        self.render_real_time_parameters(ui)?;
        self.render_create_body_window(ui)?;
        self.render_coordinate_gizmo(ui, camera)?;

        Ok(())
    }

    fn render_main_menu(&mut self, ui: &Ui) -> Result<(), String> {
        if let Some(menu_bar) = ui.begin_main_menu_bar() {
            if let Some(menu) = ui.begin_menu("Bodies") {
                if ui.menu_item("Create Body") {
                    self.show_create_window = true;
                }
                menu.end();
            }
            menu_bar.end();
        }
        Ok(())
    }

    fn render_area_editor(&mut self, ui: &Ui) -> Result<(), String> {
        ui.window("Area Editor")
            .build(|| {
                let area = &mut self.area;

                let mut gravity = area.current_gravity;
                if Drag::new("Gravity")
                    .range(-100.0, 100.0)
                    .speed(0.1)
                    .build(ui, &mut gravity)
                {
                    area.current_gravity = gravity;

                    for i in 0..area.count_bodies() {
                        if let Some(body) = area.body_mut(i) {
                            body.physical_body
                                .edit_params
                                .update_current_gravity(gravity);
                            body.physical_body.apply_edit_to_runtime();

                            if area.is_simulating() {
                                area.update_time(0.0)?;
                            } else {
                                area.update_body(i)?;
                            }
                        }
                    }

                    self.new_body
                        .physical_body
                        .edit_params
                        .update_current_gravity(gravity);
                }

                let mut friction = area.current_friction;
                if Drag::new("Friction")
                    .range(-100.0, 100.0)
                    .speed(0.1)
                    .build(ui, &mut friction)
                {
                    area.current_friction = friction;
                    if area.is_simulating() {
                        area.update_time(0.0)?;
                    }
                }
                Ok(())
            })
            .unwrap_or(Ok(()))
    }

    fn render_body_list(&mut self, ui: &Ui) -> Result<(), String> {
        let mut body_to_delete: Option<usize> = None;

        ui.window("Bodies").build(|| {
            let area = &self.area;

            for i in 0..area.count_bodies() {
                let name_body = area.body(i).unwrap().render_body.name.clone();

                let available_width = ui.content_region_avail()[0];
                let button_width = 90.0;
                let selectable_width =
                    available_width - button_width - unsafe { ui.style().item_spacing[0] };
                let button_height =
                    ui.text_line_height() + unsafe { ui.style().frame_padding[1] } * 2.0;

                ui.group(|| {
                    if ui
                        .selectable_config(&name_body)
                        .selected(self.selected_body_index == i as i32)
                        .size([selectable_width, button_height])
                        .build()
                    {
                        self.selected_body_index = i as i32;
                    }

                    ui.same_line();

                    if ui.button_with_size(
                        format!("Delete Body##{}", name_body),
                        [button_width, button_height],
                    ) {
                        body_to_delete = Some(i);
                    }
                });
            }
        });

        if let Some(index) = body_to_delete {
            if let Err(err) = self.area.remove_body(index) {
                return Err(format!("Failed to delete body: {}", err));
            }

            if self.selected_body_index >= index as i32 {
                self.selected_body_index = if self.selected_body_index > 0 {
                    self.selected_body_index - 1
                } else {
                    -1
                };
            }
        }

        Ok(())
    }

    fn render_body_editor(&mut self, ui: &Ui) -> Result<(), String> {
        if self.selected_body_index < 0 {
            return Ok(());
        }

        let selected_body_index = self.selected_body_index as usize;
        if selected_body_index >= self.area.count_bodies() {
            return Ok(());
        }

        let mut should_delete = false;

        ui.window("Body Editor").build(|| {
            let area = &mut self.area;
            let mut should_apply_changes = false;

            ui.text(format!(
                "Selected Body: {}",
                area.body(selected_body_index).unwrap().render_body.name
            ));
            ui.separator();

            if ui.button("Delete Body") {
                should_delete = true;
            }
            ui.separator();

            let mut name_body_buffer = None;

            if let Some(body) = area.body_mut(selected_body_index) {
                self.name_body_buffer = body.render_body.name.clone();
                if ui
                    .input_text("Body Name", &mut self.name_body_buffer)
                    .callback(
                        imgui::InputTextCallback::EDIT,
                        MaxLengthEnforcer { max_length: 16 },
                    )
                    .enter_returns_true(true)
                    .build()
                {
                    name_body_buffer = Some(self.name_body_buffer.clone());
                }

                let body_types = ["Sphere", "Rectangle"];
                let mut current_type = body.render_body.body_type as usize;
                if ui.combo("Body Type", &mut current_type, &body_types, |item| {
                    std::borrow::Cow::Borrowed(item)
                }) {
                    body.render_body.body_type = match current_type {
                        0 => BodyType::Sphere,
                        1 => BodyType::Rectangle,
                        _ => BodyType::Sphere,
                    };
                    should_apply_changes = true;
                }

                let mut is_kinematic = body.physical_body.edit_params.is_kinematic;
                if ui.checkbox("Is Kinematic", &mut is_kinematic) {
                    body.physical_body.edit_params.is_kinematic = is_kinematic;
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                self.texture_buffer = body.render_body.texture_path.clone();
                if ui
                    .input_text("Texture Path", &mut self.texture_buffer)
                    .enter_returns_true(true)
                    .build()
                {
                    body.render_body.texture_path = self.texture_buffer.clone();
                    body.render_body.texture_is_update = true;
                    should_apply_changes = true;
                }

                match body.render_body.body_type {
                    BodyType::Sphere => {
                        let mut radius = body.render_body.dimensions.x;
                        if Drag::new("Radius")
                            .range(0.1, 100.0)
                            .speed(0.1)
                            .build(ui, &mut radius)
                        {
                            radius = if radius <= 0.0 { 0.001 } else { radius };
                            body.render_body.dimensions = glm::vec3(radius, radius, radius);
                            should_apply_changes = true;
                        }
                    }
                    BodyType::Rectangle => {
                        let mut dims = [
                            body.render_body.dimensions.x,
                            body.render_body.dimensions.y,
                            body.render_body.dimensions.z,
                        ];
                        if Drag::new("Dimensions")
                            .range(0.1, 100.0)
                            .speed(0.1)
                            .build_array(ui, &mut dims)
                        {
                            dims[0] = if dims[0] <= 0.0 { 0.001 } else { dims[0] };
                            dims[1] = if dims[1] <= 0.0 { 0.001 } else { dims[1] };
                            dims[2] = if dims[2] <= 0.0 { 0.001 } else { dims[2] };
                            body.render_body.dimensions = glm::vec3(dims[0], dims[1], dims[2]);
                            should_apply_changes = true;
                        }
                    }
                }

                ui.separator();
                ui.text("Physical Parameters");

                let mut pos = [
                    body.physical_body.edit_params.position.x,
                    body.physical_body.edit_params.position.y,
                    body.physical_body.edit_params.position.z,
                ];
                if Drag::new("Position").speed(0.1).build_array(ui, &mut pos) {
                    body.physical_body
                        .edit_params
                        .update_position(glm::vec3(pos[0], pos[1], pos[2]));
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                let min_y = match body.render_body.body_type {
                    BodyType::Sphere => body.render_body.dimensions.x,
                    BodyType::Rectangle => body.render_body.dimensions.y,
                };

                if pos[1] < min_y {
                    body.physical_body
                        .edit_params
                        .update_position(glm::vec3(pos[0], min_y, pos[2]));
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                let mut vel = [
                    body.physical_body.edit_params.velocity.x,
                    body.physical_body.edit_params.velocity.y,
                    body.physical_body.edit_params.velocity.z,
                ];
                if Drag::new("Velocity").speed(0.1).build_array(ui, &mut vel) {
                    body.physical_body
                        .edit_params
                        .update_velocity(glm::vec3(vel[0], vel[1], vel[2]));
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                let mut mass = body.physical_body.edit_params.mass;
                if Drag::new("Mass")
                    .range(0.001, 1000.0)
                    .speed(0.1)
                    .build(ui, &mut mass)
                {
                    body.physical_body.edit_params.update_mass(mass);
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                let mut friction = body.physical_body.edit_params.friction;
                if Drag::new("Friction")
                    .range(0.001, 1000.0)
                    .speed(0.1)
                    .build(ui, &mut friction)
                {
                    body.physical_body.edit_params.update_friction(friction);
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }

                let mut force = [
                    body.physical_body.edit_params.force.x,
                    body.physical_body.edit_params.force.y,
                    body.physical_body.edit_params.force.z,
                ];
                if Drag::new("Force").speed(0.1).build_array(ui, &mut force) {
                    body.physical_body
                        .edit_params
                        .update_force(glm::vec3(force[0], force[1], force[2]));
                    body.physical_body.apply_edit_to_runtime();
                    should_apply_changes = true;
                }
            }

            if let Some(name) = name_body_buffer
                && !self.area.name_bodies.contains(&name)
            {
                self.area.name_bodies.remove(
                    &self
                        .area
                        .body(selected_body_index)
                        .unwrap()
                        .render_body
                        .name
                        .clone(),
                );
                self.area
                    .body_mut(selected_body_index)
                    .unwrap()
                    .render_body
                    .name = name.clone();
                self.area.name_bodies.insert(name.clone());
                self.area
                    .body_mut(selected_body_index)
                    .unwrap()
                    .physical_body
                    .apply_edit_to_runtime();
                should_apply_changes = true;
            }

            if should_apply_changes {
                if self.area.is_simulating() {
                    self.area.update_time(0.0).unwrap();
                } else {
                    self.area.update_body(selected_body_index).unwrap();
                }
            }
        });

        if should_delete {
            if let Err(err) = self.area.remove_body(selected_body_index) {
                return Err(format!("Failed to delete body: {}", err));
            }
            self.selected_body_index = -1;
        }

        Ok(())
    }

    fn render_real_time_parameters(&self, ui: &Ui) -> Result<(), String> {
        if self.selected_body_index < 0 {
            return Ok(());
        }

        let area = &self.area;

        if self.selected_body_index >= area.count_bodies() as i32 {
            return Ok(());
        }

        if let Some(body) = area.body(self.selected_body_index as usize) {
            ui.window("Real-Time Parameters").build(|| {
                ui.text(format!(
                    "Selected Body: {}",
                    area.body(self.selected_body_index as usize)
                        .unwrap()
                        .render_body
                        .name
                ));
                ui.separator();

                let phys = &body.physical_body;

                Self::display_vector3(ui, "Position", &phys.position);
                Self::display_vector3(ui, "Displacement", &phys.displacement);
                Self::display_vector3(ui, "Velocity", &phys.velocity);
                Self::display_vector3(ui, "Acceleration", &phys.acceleration);

                ui.text(format!("Distance: {:.2}", phys.distance));
                ui.text(format!("Angular Velocity: {:.2}", phys.angular_velocity));
                ui.text(format!(
                    "Angular Acceleration: {:.2}",
                    phys.angular_acceleration
                ));
                ui.text(format!("Mass: {:.2}", phys.mass));
                ui.text(format!("Friction: {:.2}", phys.friction));

                Self::display_vector3(ui, "Force", &phys.force);
                //ui.text(format!("Net Force: {:.2}", phys.net_force));

                Self::display_vector3(ui, "Gravity Force", &phys.gravity_force);
                //Self::display_vector3(ui, "Friction Force", &phys.friction_force);
                //Self::display_vector3(ui, "Normal Force", &phys.normal_force);
                //Self::display_vector3(ui, "Elastic Force", &phys.elastic_force);
                Self::display_vector3(ui, "Momentum", &phys.momentum);
                Self::display_vector3(ui, "Impulse", &phys.impulse);

                ui.text(format!("Kinetic Energy: {:.2}", phys.kinetic_energy));
                ui.text(format!("Potential Energy: {:.2}", phys.potential_energy));
                ui.text(format!("Total Energy: {:.2}", phys.total_mechanical_energy));
                ui.text(format!("Work: {:.2}", phys.work));
                ui.text(format!("Power: {:.2}", phys.power));

                Self::display_vector3(ui, "Torque", &phys.torque);
            });
        }
        Ok(())
    }

    fn display_vector3(ui: &Ui, label: &str, vec: &glm::Vec3) {
        ui.text(format!(
            "{}: ({:.2}, {:.2}, {:.2})",
            label, vec.x, vec.y, vec.z
        ));
    }

    fn render_simulation_controls(&mut self, ui: &Ui) -> Result<(), String> {
        ui.window("Simulation Controls")
            .build(|| {
                let area = &mut self.area;

                ui.text(format!("Current Time: {:.3}", area.current_time()));

                Drag::new("Set Time")
                    .range(0.0, 1000.0)
                    .speed(0.1)
                    .build(ui, &mut self.time);

                if ui.button("Apply Time") {
                    area.update_time(self.time)
                        .map_err(|e| format!("Failed to update time: {}", e))?;
                }

                if area.is_simulating() {
                    if ui.button("Pause") {
                        area.pause_simulation();
                    }
                } else if ui.button("Start") {
                    area.start_simulation();
                }
                Ok(())
            })
            .unwrap_or(Ok(()))
    }

    fn render_create_body_window(&mut self, ui: &Ui) -> Result<(), String> {
        if !self.show_create_window {
            return Ok(());
        }

        let mut should_close = false;
        let mut should_create = false;

        ui.window("Create Body")
            .opened(&mut self.show_create_window)
            .build(|| {
                self.name_body_buffer = self.new_body.render_body.name.clone();
                if ui
                    .input_text("Body Name", &mut self.name_body_buffer)
                    .callback(
                        imgui::InputTextCallback::EDIT,
                        MaxLengthEnforcer { max_length: 16 },
                    )
                    .enter_returns_true(true)
                    .build()
                {
                    self.new_body.render_body.name = self.name_body_buffer.clone();
                }

                let body_types = ["Sphere", "Rectangle"];
                let mut current_type = self.new_body.render_body.body_type as usize;
                if ui.combo("Body Type", &mut current_type, &body_types, |item| {
                    std::borrow::Cow::Borrowed(item)
                }) {
                    self.new_body.render_body.body_type = match current_type {
                        0 => BodyType::Sphere,
                        1 => BodyType::Rectangle,
                        _ => BodyType::Sphere,
                    };
                }

                let mut is_kinematic = self.new_body.physical_body.edit_params.is_kinematic;
                if ui.checkbox("Is Kinematic", &mut is_kinematic) {
                    self.new_body.physical_body.edit_params.is_kinematic = is_kinematic;
                }

                self.texture_buffer = self.new_body.render_body.texture_path.clone();
                if ui
                    .input_text("Texture Path", &mut self.texture_buffer)
                    .enter_returns_true(true)
                    .build()
                {
                    self.new_body.render_body.texture_path = self.texture_buffer.clone();
                }

                match self.new_body.render_body.body_type {
                    BodyType::Sphere => {
                        let mut radius = self.new_body.render_body.dimensions.x;
                        if Drag::new("Radius")
                            .range(0.1, 100.0)
                            .speed(0.1)
                            .build(ui, &mut radius)
                        {
                            radius = if radius <= 0.0 { 0.001 } else { radius };
                            self.new_body.render_body.dimensions =
                                glm::vec3(radius, radius, radius);
                        }
                    }
                    BodyType::Rectangle => {
                        let mut dims = [
                            self.new_body.render_body.dimensions.x,
                            self.new_body.render_body.dimensions.y,
                            self.new_body.render_body.dimensions.z,
                        ];
                        if Drag::new("Dimensions")
                            .range(0.1, 100.0)
                            .speed(0.1)
                            .build_array(ui, &mut dims)
                        {
                            dims[0] = if dims[0] <= 0.0 { 0.001 } else { dims[0] };
                            dims[1] = if dims[1] <= 0.0 { 0.001 } else { dims[1] };
                            dims[2] = if dims[2] <= 0.0 { 0.001 } else { dims[2] };
                            self.new_body.render_body.dimensions =
                                glm::vec3(dims[0], dims[1], dims[2]);
                        }
                    }
                }

                ui.separator();
                ui.text("Physical Parameters");

                let mut pos = [
                    self.new_body.physical_body.edit_params.position.x,
                    self.new_body.physical_body.edit_params.position.y,
                    self.new_body.physical_body.edit_params.position.z,
                ];
                if Drag::new("Position").speed(0.1).build_array(ui, &mut pos) {
                    self.new_body
                        .physical_body
                        .edit_params
                        .update_position(glm::vec3(pos[0], pos[1], pos[2]));
                }

                let min_y = match self.new_body.render_body.body_type {
                    BodyType::Sphere => self.new_body.render_body.dimensions.x,
                    BodyType::Rectangle => self.new_body.render_body.dimensions.y,
                };

                if pos[1] < min_y {
                    self.new_body
                        .physical_body
                        .edit_params
                        .update_position(glm::vec3(pos[0], min_y, pos[2]));
                }

                let mut vel = [
                    self.new_body.physical_body.edit_params.velocity.x,
                    self.new_body.physical_body.edit_params.velocity.y,
                    self.new_body.physical_body.edit_params.velocity.z,
                ];
                if Drag::new("Velocity").speed(0.1).build_array(ui, &mut vel) {
                    self.new_body
                        .physical_body
                        .edit_params
                        .update_velocity(glm::vec3(vel[0], vel[1], vel[2]));
                }

                let mut mass = self.new_body.physical_body.edit_params.mass;
                if Drag::new("Mass")
                    .range(0.001, 1000.0)
                    .speed(0.1)
                    .build(ui, &mut mass)
                {
                    self.new_body.physical_body.edit_params.update_mass(mass);
                }

                let mut friction = self.new_body.physical_body.edit_params.friction;
                if Drag::new("Friction")
                    .range(0.001, 1000.0)
                    .speed(0.1)
                    .build(ui, &mut friction)
                {
                    self.new_body
                        .physical_body
                        .edit_params
                        .update_friction(friction);
                }

                let mut force = [
                    self.new_body.physical_body.edit_params.force.x,
                    self.new_body.physical_body.edit_params.force.y,
                    self.new_body.physical_body.edit_params.force.z,
                ];
                if Drag::new("Force").speed(0.1).build_array(ui, &mut force) {
                    self.new_body
                        .physical_body
                        .edit_params
                        .update_force(glm::vec3(force[0], force[1], force[2]));
                }

                ui.separator();

                if ui.button("Create")
                    && !self
                        .area
                        .name_bodies
                        .contains(&self.new_body.render_body.name)
                {
                    self.new_body.render_body.name = self.new_body.render_body.name.clone();
                    should_create = true;
                }

                ui.same_line();

                if ui.button("Cancel") {
                    should_close = true;
                }
            });

        if should_create {
            let area = &mut self.area;

            self.new_body.physical_body.apply_edit_to_runtime();

            if let Err(err) = area.add_body(self.new_body.clone()) {
                return Err(format!("Failed to create body: {}", err));
            }

            self.show_create_window = false;
        }

        if should_close {
            self.show_create_window = false;
        }

        Ok(())
    }

    fn render_coordinate_gizmo(&self, ui: &Ui, camera: &Camera) -> Result<(), String> {
        ui.window("Coordinate Gizmo")
            .size([140.0, 140.0], imgui::Condition::FirstUseEver)
            .flags(WindowFlags::NO_SCROLLBAR | WindowFlags::NO_TITLE_BAR)
            .build(|| {
                let draw_list = ui.get_window_draw_list();
                let window_pos = ui.window_pos();
                let content_region = ui.content_region_avail();
                let style = unsafe { ui.style() };
                let cx = window_pos[0] + style.window_padding[0] + content_region[0] * 0.5;
                let cy = window_pos[1] + style.window_padding[1] + content_region[1] * 0.5;
                let size = (content_region[0].min(content_region[1]) * 0.45).max(25.0);

                let view = camera.camera_data.view;
                let dirs = [
                    glm::vec3(view.m11, view.m21, view.m31),
                    glm::vec3(view.m12, view.m22, view.m32),
                    glm::vec3(view.m13, view.m23, view.m33),
                ];
                let colors = [0xFF4444FF, 0xFF44FF44, 0xFFFF4444];

                let mut ends = [(0.0f32, 0u8, false, glm::vec3(0.0, 0.0, 0.0), 0u32, None); 6];
                let labels = ["X", "Y", "Z"];

                for i in 0..3 {
                    let pos_dir = dirs[i];
                    let neg_dir = glm::vec3(-pos_dir.x, -pos_dir.y, -pos_dir.z);
                    ends[2 * i] = (
                        pos_dir.z,
                        i as u8,
                        true,
                        pos_dir,
                        colors[i],
                        Some(labels[i]),
                    );
                    ends[2 * i + 1] = (neg_dir.z, i as u8, false, neg_dir, colors[i], None);
                }

                ends.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                for (_, axis_id, is_positive, dir, color, label) in ends {
                    let x = cx + dir.x * size;
                    let y = cy - dir.y * size;

                    if is_positive {
                        draw_list
                            .add_line([cx, cy], [x, y], color)
                            .thickness(3.0)
                            .build();
                    }

                    for i in 0..3 {
                        draw_list
                            .add_circle([x, y], (6 - i) as f32, color)
                            .thickness(1.0)
                            .build();
                    }

                    if let Some(text) = label {
                        let offset = match axis_id {
                            0 => [6.0, -6.0],
                            1 => [-6.0, -18.0],
                            _ => [-12.0, -6.0],
                        };
                        draw_list.add_text([x + offset[0], y + offset[1]], 0xFFFFFFFF, text);
                    }
                }

                draw_list
                    .add_circle([cx, cy], 4.0, 0xFF888888)
                    .thickness(0.0)
                    .build();
            });

        Ok(())
    }
}
