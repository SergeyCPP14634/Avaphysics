use nalgebra_glm as glm;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    Sphere,
    Rectangle,
}

#[derive(Clone)]
pub struct EditParameters {
    pub position: glm::Vec3,
    pub velocity: glm::Vec3,
    pub mass: f32,
    pub friction: f32,
    pub force: glm::Vec3,
    pub is_kinematic: bool,

    pub acceleration: glm::Vec3,
    pub momentum: glm::Vec3,
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_mechanical_energy: f32,
    pub net_force: f32,
    pub gravity_force: glm::Vec3,

    pub current_gravity: f32,

    pub angular_velocity: f32,
    pub angular_acceleration: f32,
    pub torque: glm::Vec3,

    pub displacement: glm::Vec3,
    pub distance: f32,
    pub friction_force: glm::Vec3,
    pub normal_force: glm::Vec3,
    pub elastic_force: glm::Vec3,
    pub impulse: glm::Vec3,
    pub work: f32,
    pub power: f32,
}

impl Default for EditParameters {
    fn default() -> Self {
        let mut params = Self {
            position: glm::vec3(0.0, 0.0, 0.0),
            velocity: glm::vec3(0.0, 0.0, 0.0),
            mass: 1.0,
            friction: 0.2,
            force: glm::vec3(0.0, 0.0, 0.0),
            is_kinematic: false,

            acceleration: glm::vec3(0.0, 0.0, 0.0),
            momentum: glm::vec3(0.0, 0.0, 0.0),
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_mechanical_energy: 0.0,
            net_force: 0.0,
            gravity_force: glm::vec3(0.0, -9.81, 0.0),

            current_gravity: -9.81,

            angular_velocity: 0.0,
            angular_acceleration: 0.0,
            torque: glm::vec3(0.0, 0.0, 0.0),

            displacement: glm::vec3(0.0, 0.0, 0.0),
            distance: 0.0,
            friction_force: glm::vec3(0.0, 0.0, 0.0),
            normal_force: glm::vec3(0.0, 0.0, 0.0),
            elastic_force: glm::vec3(0.0, 0.0, 0.0),
            impulse: glm::vec3(0.0, 0.0, 0.0),
            work: 0.0,
            power: 0.0,
        };
        params.recompute_instantaneous();
        params
    }
}

impl EditParameters {
    fn recompute_instantaneous(&mut self) {
        let mass = self.mass.max(0.001);
        self.mass = mass;

        self.gravity_force = glm::vec3(0.0, mass * self.current_gravity, 0.0);

        if self.is_kinematic {
            self.acceleration = glm::vec3(0.0, 0.0, 0.0);
            self.net_force = 0.0;
        } else {
            let total_force = self.force + self.gravity_force;
            self.net_force = glm::length(&total_force);
            self.acceleration = total_force / mass;
        }

        self.momentum = mass * self.velocity;
        self.kinetic_energy = 0.5 * mass * glm::dot(&self.velocity, &self.velocity);
        self.potential_energy = mass * self.current_gravity * self.position.y;
        self.total_mechanical_energy = self.kinetic_energy + self.potential_energy;
    }

    pub fn update_position(&mut self, position: glm::Vec3) {
        self.position = position;
        self.recompute_instantaneous();
    }

    pub fn update_velocity(&mut self, velocity: glm::Vec3) {
        self.velocity = velocity;
        self.recompute_instantaneous();
    }

    pub fn update_mass(&mut self, mass: f32) {
        self.mass = mass;
        self.recompute_instantaneous();
    }

    pub fn update_friction(&mut self, friction: f32) {
        self.friction = friction;
        self.recompute_instantaneous();
    }

    pub fn update_force(&mut self, force: glm::Vec3) {
        self.force = force;
        self.recompute_instantaneous();
    }

    pub fn update_is_kinematic(&mut self, is_kinematic: bool) {
        self.is_kinematic = is_kinematic;
        self.recompute_instantaneous();
    }

    pub fn update_current_gravity(&mut self, g: f32) {
        self.current_gravity = g;
        self.recompute_instantaneous();
    }
}

#[derive(Clone)]
pub struct PhysicalBody {
    pub position: glm::Vec3,
    pub displacement: glm::Vec3,
    pub velocity: glm::Vec3,
    pub acceleration: glm::Vec3,
    pub distance: f32,
    pub angular_velocity: f32,
    pub angular_acceleration: f32,
    pub mass: f32,
    pub friction: f32,
    pub force: glm::Vec3,
    pub net_force: f32,
    pub gravity_force: glm::Vec3,
    pub friction_force: glm::Vec3,
    pub normal_force: glm::Vec3,
    pub elastic_force: glm::Vec3,
    pub momentum: glm::Vec3,
    pub impulse: glm::Vec3,
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_mechanical_energy: f32,
    pub work: f32,
    pub power: f32,
    pub torque: glm::Vec3,
    pub is_kinematic: bool,
    pub current_gravity: f32,
    pub edit_params: EditParameters,
}

impl Default for PhysicalBody {
    fn default() -> Self {
        Self {
            position: glm::vec3(0.0, 0.0, 0.0),
            displacement: glm::vec3(0.0, 0.0, 0.0),
            velocity: glm::vec3(0.0, 0.0, 0.0),
            acceleration: glm::vec3(0.0, 0.0, 0.0),
            distance: 0.0,
            angular_velocity: 0.0,
            angular_acceleration: 0.0,
            mass: 1.0,
            friction: 0.2,
            force: glm::vec3(0.0, 0.0, 0.0),
            net_force: 0.0,
            gravity_force: glm::vec3(0.0, 0.0, 0.0),
            friction_force: glm::vec3(0.0, 0.0, 0.0),
            normal_force: glm::vec3(0.0, 0.0, 0.0),
            elastic_force: glm::vec3(0.0, 0.0, 0.0),
            momentum: glm::vec3(0.0, 0.0, 0.0),
            impulse: glm::vec3(0.0, 0.0, 0.0),
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_mechanical_energy: 0.0,
            work: 0.0,
            power: 0.0,
            torque: glm::vec3(0.0, 0.0, 0.0),
            is_kinematic: false,
            current_gravity: -9.81,
            edit_params: EditParameters::default(),
        }
    }
}

impl PhysicalBody {
    pub fn new(
        position: glm::Vec3,
        displacement: glm::Vec3,
        velocity: glm::Vec3,
        acceleration: glm::Vec3,
        distance: f32,
        angular_velocity: f32,
        angular_acceleration: f32,
        mass: f32,
        friction: f32,
        force: glm::Vec3,
        net_force: f32,
        gravity_force: glm::Vec3,
        friction_force: glm::Vec3,
        normal_force: glm::Vec3,
        elastic_force: glm::Vec3,
        momentum: glm::Vec3,
        impulse: glm::Vec3,
        kinetic_energy: f32,
        potential_energy: f32,
        total_mechanical_energy: f32,
        work: f32,
        power: f32,
        torque: glm::Vec3,
        is_kinematic: bool,
        current_gravity: f32,
    ) -> Self {
        let mut body = Self {
            position,
            displacement,
            velocity,
            acceleration,
            distance,
            angular_velocity,
            angular_acceleration,
            mass,
            friction,
            force,
            net_force,
            gravity_force,
            friction_force,
            normal_force,
            elastic_force,
            momentum,
            impulse,
            kinetic_energy,
            potential_energy,
            total_mechanical_energy,
            work,
            power,
            torque,
            is_kinematic,
            current_gravity,
            edit_params: EditParameters::default(),
        };
        body.sync_edit_from_runtime();
        body
    }

    pub fn sync_edit_from_runtime(&mut self) {
        self.edit_params = EditParameters {
            position: self.position,
            displacement: self.displacement,
            velocity: self.velocity,
            acceleration: self.acceleration,
            distance: self.distance,
            angular_velocity: self.angular_velocity,
            angular_acceleration: self.angular_acceleration,
            mass: self.mass,
            friction: self.friction,
            force: self.force,
            net_force: self.net_force,
            gravity_force: self.gravity_force,
            friction_force: self.friction_force,
            normal_force: self.normal_force,
            elastic_force: self.elastic_force,
            momentum: self.momentum,
            impulse: self.impulse,
            kinetic_energy: self.kinetic_energy,
            potential_energy: self.potential_energy,
            total_mechanical_energy: self.total_mechanical_energy,
            work: self.work,
            power: self.power,
            torque: self.torque,
            is_kinematic: self.is_kinematic,
            current_gravity: self.current_gravity,
        };
    }

    pub fn apply_edit_to_runtime(&mut self) {
        let params = &self.edit_params;
        self.position = params.position;
        self.displacement = params.displacement;
        self.velocity = params.velocity;
        self.acceleration = params.acceleration;
        self.distance = params.distance;
        self.angular_velocity = params.angular_velocity;
        self.angular_acceleration = params.angular_acceleration;
        self.mass = params.mass;
        self.friction = params.friction;
        self.force = params.force;
        self.net_force = params.net_force;
        self.gravity_force = params.gravity_force;
        self.friction_force = params.friction_force;
        self.normal_force = params.normal_force;
        self.elastic_force = params.elastic_force;
        self.momentum = params.momentum;
        self.impulse = params.impulse;
        self.kinetic_energy = params.kinetic_energy;
        self.potential_energy = params.potential_energy;
        self.total_mechanical_energy = params.total_mechanical_energy;
        self.work = params.work;
        self.power = params.power;
        self.torque = params.torque;
        self.is_kinematic = params.is_kinematic;
        self.current_gravity = params.current_gravity;
    }
}

#[derive(Clone)]
pub struct Vertex {
    pub position: glm::Vec3,
    pub texture_position: glm::Vec3,
    pub normal: glm::Vec3,
    pub color: glm::Vec3,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: glm::vec3(0.0, 0.0, 0.0),
            texture_position: glm::vec3(0.0, 0.0, 0.0),
            normal: glm::vec3(0.0, 0.0, 0.0),
            color: glm::vec3(0.0, 0.0, 0.0),
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct RenderBodyData {
    pub model: glm::Mat4,
    pub position_render_body: glm::Vec4,
    pub shininess: f32,
}

impl Default for RenderBodyData {
    fn default() -> Self {
        Self {
            model: glm::Mat4::identity(),
            position_render_body: glm::vec4(0.0, 0.0, 0.0, 1.0),
            shininess: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct RenderBody {
    pub render_body_data: RenderBodyData,
    pub name: String,
    pub texture_path: String,
    pub body_type: BodyType,
    pub dimensions: glm::Vec3,
    pub rotation: glm::Quat,
    pub texture_is_update: bool,
}

impl Default for RenderBody {
    fn default() -> Self {
        Self {
            render_body_data: RenderBodyData::default(),
            name: String::new(),
            texture_path: String::new(),
            body_type: BodyType::Sphere,
            dimensions: glm::vec3(1.0, 1.0, 1.0),
            rotation: glm::quat_identity(),
            texture_is_update: false,
        }
    }
}

impl RenderBody {
    pub fn new(
        name: String,
        texture_path: String,
        body_type: BodyType,
        dimensions: glm::Vec3,
    ) -> Self {
        Self {
            name,
            texture_path,
            body_type,
            dimensions,
            ..Self::default()
        }
    }

    pub fn update(&mut self, physical_body: &PhysicalBody) {
        self.render_body_data.position_render_body = glm::vec4(
            physical_body.position.x,
            physical_body.position.y,
            physical_body.position.z,
            1.0,
        );

        let mut model = glm::translate(&glm::Mat4::identity(), &physical_body.position);

        model *= glm::quat_to_mat4(&self.rotation);

        match self.body_type {
            BodyType::Rectangle => {
                model = glm::scale(&model, &self.dimensions);
            }
            BodyType::Sphere => {
                let radius = self.dimensions.x;
                model = glm::scale(&model, &glm::vec3(radius, radius, radius));
            }
        }

        self.render_body_data.model = model;
    }
}

#[derive(Clone, Default)]
pub struct Body {
    pub physical_body: PhysicalBody,
    pub render_body: RenderBody,
}

impl Body {
    pub fn new(physical_body: PhysicalBody, render_body: RenderBody) -> Self {
        Self {
            physical_body,
            render_body,
        }
    }
}
