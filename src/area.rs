use crate::body::{Body as AvaBody, BodyType};
use nalgebra_glm as glm;
use rapier3d::prelude::*;
use std::collections::HashSet;

const FIXED_TIMESTEP: f32 = 0.005;

pub struct Area {
    bodies: Vec<AvaBody>,
    pub name_bodies: HashSet<String>,
    current_time: f32,
    current_gravity: f32,
    current_friction: f32,
    current_restitution: f32,
    accumulator: f32,
    is_simulating: bool,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    body_handles: Vec<RigidBodyHandle>,
    ground_handle: ColliderHandle,
    island_manager: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
}

impl Default for Area {
    fn default() -> Self {
        let integration_parameters = IntegrationParameters {
            dt: FIXED_TIMESTEP,
            ..Default::default()
        };

        let mut collider_set = ColliderSet::new();
        let ground_shape = SharedShape::halfspace(Vec3::new(0.0, 1.0, 0.0));
        let ground_collider = ColliderBuilder::new(ground_shape)
            .translation(Vec3::new(0.0, 0.0, 0.0))
            .friction(0.2)
            .restitution(0.0);
        let ground_handle = collider_set.insert(ground_collider);

        Self {
            bodies: Vec::new(),
            name_bodies: HashSet::new(),
            current_time: 0.0,
            current_gravity: -9.81,
            current_friction: 0.2,
            current_restitution: 0.0,
            accumulator: 0.0,
            is_simulating: false,
            rigid_body_set: RigidBodySet::new(),
            collider_set,
            physics_pipeline: PhysicsPipeline::new(),
            integration_parameters,
            body_handles: Vec::new(),
            ground_handle,
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
        }
    }
}

impl Area {
    pub fn new() -> Result<Self, String> {
        Ok(Self::default())
    }

    fn create_rigid_body_builder(phys: &crate::body::PhysicalBody) -> RigidBodyBuilder {
        if phys.is_kinematic {
            RigidBodyBuilder::kinematic_position_based()
        } else {
            RigidBodyBuilder::dynamic()
                .gravity_scale(1.0)
                .ccd_enabled(true)
        }
    }

    fn create_collider_builder(
        shape: SharedShape,
        friction: f32,
        restitution: f32,
        mass: f32,
    ) -> ColliderBuilder {
        ColliderBuilder::new(shape)
            .mass(mass)
            .friction(friction)
            .restitution(restitution)
    }

    pub fn add_body(&mut self, body: AvaBody) -> Result<(), String> {
        self.validate_body(&body)?;

        let rb_handle = self.create_rigid_body_from_body(&body)?;
        self.create_collider_for_body(&body, rb_handle)?;
        self.set_body_velocity(&body, rb_handle);

        self.name_bodies.insert(body.render_body.name.clone());
        self.bodies.push(body);
        self.body_handles.push(rb_handle);

        self.update_time(0.0)?;
        Ok(())
    }

    fn create_shape_for_body(render_body: &crate::body::RenderBody) -> Result<SharedShape, String> {
        match render_body.body_type {
            BodyType::Sphere => {
                let radius = render_body.dimensions.x;
                Ok(SharedShape::ball(radius))
            }
            BodyType::Rectangle => {
                let half = render_body.dimensions;
                Ok(SharedShape::cuboid(half.x, half.y, half.z))
            }
        }
    }

    pub fn update_physics(&mut self) -> Result<(), String> {
        for (i, body) in self.bodies.iter().enumerate() {
            if let Some(rb_handle) = self.body_handles.get(i)
                && let Some(rb) = self.rigid_body_set.get_mut(*rb_handle)
                && !body.physical_body.is_kinematic
            {
                let force = body.physical_body.force;
                let external_force = Vec3::new(force.x, force.y, force.z);
                rb.reset_forces(true);
                rb.add_force(external_force, true);
            }
        }

        self.physics_pipeline.step(
            Vec3::new(0.0, self.current_gravity, 0.0),
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );

        self.sync_physics_to_bodies()?;
        self.current_time += FIXED_TIMESTEP;
        Ok(())
    }

    fn sync_physics_to_bodies(&mut self) -> Result<(), String> {
        let current_gravity = self.current_gravity;

        for (i, body) in self.bodies.iter_mut().enumerate() {
            if let Some(rb_handle) = self.body_handles.get(i)
                && let Some(rb) = self.rigid_body_set.get(*rb_handle)
            {
                Self::update_body_transform(body, rb);
                Self::calculate_body_physics(body, self.current_time, current_gravity);
                Self::calculate_body_forces(body, current_gravity);
            }
        }
        Ok(())
    }

    pub fn update_render(&mut self) {
        for body in &mut self.bodies {
            body.render_body.update(&body.physical_body);
        }
    }

    pub fn update_simulation(&mut self, delta_time: f32) -> Result<(), String> {
        if self.is_simulating {
            let clamped_dt = delta_time.min(0.1);
            self.accumulator += clamped_dt;

            while self.accumulator >= FIXED_TIMESTEP {
                self.update_physics()?;
                self.accumulator -= FIXED_TIMESTEP;
            }
        }
        self.update_render();
        Ok(())
    }

    pub fn start_simulation(&mut self) {
        self.is_simulating = true;
    }

    pub fn pause_simulation(&mut self) {
        self.is_simulating = false;
    }

    pub fn update_current_gravity(&mut self, gravity: f32) {
        self.current_gravity = gravity;
    }

    pub fn update_current_friction(&mut self, friction: f32) {
        self.current_friction = friction;
        if let Some(collider) = self.collider_set.get_mut(self.ground_handle) {
            collider.set_friction(friction);
        }
    }

    pub fn update_current_restitution(&mut self, restitution: f32) {
        self.current_restitution = restitution;
        if let Some(collider) = self.collider_set.get_mut(self.ground_handle) {
            collider.set_restitution(restitution);
        }
    }

    pub fn update_time(&mut self, time: f32) -> Result<(), String> {
        let target_time = time.max(0.0);
        self.reset_bodies_to_initial_state()?;
        self.simulate_to_time(target_time)?;
        self.current_time = target_time;
        Ok(())
    }

    pub fn reset_simulation(&mut self) -> Result<(), String> {
        self.reset_bodies_to_initial_state()?;
        Ok(())
    }

    pub fn update_body(&mut self, index: usize) -> Result<(), String> {
        self.validate_body_index(index)?;

        let old_handle = self.body_handles[index];
        self.remove_rigid_body(old_handle);

        let (rb_builder, shape, friction, restitution, mass, pos, vel) = {
            let body = &self.bodies[index];
            let rb_builder = Self::create_rigid_body_builder(&body.physical_body);
            let shape = Self::create_shape_for_body(&body.render_body)?;
            let friction = body.physical_body.edit_params.friction;
            let restitution = if body.physical_body.edit_params.is_restitution {
                1.0_f32
            } else {
                0.0_f32
            };
            let mass = body.physical_body.mass;
            let pos = body.physical_body.position;
            let vel = body.physical_body.velocity;
            (rb_builder, shape, friction, restitution, mass, pos, vel)
        };

        let rb_handle = self
            .rigid_body_set
            .insert(rb_builder.translation(Vec3::new(pos.x, pos.y, pos.z)));
        let collider_builder = Self::create_collider_builder(shape, friction, restitution, mass);
        self.collider_set
            .insert_with_parent(collider_builder, rb_handle, &mut self.rigid_body_set);

        if let Some(rb) = self.rigid_body_set.get_mut(rb_handle) {
            rb.set_linvel(Vec3::new(vel.x, vel.y, vel.z), true);
        }

        self.body_handles[index] = rb_handle;
        Ok(())
    }

    pub fn remove_body(&mut self, index: usize) -> Result<(), String> {
        self.validate_body_index(index)?;

        let handle = self.body_handles.remove(index);
        self.remove_rigid_body(handle);

        let name = self.bodies.remove(index).render_body.name;
        self.name_bodies.remove(&name);

        self.update_time(0.0)?;
        Ok(())
    }

    pub fn current_gravity(&self) -> f32 {
        self.current_gravity
    }

    pub fn current_friction(&self) -> f32 {
        self.current_friction
    }

    pub fn current_restitution(&self) -> f32 {
        self.current_restitution
    }

    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    pub fn is_simulating(&self) -> bool {
        self.is_simulating
    }

    pub fn count_bodies(&self) -> usize {
        self.bodies.len()
    }

    pub fn body(&self, index: usize) -> Option<&AvaBody> {
        self.bodies.get(index)
    }

    pub fn body_mut(&mut self, index: usize) -> Option<&mut AvaBody> {
        self.bodies.get_mut(index)
    }

    fn validate_body(&self, body: &AvaBody) -> Result<(), String> {
        if body.physical_body.mass <= 0.0 {
            return Err("Mass must be positive".to_string());
        }
        if body.render_body.texture_path.is_empty() {
            return Err("Texture path is empty".to_string());
        }
        Ok(())
    }

    fn validate_body_index(&self, index: usize) -> Result<(), String> {
        if index >= self.bodies.len() || index >= self.body_handles.len() {
            return Err(format!("Invalid body index: {}", index));
        }
        Ok(())
    }

    fn create_rigid_body_from_body(&mut self, body: &AvaBody) -> Result<RigidBodyHandle, String> {
        let rb_builder = Self::create_rigid_body_builder(&body.physical_body);
        let pos = body.physical_body.position;
        Ok(self
            .rigid_body_set
            .insert(rb_builder.translation(Vec3::new(pos.x, pos.y, pos.z))))
    }

    fn create_collider_for_body(
        &mut self,
        body: &AvaBody,
        rb_handle: RigidBodyHandle,
    ) -> Result<(), String> {
        let shape = Self::create_shape_for_body(&body.render_body)?;
        let friction = body.physical_body.edit_params.friction;
        let restitution = if body.physical_body.edit_params.is_restitution {
            1.0_f32
        } else {
            0.0_f32
        };
        let mass = body.physical_body.mass;

        let collider_builder = Self::create_collider_builder(shape, friction, restitution, mass);
        self.collider_set
            .insert_with_parent(collider_builder, rb_handle, &mut self.rigid_body_set);
        Ok(())
    }

    fn set_body_velocity(&mut self, body: &AvaBody, rb_handle: RigidBodyHandle) {
        if let Some(rb) = self.rigid_body_set.get_mut(rb_handle) {
            let vel = body.physical_body.velocity;
            rb.set_linvel(Vec3::new(vel.x, vel.y, vel.z), true);
        }
    }

    fn remove_rigid_body(&mut self, handle: RigidBodyHandle) {
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }

    fn update_body_transform(body: &mut AvaBody, rb: &RigidBody) {
        let pos = rb.translation();
        let rot = rb.rotation();
        let vel = rb.linvel();

        body.physical_body.position = glm::vec3(pos.x, pos.y, pos.z);
        body.physical_body.velocity = glm::vec3(vel.x, vel.y, vel.z);
        body.render_body.rotation = glm::quat(rot.x, rot.y, rot.z, rot.w);
    }

    fn calculate_body_physics(body: &mut AvaBody, current_time: f32, current_gravity: f32) {
        let prev_vel = body.physical_body.edit_params.velocity;
        body.physical_body.acceleration = (body.physical_body.velocity - prev_vel) / current_time;

        body.physical_body.momentum = body.physical_body.mass * body.physical_body.velocity;
        body.physical_body.kinetic_energy = 0.5
            * body.physical_body.mass
            * glm::dot(&body.physical_body.velocity, &body.physical_body.velocity);
        body.physical_body.potential_energy =
            body.physical_body.mass * current_gravity * body.physical_body.position.y;
        body.physical_body.total_mechanical_energy =
            body.physical_body.kinetic_energy + body.physical_body.potential_energy;

        let distance_vec = body.physical_body.velocity * FIXED_TIMESTEP;
        body.physical_body.displacement =
            body.physical_body.position - body.physical_body.edit_params.position;
        body.physical_body.distance += glm::length(&distance_vec);
    }

    fn calculate_body_forces(body: &mut AvaBody, current_gravity: f32) {
        body.physical_body.gravity_force =
            glm::vec3(0.0, body.physical_body.mass * current_gravity, 0.0);
        let total_force = body.physical_body.force + body.physical_body.gravity_force;
        body.physical_body.net_force = glm::length(&total_force);

        let work = glm::dot(
            &total_force,
            &(body.physical_body.velocity * FIXED_TIMESTEP),
        );
        body.physical_body.work += work;
        body.physical_body.power = work / FIXED_TIMESTEP;
        body.physical_body.impulse = total_force * FIXED_TIMESTEP;

        if body.physical_body.is_kinematic || body.physical_body.position.y <= 0.0 {
            body.physical_body.normal_force =
                glm::vec3(0.0, -body.physical_body.gravity_force.y, 0.0);
        } else {
            body.physical_body.normal_force = glm::vec3(0.0, 0.0, 0.0);
        }

        body.physical_body.friction_force = glm::vec3(0.0, 0.0, 0.0);
        body.physical_body.elastic_force = glm::vec3(0.0, 0.0, 0.0);
    }

    fn reset_bodies_to_initial_state(&mut self) -> Result<(), String> {
        self.current_time = 0.0;
        self.is_simulating = false;

        for i in 0..self.bodies.len() {
            self.bodies[i].physical_body.apply_edit_to_runtime();
            self.update_body(i)?;
            self.bodies[i].render_body.rotation = glm::quat_identity();
        }
        Ok(())
    }

    fn simulate_to_time(&mut self, target_time: f32) -> Result<(), String> {
        let mut simulated_time = 0.0;
        while simulated_time < target_time {
            self.update_physics()?;
            simulated_time += FIXED_TIMESTEP;
        }
        Ok(())
    }
}
