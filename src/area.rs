use crate::body::{Body as AvaBody, BodyType};
use nalgebra_glm as glm;
use rapier3d::prelude::*;
use std::collections::{HashMap, HashSet};

const FIXED_TIMESTEP: f32 = 0.005;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

pub struct Chunk {
    pub coord: ChunkCoord,
    pub collider_handle: ColliderHandle,
}

impl Chunk {
    const SIZE_X: f32 = 50.0;
    const SIZE_Y: f32 = 1.0;
    const SIZE_Z: f32 = 50.0;
    const THRESHOLD: f32 = 10.0;
    const OVERLAP: f32 = 0.05;

    fn new(coord: ChunkCoord, collider_handle: ColliderHandle) -> Self {
        Self {
            coord,
            collider_handle,
        }
    }

    fn chunk_coord(pos: glm::Vec3) -> ChunkCoord {
        ChunkCoord {
            x: (pos.x / Self::SIZE_X).floor() as i32,
            z: (pos.z / Self::SIZE_Z).floor() as i32,
        }
    }

    fn position(coord: &ChunkCoord) -> glm::Vec3 {
        let step_x = Self::SIZE_X - 2.0 * Self::OVERLAP;
        let step_z = Self::SIZE_Z - 2.0 * Self::OVERLAP;
        glm::vec3(
            coord.x as f32 * step_x + Self::SIZE_X * 0.5,
            -Self::SIZE_Y * 0.5,
            coord.z as f32 * step_z + Self::SIZE_Z * 0.5,
        )
    }
}

pub struct Area {
    bodies: Vec<AvaBody>,
    pub name_bodies: HashSet<String>,
    current_time: f32,
    pub current_gravity: f32,
    pub current_friction: f32,
    accumulator: f32,
    is_simulating: bool,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    body_handles: Vec<RigidBodyHandle>,
    chunks: HashMap<ChunkCoord, Chunk>,
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

        Self {
            bodies: Vec::new(),
            name_bodies: HashSet::new(),
            current_time: 0.0,
            current_gravity: -9.81,
            current_friction: 0.2,
            accumulator: 0.0,
            is_simulating: false,
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            integration_parameters,
            body_handles: Vec::new(),
            chunks: HashMap::new(),
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

    fn create_collider_builder(shape: SharedShape, friction: f32, mass: f32) -> ColliderBuilder {
        let volume = if let Some(ball) = shape.as_ball() {
            let r = ball.radius;
            (4.0 / 3.0) * std::f32::consts::PI * r * r * r
        } else if let Some(cuboid) = shape.as_cuboid() {
            let half_extents = cuboid.half_extents;
            8.0 * half_extents.x * half_extents.y * half_extents.z
        } else {
            1.0
        };

        let density = mass / volume;

        ColliderBuilder::new(shape)
            .density(density)
            .friction(friction)
            .restitution(0.0)
    }

    fn add_chunk_neighbors(
        needed_coords: &mut HashSet<ChunkCoord>,
        chunk_coord: ChunkCoord,
        local_pos: f32,
        axis_threshold: f32,
        is_x_axis: bool,
    ) {
        if local_pos > axis_threshold - Chunk::THRESHOLD {
            let neighbor = if is_x_axis {
                ChunkCoord {
                    x: chunk_coord.x + 1,
                    z: chunk_coord.z,
                }
            } else {
                ChunkCoord {
                    x: chunk_coord.x,
                    z: chunk_coord.z + 1,
                }
            };
            needed_coords.insert(neighbor);
        }
        if local_pos < Chunk::THRESHOLD {
            let neighbor = if is_x_axis {
                ChunkCoord {
                    x: chunk_coord.x - 1,
                    z: chunk_coord.z,
                }
            } else {
                ChunkCoord {
                    x: chunk_coord.x,
                    z: chunk_coord.z - 1,
                }
            };
            needed_coords.insert(neighbor);
        }
    }

    pub fn add_body(&mut self, body: AvaBody) -> Result<(), String> {
        let phys = &body.physical_body;
        let rend = &body.render_body;

        if phys.mass <= 0.0 {
            return Err("Mass must be positive".to_string());
        }
        if rend.texture_path.is_empty() {
            return Err("Texture path is empty".to_string());
        }

        let shape = Self::create_shape_for_body(rend)?;
        let friction = phys.edit_params.friction;
        let rb_builder = Self::create_rigid_body_builder(phys);
        let pos = phys.position;
        let vel = phys.velocity;
        let mass = phys.mass;

        self.name_bodies.insert(rend.name.clone());
        self.bodies.push(body);

        let rb_handle = self
            .rigid_body_set
            .insert(rb_builder.translation(Vec3::new(pos.x, pos.y, pos.z)));

        let collider_builder = Self::create_collider_builder(shape, friction, mass);
        self.collider_set
            .insert_with_parent(collider_builder, rb_handle, &mut self.rigid_body_set);
        self.body_handles.push(rb_handle);

        if let Some(rb) = self.rigid_body_set.get_mut(rb_handle) {
            rb.set_linvel(Vec3::new(vel.x, vel.y, vel.z), true);
        }

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

    fn create_chunk_collider(
        collider_set: &mut ColliderSet,
        rigid_body_set: &mut RigidBodySet,
        coord: &ChunkCoord,
        friction: f32,
    ) -> Result<ColliderHandle, String> {
        let position = Chunk::position(coord);
        let ground_shape = SharedShape::cuboid(
            Chunk::SIZE_X * 0.5 + Chunk::OVERLAP,
            Chunk::SIZE_Y * 0.5,
            Chunk::SIZE_Z * 0.5 + Chunk::OVERLAP,
        );

        let ground_rb =
            RigidBodyBuilder::fixed().translation(Vec3::new(position.x, position.y, position.z));

        let rb_handle = rigid_body_set.insert(ground_rb);
        let collider = Self::create_collider_builder(ground_shape, friction, 0.0);
        let handle = collider_set.insert_with_parent(collider, rb_handle, rigid_body_set);
        Ok(handle)
    }

    fn update_chunks(&mut self) -> Result<(), String> {
        if self.bodies.is_empty() {
            return Ok(());
        }

        let mut needed_coords = HashSet::new();

        for body in &self.bodies {
            let pos = body.physical_body.position;
            let chunk_coord = Chunk::chunk_coord(pos);
            needed_coords.insert(chunk_coord);

            let local_x = pos.x - (chunk_coord.x as f32 * (Chunk::SIZE_X - 2.0 * Chunk::OVERLAP));
            let local_z = pos.z - (chunk_coord.z as f32 * (Chunk::SIZE_Z - 2.0 * Chunk::OVERLAP));

            Self::add_chunk_neighbors(
                &mut needed_coords,
                chunk_coord,
                local_x,
                Chunk::SIZE_X,
                true,
            );
            Self::add_chunk_neighbors(
                &mut needed_coords,
                chunk_coord,
                local_z,
                Chunk::SIZE_Z,
                false,
            );
        }

        let to_remove: Vec<_> = self
            .chunks
            .keys()
            .filter(|c| !needed_coords.contains(c))
            .cloned()
            .collect();
        for coord in to_remove {
            if let Some(chunk) = self.chunks.remove(&coord) {
                self.collider_set.remove(
                    chunk.collider_handle,
                    &mut self.island_manager,
                    &mut self.rigid_body_set,
                    true,
                );
            }
        }

        for coord in needed_coords {
            if !self.chunks.contains_key(&coord) {
                let handle = Self::create_chunk_collider(
                    &mut self.collider_set,
                    &mut self.rigid_body_set,
                    &coord,
                    self.current_friction,
                )?;
                self.chunks.insert(coord, Chunk::new(coord, handle));
            }
        }

        Ok(())
    }

    pub fn update_physics(&mut self) -> Result<(), String> {
        self.update_chunks()?;

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
        for (i, body) in self.bodies.iter_mut().enumerate() {
            if let Some(rb_handle) = self.body_handles.get(i)
                && let Some(rb) = self.rigid_body_set.get(*rb_handle)
            {
                let pos = rb.translation();
                let rot = rb.rotation();
                let vel = rb.linvel();

                body.physical_body.position = glm::vec3(pos.x, pos.y, pos.z);
                body.physical_body.velocity = glm::vec3(vel.x, vel.y, vel.z);
                body.render_body.rotation = glm::quat(rot.x, rot.y, rot.z, rot.w);

                let prev_vel = body.physical_body.edit_params.velocity;
                let dt = if self.current_time > 1e-6 {
                    self.current_time
                } else {
                    FIXED_TIMESTEP
                };
                body.physical_body.acceleration = (body.physical_body.velocity - prev_vel) / dt;

                body.physical_body.gravity_force =
                    glm::vec3(0.0, body.physical_body.mass * self.current_gravity, 0.0);
                let total_force = body.physical_body.force + body.physical_body.gravity_force;
                body.physical_body.net_force = glm::length(&total_force);

                body.physical_body.momentum = body.physical_body.mass * body.physical_body.velocity;
                body.physical_body.kinetic_energy = 0.5
                    * body.physical_body.mass
                    * glm::dot(&body.physical_body.velocity, &body.physical_body.velocity);
                body.physical_body.potential_energy =
                    body.physical_body.mass * self.current_gravity * body.physical_body.position.y;
                body.physical_body.total_mechanical_energy =
                    body.physical_body.kinetic_energy + body.physical_body.potential_energy;

                let distance_vec = body.physical_body.velocity * FIXED_TIMESTEP;
                body.physical_body.displacement =
                    body.physical_body.position - body.physical_body.edit_params.position;
                body.physical_body.distance += glm::length(&distance_vec);

                let work = glm::dot(&total_force, &distance_vec);
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

    pub fn update_time(&mut self, time: f32) -> Result<(), String> {
        let target_time = time.max(0.0);
        self.current_time = 0.0;
        self.is_simulating = false;

        for chunk in self.chunks.values() {
            self.collider_set.remove(
                chunk.collider_handle,
                &mut self.island_manager,
                &mut self.rigid_body_set,
                true,
            );
        }
        self.chunks.clear();

        for i in 0..self.bodies.len() {
            self.bodies[i].physical_body.apply_edit_to_runtime();
            self.update_body(i)?;
            self.bodies[i].render_body.rotation = glm::quat_identity();
        }

        let mut simulated_time = 0.0;
        while simulated_time < target_time {
            self.update_physics()?;
            simulated_time += FIXED_TIMESTEP;
        }

        self.current_time = target_time;
        Ok(())
    }

    pub fn reset_simulation(&mut self) -> Result<(), String> {
        self.current_time = 0.0;
        self.is_simulating = false;
        for i in 0..self.bodies.len() {
            self.bodies[i].physical_body.apply_edit_to_runtime();
            self.update_body(i)?;
            self.bodies[i].render_body.rotation = glm::quat_identity();
        }
        Ok(())
    }

    pub fn update_body(&mut self, index: usize) -> Result<(), String> {
        if index >= self.bodies.len() || index >= self.body_handles.len() {
            return Err(format!("Invalid body index: {}", index));
        }

        let old_handle = self.body_handles[index];
        self.rigid_body_set.remove(
            old_handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );

        let body = &self.bodies[index];
        let phys = &body.physical_body;
        let rend = &body.render_body;

        let shape = Self::create_shape_for_body(rend)?;
        let friction = phys.edit_params.friction;

        let rb_builder = Self::create_rigid_body_builder(phys);
        let pos = phys.position;
        let rb_handle = self
            .rigid_body_set
            .insert(rb_builder.translation(Vec3::new(pos.x, pos.y, pos.z)));

        let collider_builder = Self::create_collider_builder(shape, friction, phys.mass);
        self.collider_set
            .insert_with_parent(collider_builder, rb_handle, &mut self.rigid_body_set);

        if let Some(rb) = self.rigid_body_set.get_mut(rb_handle) {
            let vel = phys.velocity;
            rb.set_linvel(Vec3::new(vel.x, vel.y, vel.z), true);
        }

        self.body_handles[index] = rb_handle;
        Ok(())
    }

    pub fn remove_body(&mut self, index: usize) -> Result<(), String> {
        if index >= self.bodies.len() || index >= self.body_handles.len() {
            return Err(format!("Invalid body index: {}", index));
        }

        let handle = self.body_handles.remove(index);
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );

        let name = self.bodies.remove(index).render_body.name;
        self.name_bodies.remove(&name);
        self.update_time(0.0)?;
        Ok(())
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
}
