use crate::body::{Body as AvaBody, BodyType};
use joltc_sys::*;
use nalgebra_glm as glm;
use rolt::{
    BodyId, BroadPhaseLayer, BroadPhaseLayerInterface, FromJolt, IntoJolt, ObjectLayer,
    ObjectLayerPairFilter, ObjectVsBroadPhaseLayerFilter, PhysicsSystem, Quat, RVec3, Vec3,
    factory_delete, factory_init, register_default_allocator, register_types, unregister_types,
};
use std::collections::HashMap;
use std::collections::HashSet;
use std::ptr;

const OL_NON_MOVING: u16 = 0;
const OL_MOVING: u16 = 1;

const BPL_NON_MOVING: u8 = 0;
const BPL_MOVING: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

pub struct Chunk {
    pub coord: ChunkCoord,
    pub body_id: BodyId,
}

impl Chunk {
    const SIZE_X: f32 = 100.0;
    const SIZE_Y: f32 = 10.0;
    const SIZE_Z: f32 = 100.0;
    const THRESHOLD: f32 = 10.0;

    fn new(coord: ChunkCoord, body_id: BodyId) -> Self {
        Self { coord, body_id }
    }

    fn chunk_coord(pos: glm::Vec3) -> ChunkCoord {
        ChunkCoord {
            x: (pos.x / Self::SIZE_X).floor() as i32,
            z: (pos.z / Self::SIZE_Z).floor() as i32,
        }
    }

    fn position(coord: &ChunkCoord) -> glm::Vec3 {
        glm::vec3(
            coord.x as f32 * Self::SIZE_X + Self::SIZE_X * 0.5,
            -Self::SIZE_Y * 0.5,
            coord.z as f32 * Self::SIZE_Z + Self::SIZE_Z * 0.5,
        )
    }
}

struct MyBroadPhaseLayerInterface;

impl BroadPhaseLayerInterface for MyBroadPhaseLayerInterface {
    fn get_num_broad_phase_layers(&self) -> u32 {
        2
    }

    fn get_broad_phase_layer(&self, layer: ObjectLayer) -> BroadPhaseLayer {
        match layer.raw() {
            OL_NON_MOVING => BroadPhaseLayer::new(BPL_NON_MOVING),
            OL_MOVING => BroadPhaseLayer::new(BPL_MOVING),
            _ => unreachable!("Invalid object layer"),
        }
    }
}

struct MyObjectVsBroadPhase;

impl ObjectVsBroadPhaseLayerFilter for MyObjectVsBroadPhase {
    fn should_collide(&self, _layer: ObjectLayer, _broad_phase: BroadPhaseLayer) -> bool {
        true
    }
}

struct MyObjectLayerPair;

impl ObjectLayerPairFilter for MyObjectLayerPair {
    fn should_collide(&self, _layer1: ObjectLayer, _layer2: ObjectLayer) -> bool {
        true
    }
}

pub struct Area {
    bodies: Vec<AvaBody>,
    pub name_bodies: HashSet<String>,
    current_time: f32,
    pub current_gravity: f32,
    pub current_friction: f32,
    is_simulating: bool,
    physics_system: PhysicsSystem,
    temp_allocator: *mut JPC_TempAllocatorImpl,
    job_system: *mut JPC_JobSystemThreadPool,
    rolt_body_ids: Vec<BodyId>,
    chunks: HashMap<ChunkCoord, Chunk>,
}

impl Area {
    pub fn new() -> Result<Self, String> {
        register_default_allocator();
        factory_init();
        register_types();

        let mut physics_system = PhysicsSystem::new();
        physics_system.init(
            1024,
            0,
            256,
            1024,
            MyBroadPhaseLayerInterface,
            MyObjectVsBroadPhase,
            MyObjectLayerPair,
        );

        let temp_allocator = unsafe { JPC_TempAllocatorImpl_new(10 * 1024 * 1024) };
        if temp_allocator.is_null() {
            return Err("Failed to create temp allocator".to_string());
        }

        let job_system = unsafe {
            JPC_JobSystemThreadPool_new2(
                JPC_MAX_PHYSICS_JOBS as u32,
                JPC_MAX_PHYSICS_BARRIERS as u32,
            )
        };
        if job_system.is_null() {
            unsafe { JPC_TempAllocatorImpl_delete(temp_allocator) };
            return Err("Failed to create job system".to_string());
        }

        Ok(Self {
            bodies: Vec::new(),
            name_bodies: HashSet::new(),
            current_time: 0.0,
            current_gravity: -9.81,
            current_friction: 0.2,
            is_simulating: false,
            physics_system,
            temp_allocator,
            job_system,
            rolt_body_ids: Vec::new(),
            chunks: HashMap::new(),
        })
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

        self.name_bodies.insert(body.render_body.name.clone());
        self.bodies.push(body.clone());

        let new_phys = &mut self
            .bodies
            .last_mut()
            .ok_or("Failed to add body")?
            .physical_body;

        let shape_ptr = Self::create_shape_for_body(phys, rend)?;
        if shape_ptr.is_null() {
            self.bodies.pop();
            return Err("Failed to create physics shape".to_string());
        }

        let motion_type = if new_phys.is_kinematic {
            JPC_MOTION_TYPE_KINEMATIC
        } else {
            JPC_MOTION_TYPE_DYNAMIC
        };

        let body_interface = self.physics_system.body_interface();
        let created_body = unsafe {
            body_interface.create_body(&JPC_BodyCreationSettings {
                Position: RVec3::new(
                    new_phys.position.x,
                    new_phys.position.y,
                    new_phys.position.z,
                )
                .into_jolt(),
                Rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0).into_jolt(),
                MotionType: motion_type,
                ObjectLayer: OL_MOVING,
                Shape: shape_ptr,
                LinearVelocity: Vec3::new(
                    new_phys.velocity.x,
                    new_phys.velocity.y,
                    new_phys.velocity.z,
                )
                .into_jolt(),
                AllowSleeping: false,
                GravityFactor: 0.0,
                LinearDamping: 0.0,
                Friction: body.physical_body.edit_params.friction,
                ..Default::default()
            })
        };

        let body_id = created_body.id();
        body_interface.add_body(body_id, JPC_ACTIVATION_ACTIVATE);
        self.rolt_body_ids.push(body_id);

        self.update_time(0.0)?;

        Ok(())
    }

    fn create_shape_for_body(
        physical_body: &crate::body::PhysicalBody,
        render_body: &crate::body::RenderBody,
    ) -> Result<*mut JPC_Shape, String> {
        unsafe {
            let mut shape = ptr::null_mut();
            let mut error = ptr::null_mut();

            let success = match render_body.body_type {
                BodyType::Sphere => {
                    let radius = render_body.dimensions.x;
                    let volume = 4.0 / 3.0 * std::f32::consts::PI * radius * radius * radius;
                    let density = if volume > 0.0 {
                        physical_body.mass / volume
                    } else {
                        0.0
                    };

                    let settings = JPC_SphereShapeSettings {
                        UserData: 0,
                        Density: density,
                        Radius: render_body.dimensions.x,
                    };
                    JPC_SphereShapeSettings_Create(&settings, &mut shape, &mut error)
                }
                BodyType::Rectangle => {
                    let half = render_body.dimensions;
                    let volume = 8.0 * half.x * half.y * half.z;
                    let density = if volume > 0.0 {
                        physical_body.mass / volume
                    } else {
                        0.0
                    };

                    let settings = JPC_BoxShapeSettings {
                        UserData: 0,
                        Density: density,
                        HalfExtent: Vec3::new(
                            render_body.dimensions.x,
                            render_body.dimensions.y,
                            render_body.dimensions.z,
                        )
                        .into_jolt(),
                        ConvexRadius: 0.0,
                        ..Default::default()
                    };
                    JPC_BoxShapeSettings_Create(&settings, &mut shape, &mut error)
                }
            };

            if success {
                Ok(shape)
            } else {
                Err("Failed to create physics shape".to_string())
            }
        }
    }

    fn create_chunk_body(
        physics_system: &PhysicsSystem,
        coord: &ChunkCoord,
        current_friction: f32,
    ) -> Result<BodyId, String> {
        let ground_shape = unsafe {
            let settings = JPC_BoxShapeSettings {
                UserData: 0,
                Density: 0.0,
                HalfExtent: Vec3::new(
                    Chunk::SIZE_X * 0.5 + 10.0,
                    Chunk::SIZE_Y * 0.5,
                    Chunk::SIZE_Z * 0.5 + 10.0,
                )
                .into_jolt(),
                ConvexRadius: 0.0,
                ..Default::default()
            };
            let mut shape = ptr::null_mut();
            let mut error = ptr::null_mut();
            if !JPC_BoxShapeSettings_Create(&settings, &mut shape, &mut error) {
                return Err("Failed to create chunk shape".to_string());
            }
            shape
        };

        let position = Chunk::position(coord);
        let body_interface = physics_system.body_interface();
        let ground_body = unsafe {
            body_interface.create_body(&JPC_BodyCreationSettings {
                Position: RVec3::new(position.x, position.y, position.z).into_jolt(),
                Rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0).into_jolt(),
                MotionType: JPC_MOTION_TYPE_STATIC,
                ObjectLayer: OL_NON_MOVING,
                Shape: ground_shape,
                GravityFactor: 0.0,
                LinearDamping: 0.0,
                Friction: current_friction,
                ..Default::default()
            })
        };

        let body_id = ground_body.id();
        body_interface.add_body(body_id, JPC_ACTIVATION_DONT_ACTIVATE);
        Ok(body_id)
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

            let local_x = pos.x - (chunk_coord.x as f32 * Chunk::SIZE_X);
            let local_z = pos.z - (chunk_coord.z as f32 * Chunk::SIZE_Z);

            if local_x > Chunk::SIZE_X - Chunk::THRESHOLD {
                needed_coords.insert(ChunkCoord {
                    x: chunk_coord.x + 1,
                    z: chunk_coord.z,
                });
            }
            if local_x < Chunk::THRESHOLD {
                needed_coords.insert(ChunkCoord {
                    x: chunk_coord.x - 1,
                    z: chunk_coord.z,
                });
            }
            if local_z > Chunk::SIZE_Z - Chunk::THRESHOLD {
                needed_coords.insert(ChunkCoord {
                    x: chunk_coord.x,
                    z: chunk_coord.z + 1,
                });
            }
            if local_z < Chunk::THRESHOLD {
                needed_coords.insert(ChunkCoord {
                    x: chunk_coord.x,
                    z: chunk_coord.z - 1,
                });
            }
        }

        needed_coords.retain(|coord| !self.chunks.contains_key(coord));

        for coord in needed_coords {
            let body_id =
                Self::create_chunk_body(&self.physics_system, &coord, self.current_friction)?;
            self.chunks.insert(coord, Chunk::new(coord, body_id));
        }

        Ok(())
    }

    pub fn update_physics(&mut self, delta_time: f32) -> Result<(), String> {
        if delta_time <= 0.0 {
            return Ok(());
        }

        self.update_chunks()?;

        let body_interface = self.physics_system.body_interface();

        for (i, body) in self.bodies.iter().enumerate() {
            let phys = &body.physical_body;

            if let Some(&body_id) = self.rolt_body_ids.get(i)
                && !phys.is_kinematic
            {
                let external_force = Vec3::new(phys.force.x, phys.force.y, phys.force.z);
                let gravity_force = Vec3::new(0.0, phys.mass * self.current_gravity, 0.0);
                let total_force = external_force + gravity_force;

                unsafe {
                    JPC_BodyInterface_AddForce(
                        body_interface.as_raw(),
                        body_id.raw(),
                        total_force.into_jolt(),
                    );
                }
            }
        }

        unsafe {
            self.physics_system
                .update(delta_time, 1, self.temp_allocator, self.job_system);
        }

        self.sync_physics_to_bodies(delta_time)?;

        self.current_time += delta_time;
        Ok(())
    }

    fn sync_physics_to_bodies(&mut self, delta_time: f32) -> Result<(), String> {
        if delta_time <= 0.0 {
            return Ok(());
        }

        let body_interface = self.physics_system.body_interface();

        for (body, &body_id) in self.bodies.iter_mut().zip(self.rolt_body_ids.iter()) {
            let phys = &mut body.physical_body;

            let rot =
                unsafe { JPC_BodyInterface_GetRotation(body_interface.as_raw(), body_id.raw()) };

            let rot_quat = Quat::from_jolt(rot);
            body.render_body.rotation = glm::quat(rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w);

            let pos = body_interface.center_of_mass_position(body_id);
            let vel = unsafe {
                JPC_BodyInterface_GetLinearVelocity(body_interface.as_raw(), body_id.raw())
            };
            let vel_vec = Vec3::from_jolt(vel);

            phys.position = glm::vec3(pos.x, pos.y, pos.z);
            phys.velocity = glm::vec3(vel_vec.x, vel_vec.y, vel_vec.z);

            phys.acceleration = (phys.velocity - phys.edit_params.velocity) / self.current_time;

            phys.gravity_force = glm::vec3(0.0, phys.mass * self.current_gravity, 0.0);
            let total_force_vector = phys.force + phys.gravity_force;
            phys.net_force = glm::length(&total_force_vector);

            phys.momentum = phys.mass * phys.velocity;
            phys.kinetic_energy = 0.5 * phys.mass * glm::dot(&phys.velocity, &phys.velocity);
            phys.potential_energy = phys.mass * self.current_gravity * phys.position.y;
            phys.total_mechanical_energy = phys.kinetic_energy + phys.potential_energy;

            let distance = phys.velocity * delta_time;
            phys.displacement = phys.position - phys.edit_params.position;
            phys.distance += glm::length(&distance);

            let work = glm::dot(&total_force_vector, &distance);
            phys.work += work;
            phys.power = work / delta_time;

            phys.impulse = total_force_vector * delta_time;

            if phys.is_kinematic || phys.position.y <= 0.0 {
                phys.normal_force = glm::vec3(0.0, -phys.gravity_force.y, 0.0);
            } else {
                phys.normal_force = glm::vec3(0.0, 0.0, 0.0);
            }

            phys.friction_force = glm::vec3(0.0, 0.0, 0.0);
            phys.elastic_force = glm::vec3(0.0, 0.0, 0.0);
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
            self.update_physics(delta_time)?;
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

        let body_interface = self.physics_system.body_interface();
        for chunk in self.chunks.values() {
            body_interface.remove_body(chunk.body_id);
            body_interface.destroy_body(chunk.body_id);
        }
        self.chunks.clear();

        for i in 0..self.bodies.len() {
            self.bodies[i].physical_body.apply_edit_to_runtime();
            self.update_body(i)?;
            self.bodies[i].render_body.rotation = glm::quat_identity();
        }

        if target_time > 0.0 {
            let step = 0.01;
            let mut accumulated = 0.0;
            while accumulated < target_time {
                let dt = (target_time - accumulated).min(step);
                self.update_physics(dt)?;
                accumulated += dt;
            }
            self.update_render();
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
        if index >= self.bodies.len() || index >= self.rolt_body_ids.len() {
            return Err(format!("Invalid body index: {}", index));
        }

        let body = &self.bodies[index];
        let phys = &body.physical_body;
        let rend = &body.render_body;
        let old_id = self.rolt_body_ids[index];

        let body_interface = self.physics_system.body_interface();
        body_interface.remove_body(old_id);
        body_interface.destroy_body(old_id);

        let shape_ptr = Self::create_shape_for_body(phys, rend)?;
        if shape_ptr.is_null() {
            return Err("Failed to create physics shape for update".to_string());
        }

        let motion_type = if phys.is_kinematic {
            JPC_MOTION_TYPE_KINEMATIC
        } else {
            JPC_MOTION_TYPE_DYNAMIC
        };

        let created_body = unsafe {
            body_interface.create_body(&JPC_BodyCreationSettings {
                Position: RVec3::new(phys.position.x, phys.position.y, phys.position.z).into_jolt(),
                Rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0).into_jolt(),
                MotionType: motion_type,
                ObjectLayer: OL_MOVING,
                Shape: shape_ptr,
                LinearVelocity: Vec3::new(phys.velocity.x, phys.velocity.y, phys.velocity.z)
                    .into_jolt(),
                AllowSleeping: false,
                GravityFactor: 0.0,
                LinearDamping: 0.0,
                Friction: body.physical_body.edit_params.friction,
                ..Default::default()
            })
        };

        let new_id = created_body.id();
        body_interface.add_body(new_id, JPC_ACTIVATION_ACTIVATE);
        self.rolt_body_ids[index] = new_id;

        Ok(())
    }

    pub fn remove_body(&mut self, index: usize) -> Result<(), String> {
        if index >= self.bodies.len() || index >= self.rolt_body_ids.len() {
            return Err(format!("Invalid body index: {}", index));
        }

        let body_interface = self.physics_system.body_interface();
        let body_id = self.rolt_body_ids.remove(index);

        body_interface.remove_body(body_id);
        body_interface.destroy_body(body_id);

        self.name_bodies
            .remove(&self.bodies[index].render_body.name);
        self.bodies.remove(index);

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

impl Drop for Area {
    fn drop(&mut self) {
        let body_interface = self.physics_system.body_interface();

        for &body_id in &self.rolt_body_ids {
            body_interface.remove_body(body_id);
            body_interface.destroy_body(body_id);
        }

        for chunk in self.chunks.values() {
            body_interface.remove_body(chunk.body_id);
            body_interface.destroy_body(chunk.body_id);
        }

        unsafe {
            if !self.job_system.is_null() {
                JPC_JobSystemThreadPool_delete(self.job_system);
            }
            if !self.temp_allocator.is_null() {
                JPC_TempAllocatorImpl_delete(self.temp_allocator);
            }
        }

        unregister_types();
        factory_delete();
    }
}
