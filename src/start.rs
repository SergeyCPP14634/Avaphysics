extern crate nalgebra_glm as glm;
extern crate sdl3;

mod area;
mod body;
mod camera;
mod gui;
mod physical_renderer;
mod vulkan_renderer;

use sdl3::event::Event;
use sdl3::keyboard::Keycode;

use crate::area::Area;
use crate::camera::*;
use crate::gui::*;
use crate::physical_renderer::*;

fn main() {
    let mut physical_renderer = PhysicalRenderer::new(800, 600).unwrap();

    let mut camera = Camera::new(
        glm::vec3(0.0, 0.0, 0.0),
        physical_renderer.window.size_in_pixels().0,
        physical_renderer.window.size_in_pixels().1,
        0.1,
        50.0,
    );

    let area = Area::new().unwrap();

    let mut gui = Gui::new(area);

    camera.toggle_cursor_mode(&physical_renderer.sdl_context, &physical_renderer.window);

    let mut last_time = std::time::Instant::now();

    'running: loop {
        for event in physical_renderer
            .sdl_context
            .event_pump()
            .unwrap()
            .poll_iter()
        {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::Window {
                    win_event: sdl3::event::WindowEvent::Resized(..),
                    ..
                } => {
                    camera.width = physical_renderer.window.size_in_pixels().0;
                    camera.height = physical_renderer.window.size_in_pixels().1;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Tab),
                    repeat: false,
                    ..
                } => {
                    camera.toggle_cursor_mode(
                        &physical_renderer.sdl_context,
                        &physical_renderer.window,
                    );
                    physical_renderer.update_event_imgui(None);
                }
                _ => {}
            }

            physical_renderer.update_event(&event);

            if camera.is_selected() {
                camera.update_event(&event);
            } else {
                physical_renderer.update_event_imgui(Some(&event));
            }
        }

        let current_time = std::time::Instant::now();
        let delta_time = (current_time - last_time).as_secs_f32();
        last_time = current_time;

        camera.update(delta_time);
        gui.area.update_simulation(delta_time).unwrap();

        physical_renderer.update(&camera, &mut gui).unwrap();

        physical_renderer
            .render(&mut gui, &camera, delta_time)
            .unwrap();
    }
}
