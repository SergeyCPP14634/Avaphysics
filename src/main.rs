use log::*;
use std::panic;

fn main() {
    panic::set_hook(Box::new(|info| {
        error!("PANIC: {:?}", info);
    }));

    if let Err(err) = avaphysics::run_app() {
        error!("App error: {}", err);
        println!("App error: {}", err);
    }
}
