use std::time::Instant;

use raylib::prelude::*;

#[allow(unused_imports)]
use convoluted::{activation::sigmoid::Sigmoid, array::Array1D, cost::CrossEntropy, layer::{bias::BiasLayer, convolution::Convolution, dense::DenseLayer, pooling::MaxPooling, reshape::{Flatten, Shape}, LayerChain}};
use serde::{Deserialize, Serialize};

type Network = convoluted::Network<Array1D<{ 28*28 }>, LayerChain<LayerChain<LayerChain<LayerChain<DenseLayer<{ 28*28 }, 100>, (), Array1D<{ 28*28 }>>, Sigmoid, Array1D<{ 28*28 }>>, DenseLayer<100, 10>, Array1D<{ 28*28 }>>, Sigmoid, Array1D<{ 28*28 }>>, CrossEntropy, Array1D<10>, usize>;
// type Network = convoluted::Network<Array1D<{ 28*28 }>, LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<Shape<784, 28, 28>, (), Array1D<784>>, Convolution<5>, Array1D<784>>, BiasLayer<28, 28>, Array1D<784>>, Sigmoid, Array1D<784>>, MaxPooling<2, 14, 14>, Array1D<784>>, Convolution<3>, Array1D<784>>, BiasLayer<14, 14>, Array1D<784>>, Sigmoid, Array1D<784>>, Flatten<{ 14*14 }, 14, 14>, Array1D<784>>, DenseLayer<{ 14*14 }, 64>, Array1D<784>>, Sigmoid, Array1D<784>>, DenseLayer<64, 10>, Array1D<784>>, Sigmoid, Array1D<784>>, CrossEntropy, Array1D<10>, usize>;

const PIXEL_SIZE: usize = 20;

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    chance_bars: bool,
    sorted: bool,
    auto_clear: bool,
    brush_size: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self { chance_bars: true, sorted: true, auto_clear: true, brush_size: 1.5 }
    }
}

fn main() {
    let config = load_cfg();
    let network = Network::load("network_dense.bin").unwrap();
    let (mut rl, rt) = init()
        .title("MNIST classifier")
        .size(0, 0)
        .fullscreen()
        .build();

    let width = rl.get_screen_width() as usize;
    let height = rl.get_screen_height() as usize;

    rl.set_target_fps(60);

    let mut last_interaction = Instant::now();
    let mut drawing_area = Array1D::new();
    while !rl.window_should_close() {
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let mut pos = rl.get_mouse_position();
            pos -= Vector2::new(width as f32 / 2.0, height as f32 / 2.0);
            pos += Vector2::new(14.0 * PIXEL_SIZE as f32, 14.0 * PIXEL_SIZE as f32);
            pos /= PIXEL_SIZE as f32; // div by pixel size
            paintbrush(drawing_area.as_mut_slice(), pos.x, pos.y, config.brush_size);
            last_interaction = Instant::now();
        }
        if rl.is_key_pressed(KeyboardKey::KEY_DELETE) || (config.auto_clear && last_interaction.elapsed().as_secs_f64() >= 3.0) {
            drawing_area = Array1D::new();
        }

        let result = network.forward(drawing_area.clone()).0;
        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));
        
        let drawing_area_clear = !drawing_area.iter().any(|x| *x != 0.0);
        if drawing_area_clear {
            let text = "Draw a number 0-9";
            let text_length = d.measure_text(text, 80);
            d.draw_text(text, (width/2) as i32 - text_length/2, (height/2) as i32 - 14 * PIXEL_SIZE as i32 - 120, 80, Color::WHITE);
        }

        for y in 0..28 {
            for x in 0..28 {
                let col = 255 - (drawing_area[x + 28*y] * 256.0).floor() as u8;
                d.draw_rectangle((x * PIXEL_SIZE) as i32 + width as i32/2 - 14 * PIXEL_SIZE as i32, (y * PIXEL_SIZE) as i32 + height as i32/2 - 14 * PIXEL_SIZE as i32, PIXEL_SIZE as i32, PIXEL_SIZE as i32, Color::new(col, col, col, 255));
            }
        }

        if !drawing_area_clear {
            let mut sorted: Vec<_> = result.iter().enumerate().collect();
            if config.sorted {
                sorted.sort_by(|x, y| y.1.partial_cmp(x.1).unwrap());
            }
            let sum: f32 = sorted.iter().map(|x| x.1).sum();
            for (i, (number, chance)) in sorted.iter().enumerate() {
                let chance = **chance / sum;
                d.draw_text(&format!("{}: {:.2}%", number, chance * 100.0), width as i32 / 2 + 14 * PIXEL_SIZE as i32 + 20, 40 * i as i32 + height as i32 / 2 - 40 * 5 + 5, 30, Color::WHITE);
                if config.chance_bars {
                    d.draw_rectangle(width as i32 / 2 + 14 * PIXEL_SIZE as i32 + 20 + 150, 40 * i as i32 + height as i32 / 2 - 40 * 5 + 5, 100, 20, Color::GRAY);
                    let color = Color::new(255, 0, 0, 255).lerp(Color::new(0, 255, 0, 255), chance);
                    d.draw_rectangle(width as i32 / 2 + 14 * PIXEL_SIZE as i32 + 20 + 150, 40 * i as i32 + height as i32 / 2 - 40 * 5 + 5,(chance * 100.0) as i32, 20, color);
                }
            }
        }
    }
    std::fs::write("convoluted.toml", toml::to_string(&config).unwrap()).unwrap()
}

fn paintbrush(array: &mut [f32], pos_x: f32, pos_y: f32, brush_size: f32) {
    let samples = 4; // Increase for better antialiasing
    let step = 1.0 / (samples as f32);
    
    let min_x = (pos_x - brush_size).floor() as i32;
    let max_x = (pos_x + brush_size).ceil() as i32;
    let min_y = (pos_y - brush_size).floor() as i32;
    let max_y = (pos_y + brush_size).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut coverage = 0.0;

            for i in 0..samples {
                for j in 0..samples {
                    let sample_x = x as f32 + (i as f32 + 0.5) * step;
                    let sample_y = y as f32 + (j as f32 + 0.5) * step;
                    let dist_sq = (sample_x - pos_x).powi(2) + (sample_y - pos_y).powi(2);

                    if dist_sq <= brush_size.powi(2) {
                        coverage += 1.0;
                    }
                }
            }

            coverage /= (samples * samples) as f32;

            if (0..28).contains(&x) && (0..28).contains(&y) {
                let index = (y as usize) * 28 + (x as usize);
                array[index] = array[index].max(coverage);
                array[index] = array[index].clamp(0.0, 1.0);
            }
        }
    }
}

fn load_cfg() -> Config {
    let str = std::fs::read_to_string("convoluted.toml");
    if str.is_err() {
        eprintln!("{:?}", str.unwrap_err());
        return Config::default();
    }
    let str = str.unwrap();
    let cfg = toml::from_str(&str);
    if cfg.is_err() {
        eprintln!("{:?}", cfg.unwrap_err());
        return Config::default();
    }
    cfg.unwrap()
}