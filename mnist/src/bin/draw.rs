use raylib::prelude::*;

use convoluted::{activation::sigmoid::Sigmoid, array::Array1D, cost::CrossEntropy, layer::{dense::DenseLayer, LayerChain}};

type Network = convoluted::Network<Array1D<{ 28*28 }>, LayerChain<LayerChain<LayerChain<LayerChain<DenseLayer<{ 28*28 }, 100>, (), Array1D<{ 28*28 }>>, Sigmoid, Array1D<{ 28*28 }>>, DenseLayer<100, 10>, Array1D<{ 28*28 }>>, Sigmoid, Array1D<{ 28*28 }>>, CrossEntropy, Array1D<10>, usize>;

const PIXEL_SIZE: usize = 20;

fn main() {
    let mut network = Network::load("network.bin").unwrap();
    let (mut rl, rt) = init()
    .size(0, 0)
    .fullscreen()
    .build();

    let width = rl.get_screen_width() as usize;
    let height = rl.get_screen_height() as usize;

    rl.set_target_fps(60);

    let mut drawing_area = Array1D::new();
    while !rl.window_should_close() {
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let mut pos = rl.get_mouse_position();
            pos -= Vector2::new(width as f32 / 2.0, height as f32 / 2.0);
            pos += Vector2::new(14.0 * PIXEL_SIZE as f32, 14.0 * PIXEL_SIZE as f32);
            pos /= PIXEL_SIZE as f32; // div by pixel size
            paintbrush(drawing_area.as_mut_slice(), pos.x, pos.y);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_DELETE) {
            drawing_area = Array1D::new();
        }

        let result = network.forward(drawing_area.clone()).0;
        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));

        for y in 0..28 {
            for x in 0..28 {
                let col = 255 - (drawing_area[x + 28*y] * 256.0).floor() as u8;
                d.draw_rectangle((x * PIXEL_SIZE) as i32 + width as i32/2 - 14 * PIXEL_SIZE as i32, (y * PIXEL_SIZE) as i32 + height as i32/2 - 14 * PIXEL_SIZE as i32, PIXEL_SIZE as i32, PIXEL_SIZE as i32, Color::new(col, col, col, 255));
            }
        }

        let mut sorted: Vec<_> = result.iter().enumerate().collect();
        sorted.sort_by(|x, y| y.1.partial_cmp(x.1).unwrap());
        let sum: f32 = sorted.iter().map(|x| x.1).sum();
        for (i, (number, chance)) in sorted.iter().enumerate() {
            d.draw_text(&format!("{}: {:.2}%", number, *chance / sum * 100.0), width as i32 / 2 + 14 * PIXEL_SIZE as i32 + 20, 40 * i as i32 + height as i32 / 2 - 40 * 5 + 5, 30, Color::WHITE);
        }
    }
}

fn paintbrush(array: &mut [f32], pos_x: f32, pos_y: f32) {
    let radius = 1.5;
    let samples = 4; // Increase for better antialiasing
    let step = 1.0 / (samples as f32);
    
    let min_x = (pos_x - radius).floor() as i32;
    let max_x = (pos_x + radius).ceil() as i32;
    let min_y = (pos_y - radius).floor() as i32;
    let max_y = (pos_y + radius).ceil() as i32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut coverage = 0.0;

            for i in 0..samples {
                for j in 0..samples {
                    let sample_x = x as f32 + (i as f32 + 0.5) * step;
                    let sample_y = y as f32 + (j as f32 + 0.5) * step;
                    let dist_sq = (sample_x - pos_x).powi(2) + (sample_y - pos_y).powi(2);

                    if dist_sq <= radius * radius {
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
