use raylib::prelude::*;

use convoluted::{array::Array1D, cost::CrossEntropy, layer::{dense::DenseLayer, sigmoid::SigmoidLayer, LayerChain}};

type Network = convoluted::Network<Array1D<{ 28*28 }>, LayerChain<LayerChain<LayerChain<LayerChain<DenseLayer<{ 28*28 }, 100>, (), Array1D<{ 28*28 }>>, SigmoidLayer<100>, Array1D<{ 28*28 }>>, DenseLayer<100, 10>, Array1D<{ 28*28 }>>, SigmoidLayer<10>, Array1D<{ 28*28 }>>, CrossEntropy, Array1D<10>, usize>;

fn main() {
    let mut network = Network::load("network.bin").unwrap();
    let (mut rl, rt) = init()
    .size(0, 0)
    .fullscreen()
    .build();

    let mut drawing_area = Array1D::new();
    while !rl.window_should_close() {
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let mut pos = rl.get_mouse_position();
            pos /= 20.0; // div by pixel size
            paintbrush(drawing_area.as_mut_slice(), pos.x, pos.y);
            // if !(pos.x >= 28.0 || pos.y >= 28.0) {
            //     for (x_offset, y_offset) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            //         let mut x = pos.x.ceil() as usize + x_offset;
            //         if x == 0 {
            //             continue;
            //         }
            //         x -= 1;
            //         let mut y = pos.y.ceil() as usize + y_offset;
            //         if y == 0 {
            //             continue;
            //         }
            //         y -= 1;
            //         let idx = x + y * 28;
            //         drawing_area[idx] = drawing_area[idx].max(Vector2::new(x as f32, y as f32).distance_to(pos));
            //     }
            // }
        }
        if rl.is_key_pressed(KeyboardKey::KEY_DELETE) {
            drawing_area = Array1D::new();
        }

        let result = network.forward(drawing_area.clone()).0;
        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));

        for y in 0..28 {
            for x in 0..28 {
                let col = (drawing_area[x + 28*y] * 256.0).floor() as u8;
                d.draw_rectangle((x*20) as i32, (y*20) as i32, 20, 20, Color::new(col, col, col, 255));
            }
        }

        let mut sorted: Vec<_> = result.iter().enumerate().collect();
        sorted.sort_by(|x, y| y.1.partial_cmp(x.1).unwrap());
        let sum: f32 = sorted.iter().map(|x| x.1).sum();
        for (i, (number, chance)) in sorted.iter().enumerate() {
            d.draw_text(&format!("{}: {:.2}%", number, *chance / sum * 100.0), 1000, 40 * i as i32, 30, Color::WHITE);
        }
        d.draw_fps(10, 10);
    }
}

fn paintbrush(array: &mut [f32], pos_x: f32, pos_y: f32) {
    let radius = 1.0;
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
