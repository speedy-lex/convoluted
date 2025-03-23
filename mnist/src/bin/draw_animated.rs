use raylib::prelude::*;

use convoluted::{activation::sigmoid::Sigmoid, array::{Array1D, Array2D}, cost::CrossEntropy, layer::{bias::BiasLayer, convolution::Convolution, dense::DenseLayer, pooling::MaxPooling, reshape::{Flatten, Shape}, Layer, LayerChain}};
use serde::{Deserialize, Serialize};

type Network = convoluted::Network<Array1D<{ 28*28 }>, LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<Shape<784, 28, 28>, (), Array1D<784>>, Convolution<5>, Array1D<784>>, BiasLayer<28, 28>, Array1D<784>>, Sigmoid, Array1D<784>>, MaxPooling<2, 14, 14>, Array1D<784>>, Convolution<3>, Array1D<784>>, BiasLayer<14, 14>, Array1D<784>>, Sigmoid, Array1D<784>>, Flatten<{ 14*14 }, 14, 14>, Array1D<784>>, DenseLayer<{ 14*14 }, 64>, Array1D<784>>, Sigmoid, Array1D<784>>, DenseLayer<64, 10>, Array1D<784>>, Sigmoid, Array1D<784>>, CrossEntropy, Array1D<10>, usize>;

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
    let network = Network::load("network.bin").unwrap();
    let (mut rl, rt) = init()
        .title("MNIST classifier")
        .size(0, 0)
        .fullscreen()
        .build();

    let layer = network.into_layer();
    let layers = (
        layer.step.step.step.step.step.step.step.step.step.step.step.step.step,
        layer.step.step.step.step.step.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.step.next,
        layer.step.step.step.step.step.next,
        layer.step.step.step.step.next,
        layer.step.step.step.next,
        layer.step.step.next,
        layer.step.next,
        layer.next
    );
    let mut step = 0usize;

    let width = rl.get_screen_width() as usize;
    let height = rl.get_screen_height() as usize;

    rl.set_target_fps(60);

    let mut drawing_area = Array1D::<{ 28*28 }>::new();
    let mut shaped_area = Array2D::<28, 28>::new();
    let mut pooled_area = Array2D::<14, 14>::new();
    let mut dense_area_1 = Array1D::<64>::new();
    let mut dense_area_2 = Array1D::<10>::new();
    while !rl.window_should_close() {
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) && step == 0 {
            let mut pos = rl.get_mouse_position();
            pos -= Vector2::new(width as f32 / 2.0, height as f32 / 2.0);
            pos += Vector2::new(14.0 * PIXEL_SIZE as f32, 14.0 * PIXEL_SIZE as f32);
            pos /= PIXEL_SIZE as f32; // div by pixel size
            paintbrush(drawing_area.as_mut_slice(), pos.x, pos.y, config.brush_size);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_DELETE) {
            drawing_area = Array1D::new();
            step = 0;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_SPACE) {
            step += 1;

            match step {
                1 => {
                    shaped_area = layers.0.forward(drawing_area.clone()).0;
                    shaped_area = layers.1.forward(shaped_area).0;
                }
                2 => {
                    shaped_area = layers.2.forward(shaped_area).0;
                }
                3 => {
                    shaped_area = layers.3.forward(shaped_area).0;
                }
                4 => {
                    pooled_area = layers.4.forward(shaped_area.clone()).0;
                }
                5 => {
                    pooled_area = layers.5.forward(pooled_area).0;
                }
                6 => {
                    pooled_area = layers.6.forward(pooled_area).0;
                }
                7 => {
                    pooled_area = layers.7.forward(pooled_area).0;
                }
                8 => {
                    let flat_area = layers.8.forward(pooled_area.clone()).0;
                    dense_area_1 = layers.9.forward(flat_area).0;
                }
                9 => {
                    dense_area_1 = layers.10.forward(dense_area_1).0;
                }
                10 => {
                    dense_area_2 = layers.11.forward(dense_area_1.clone()).0;
                }
                11 => {
                    dense_area_2 = layers.12.forward(dense_area_2).0;
                }
                _ => {
                    step = 0;
                }
            }
        }

        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));

        match step {
            0 => {
                draw_1d_array::<{ 28*28 }, 28, 28>(&mut d, &drawing_area, width, height);        
            }
            1..=3 => {
                draw_2d_array(&mut d, &shaped_area, width, height);
            }
            4..=7 => {
                draw_2d_array(&mut d, &pooled_area, width, height);
            }
            8..=9 => {
                draw_1d_array::<{ 8*8 }, 8, 8>(&mut d, &dense_area_1, width, height);
            }
            10..=11 => {
                draw_1d_array::<10, 5, 2>(&mut d, &dense_area_2, width, height);
            }
            _ => {}
        }
        if step == 11 {
            let sorted: Vec<_> = dense_area_2.iter().enumerate().collect();
            let sum: f32 = sorted.iter().map(|x| x.1).sum();
            for (i, (number, chance)) in sorted[5..].iter().enumerate() {
                d.draw_line(width as i32/2 - (1.5*PIXEL_SIZE as f32) as i32 + (i*PIXEL_SIZE) as i32, height as i32/2 + PIXEL_SIZE as i32, 150 * i as i32 + width as i32 / 2 - (150 * 2), height as i32 / 2 - 20 + 250, Color::WHITE);
                d.draw_text(&format!("{}: {:.1}%", number, *chance / sum * 100.0), 150 * i as i32 + width as i32 / 2 + 20 - (150.0 * 2.5) as i32,  height as i32 / 2 - 15 + 250, 30, Color::WHITE);
            }
            for (i, (number, chance)) in sorted[..5].iter().enumerate() {
                let text = format!("{}: {:.1}%", number, *chance / sum * 100.0);
                d.draw_line(width as i32/2 - (1.5*PIXEL_SIZE as f32) as i32 + (i*PIXEL_SIZE) as i32, height as i32/2 - PIXEL_SIZE as i32, 150 * i as i32 + width as i32 / 2 - (150 * 2), height as i32 / 2 + 20 - 250, Color::WHITE);
                d.draw_text(&text, 150 * i as i32 + width as i32 / 2 - (150 * 2) - d.measure_text(&text, 30)/2,  height as i32 / 2 - 15 - 250, 30, Color::WHITE);
            }
        }
    }
    std::fs::write("convoluted.toml", toml::to_string(&config).unwrap()).unwrap()
}

fn draw_2d_array<const X: usize, const Y: usize>(d: &mut RaylibDrawHandle, array: &Array2D<X, Y>, width: usize, height: usize) {
    let array_max = *array.iter().flat_map(|x| x.iter()).max_by(|x, y| x.total_cmp(y)).unwrap();
    let array_min = array.iter().flat_map(|x| x.iter()).min_by(|x, y| x.total_cmp(y)).unwrap().abs();
    let scale = array_max.max(array_min);
    for y in 0..Y {
        for x in 0..X {
            let col = array.array[y][x];
            let col = if col < 0.0 {
                let col = -(col / scale * 255.0).floor() as u8;
                Color::new(col, 0, 0, 255)
            } else {
                let col = (col / scale * 255.0).floor() as u8;
                Color::new(col, col, col, 255)
            };
            d.draw_rectangle((x * PIXEL_SIZE) as i32 + width as i32/2 - (X/2 * PIXEL_SIZE) as i32, (y * PIXEL_SIZE) as i32 + height as i32/2 - (Y/2 * PIXEL_SIZE) as i32, PIXEL_SIZE as i32, PIXEL_SIZE as i32, col);
        }   
    }
}

/// N = X*Y
/// because i'm to lazy to do typenum rn
fn draw_1d_array<const N: usize, const X: usize, const Y: usize>(d: &mut RaylibDrawHandle, array: &Array1D<N>, width: usize, height: usize) {
    assert_eq!(N, X*Y);
    draw_2d_array(d, unsafe {
        std::mem::transmute::<&Array1D<N>, &Array2D<X, Y>>(array)
    }, width, height);
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