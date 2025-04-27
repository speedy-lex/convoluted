use convoluted::{activation::sigmoid::Sigmoid, array::Array1D, cost::{CostFunction, Mse}, layer::{dense::DenseLayer, LayerChain}, layer_chain, Network};

use raylib::prelude::*;

use rand::{rng, rngs::ThreadRng, Rng};

fn main() {
    let mut network = Network::<Array1D<2>, _, Mse, Array1D<1>, Array1D<1>>::new(
        layer_chain!(
            DenseLayer::random(),
            Sigmoid::new(),
            DenseLayer::<4, 1>::random(),
            Sigmoid::new()
        )
    );
    let mut rng = rng();
    let mut data = vec![];
    create_data(&mut data, &mut rng, 0);
    // let mut cost = f32::INFINITY;
    // while cost > 50.0 {
    //     data.shuffle(&mut rng);
    //     for chunk in data.chunks(10) {
    //         network.learn_batch(chunk.to_owned(), 1.0);
    //     }
    //     cost = 0.0;
    //     for point in data.clone() {
    //         let out = network.forward(point.0).0;
    //         cost += Mse::cost(&out, &point.1);
    //     }
    // }

    let (mut rl, rt) = init()
        .size(0, 0)
        .fullscreen()
        .vsync()
        .build();

    let width = rl.get_screen_width() as usize;
    let height = rl.get_screen_height() as usize;
    
    rl.set_target_fps(60);

    let mut shader = rl.load_shader_from_memory(&rt, None, Some(include_str!("shader.frag")));
    
    let screensize = shader.get_shader_location("screenSize");
    shader.set_shader_value(screensize, Vector2::new(width as f32, height as f32));

    let weights00 = shader.get_shader_location("weights00");
    let weights01 = shader.get_shader_location("weights01");
    let weights02 = shader.get_shader_location("weights02");
    let weights03 = shader.get_shader_location("weights03");
    let biases0 = shader.get_shader_location("biases0");
    let weights1 = shader.get_shader_location("weights1");
    let bias1 = shader.get_shader_location("bias1");
    
    while !rl.window_should_close() {
        if rl.is_key_down(KeyboardKey::KEY_SPACE) {
            for _ in 0..10 {
                network.learn_batch(data.to_owned(), 1.0);
            }
        }
        if rl.is_key_down(KeyboardKey::KEY_N) {
            network.learn_batch(data.to_owned(), 1.0);
        }
        let mut new_dataset = false;
        if rl.is_key_pressed(KeyboardKey::KEY_ONE) {
            new_dataset = true;
            create_data(&mut data, &mut rng, 0);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_TWO) {
            new_dataset = true;
            create_data(&mut data, &mut rng, 1);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
            new_dataset = true;
            create_data(&mut data, &mut rng, 2);
        }
        if new_dataset || rl.is_key_pressed(KeyboardKey::KEY_R) {
            network = Network::new(
                layer_chain!(
                    DenseLayer::random(),
                    Sigmoid::new(),
                    DenseLayer::<4, 1>::random(),
                    Sigmoid::new()
                )
            );
        }

        shader.set_shader_value(bias1, network.layer.next.next.step.biases[0]);
        shader.set_shader_value(weights1, network.layer.next.next.step.weights.array.as_ref()[0]);
        shader.set_shader_value(biases0, *network.layer.step.biases.array.as_ref());
        shader.set_shader_value(weights00, network.layer.step.weights[0]);
        shader.set_shader_value(weights01, network.layer.step.weights[1]);
        shader.set_shader_value(weights02, network.layer.step.weights[2]);
        shader.set_shader_value(weights03, network.layer.step.weights[3]);
        
        let mut d = rl.begin_drawing(&rt);
        d.begin_shader_mode(&mut shader).draw_rectangle(0, 0, width as i32, height as i32, Color::WHITE);
        
        d.draw_line(width as i32/2, 0, width as i32/2, height as i32, Color::WHITE);
        d.draw_line(0, height as i32/2, width as i32, height as i32/2, Color::WHITE);
        for point in &data {
            d.draw_circle(((point.0[0] + 4.0) / 8.0 * width as f32) as i32, ((point.0[1] + 4.0) / 8.0 * height as f32) as i32, 5.0, if point.1[0] != 0.0 { Color::BLUE } else { Color::RED });
        }
        let mut cost = 0.0;
        let mut correct = 0;
        for point in data.clone() {
            let out = network.forward(point.0).0;
            cost += Mse::cost(&out, &point.1);
            correct += ((out[0] - point.1[0]).abs() < 0.5) as usize 
        }
        d.draw_text(&format!("{correct}/100 {cost:02}"), 10, 100, 20, Color::WHITE);
        d.draw_fps(10, 10);
    }
}

fn create_data(data: &mut Vec<(Array1D<2>, Array1D<1>)>, rng: &mut ThreadRng, set: usize) {
    data.clear();
    for _ in 0..100 {
        let x = rng.random::<f32>() * 4.0 - 2.0;
        let y = rng.random::<f32>() * 4.0 - 2.0;

        let mut array = Array1D::<2>::new();
        array[0] = x;
        array[1] = y;

        let in_circle = if set == 0 {
            x.powi(2) + y.powi(2) <= 2.0
        } else if set == 1 {
            (1.0/-x <= y && y <= 1.0/x) || (1.0/x <= y && y <= 1.0/-x)
        } else {
            x.exp2() < y || -(-x).exp2() > y
        };
        let mut label_array = Array1D::new();
        label_array[0] = in_circle as u32 as f32;
        data.push((array, label_array));
    }
}