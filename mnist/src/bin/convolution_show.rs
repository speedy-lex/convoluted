use raylib::prelude::*;

use convoluted::{array::Array1D, cost::CrossEntropy, layer::{convolution::Convolution, dense::DenseLayer, pooling::MaxPooling, relu::ReluLayer, reshape::{Flatten, Shape}, sigmoid::SigmoidLayer, Layer, LayerChain}};

// type Network = convoluted::Network<Array1D<784>, LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<Shape<784, 28, 28>, (), Array1D<784>>, Convolution<5>, Array1D<784>>, Convolution<3>, Array1D<784>>, Flatten<784, 28, 28>, Array1D<784>>, ReluLayer<784>, Array1D<784>>, DenseLayer<784, 100>, Array1D<784>>, SigmoidLayer<100>, Array1D<784>>, DenseLayer<100, 10>, Array1D<784>>, SigmoidLayer<10>, Array1D<784>>, CrossEntropy, Array1D<10>, usize>;
// type Network = convoluted::Network<Array1D<784>, LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<Shape<784, 28, 28>, (), Array1D<784>>, Convolution<5>, Array1D<784>>, Flatten<784, 28, 28>, Array1D<784>>, ReluLayer<784>, Array1D<784>>, Shape<784, 28, 28>, Array1D<784>>, Convolution<3>, Array1D<784>>, Flatten<784, 28, 28>, Array1D<784>>, ReluLayer<784>, Array1D<784>>, DenseLayer<784, 100>, Array1D<784>>, SigmoidLayer<100>, Array1D<784>>, DenseLayer<100, 10>, Array1D<784>>, SigmoidLayer<10>, Array1D<784>>, CrossEntropy, Array1D<10>, usize>;
type Network = convoluted::Network<Array1D<784>, LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<LayerChain<Shape<784, 28, 28>, (), Array1D<784>>, Convolution<5>, Array1D<784>>, MaxPooling<2, 14, 14>, Array1D<784>>, Flatten<196, 14, 14>, Array1D<784>>, ReluLayer, Array1D<784>>, DenseLayer<196, 100>, Array1D<784>>, SigmoidLayer<100>, Array1D<784>>, DenseLayer<100, 10>, Array1D<784>>, SigmoidLayer<10>, Array1D<784>>, CrossEntropy, Array1D<10>, usize>;

fn main() {
    let layer = Network::load("network.bin").unwrap().into_layer().step.step.step.step.step;
    let train = mnist::get_mnist_train().0.into_iter().map(|x| {
        layer.forward(x).0
    }).collect::<Vec<_>>();
    let conv = layer.step.step.next.kernel;
    println!("{:?}", conv.array);

    let (mut rl, rt) = init()
    .size(28*20, 28*20)
    .build();

    let mut idx = 0;
    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
            idx += 1;
        }
        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));

        for y in 0..14 {
            for x in 0..14 {
                let col = (train[idx][x + 14*y] * 256.0).floor();
                let col = if col < 0.0 {
                    Color::new(col as u8, 0, 0, 255)
                } else {
                    Color::new(0, col as u8, 0, 255)
                };
                d.draw_rectangle((x*20) as i32, (y*20) as i32, 20, 20, col);
            }
        }
    }
}