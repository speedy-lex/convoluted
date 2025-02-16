use raylib::prelude::*;

use convoluted::{cost::CrossEntropy, layer::{dense::DenseLayer, sigmoid::SigmoidLayer, LayerChain}};

type Network = convoluted::Network<[f32; 28*28], LayerChain<LayerChain<LayerChain<LayerChain<DenseLayer<{ 28*28 }, 100>, (), [f32; 28*28]>, SigmoidLayer<100>, [f32; 28*28]>, DenseLayer<100, 10>, [f32; 28*28]>, SigmoidLayer<10>, [f32; 28*28]>, CrossEntropy, [f32; 10], usize>;

fn main() {
    let mut network = Network::load("network.bin").unwrap();
    println!("{:?}", network.forward([0.0; 28*28]).0);
}