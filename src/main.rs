use cost::{CostFunction, CrossEntropy};
use layer::{dense::DenseLayer, relu::ReluLayer, sigmoid::SigmoidLayer, LayerChain};
use network::Network;

pub mod cost;
pub mod layer;
mod network;

fn main() {
    let mut network = Network::<[f32; 2], _, CrossEntropy, [f32; 2], _>::new(
        LayerChain::new(DenseLayer::<2, 2>::random())
            .push(ReluLayer::new())
            .push(DenseLayer::<2, 2>::random())
            .push(SigmoidLayer::new())
    );
    for _ in 0..1000000 {
        network.learn_batch(vec![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]], vec![1, 1, 0, 0], 0.01);

    }
    let out = network.forward([1.0, 0.0]);
    let out2 = network.forward([0.0, 1.0]);
    let out3 = network.forward([0.0, 0.0]);
    let out4 = network.forward([1.0, 1.0]);
    println!("{out:?}, {out2:?}, {out3:?}, {out4:?}");
    println!("{}, {}, {}, {}", CrossEntropy::cost(&out, &1), CrossEntropy::cost(&out2, &1), CrossEntropy::cost(&out3, &0), CrossEntropy::cost(&out4, &0));
}
