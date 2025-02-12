use cost::{CostFunction, CrossEntropy};
use network::Network;

pub mod cost;
pub mod layer;
mod network;

fn main() {
    let mut network = Network::<Vec<f32>, _, CrossEntropy, _, _>::new();
    let out = network.forward(vec![10.0, -1.0]);
    println!("{out:?}, {}, {:?}", CrossEntropy::cost(&out, &0), CrossEntropy::derivative(&out, &0))
}
