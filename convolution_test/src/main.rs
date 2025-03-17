use convoluted::array::Array1D;
use convoluted::cost::{CostFunction, CrossEntropy};
use convoluted::layer::dense::DenseLayer;
use convoluted::layer::reshape::{Flatten, Shape};
use convoluted::layer::LayerChain;
use convoluted::Network;
use convoluted::layer::convolution::Convolution;

fn main() {
    let mut network = Network::<_, _, CrossEntropy, Array1D<4>, usize>::new(
        LayerChain::new(DenseLayer::<2, 4>::new())
            .push(Shape::<4, 2, 2>{})
            .push(Convolution::<3>::random())
            .push(Flatten::<4, 2, 2>{})
    );

    let data = vec![
        Array1D::new(); 2
    ].into_iter().enumerate().map(|(i, mut array)| {
        array[i] = 1.0;
        (array, i)
    }).collect::<Vec<_>>();

    for _ in 0..1000 {
        network.learn_batch(data.clone(), 0.1);
        println!("cost: {}", data.iter().map(|x| {
            CrossEntropy::cost(&network.forward(x.0.clone()).0,&x.1)
        }).sum::<f32>());
    }
}
