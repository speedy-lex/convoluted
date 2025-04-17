use std::io::Write;
use std::time::Instant;

use convoluted::activation::sigmoid::Sigmoid;
use convoluted::array::Array1D;
use convoluted::cost::{CostFunction, CrossEntropy};
use convoluted::layer::{dense::DenseLayer, LayerChain};
use convoluted::Network;
use rand::{rng, seq::SliceRandom};

fn main() {
    let mut network = Network::<Array1D<{ 28*28 }>, _, CrossEntropy, Array1D<10>, _>::new(
        LayerChain::new(DenseLayer::<{ 28*28 }, 100>::random())
            .push(Sigmoid::new())
            .push(DenseLayer::<100, 10>::random())
            .push(Sigmoid::new())
    );
    let (input, labels) = mnist::get_mnist_train();
    let mut data: Vec<_> = input.into_iter().zip(labels).collect();
    let (test_input, test_labels) = mnist::get_mnist_test();
    let mut rng = rng();
    for x in 0..10 {
        println!("Epoch {}/10", x+1);
        let start_time = Instant::now();
        data.shuffle(&mut rng);
        for (i, chunk) in data.chunks(10).enumerate() {
            network.learn_batch(chunk.to_owned(), 1.0);
            if i % 89 == 0 || i == 5999 {
                print!("\r{:04}/6000 | [{}>{}] {:.1}%", i+1, "=".repeat(i/300), " ".repeat(19 - i/300), (i+1) as f32 / 60.0);
                std::io::stdout().flush().unwrap();
            }
        }
        println!();
        println!("Epoch time: {:.03}", start_time.elapsed().as_secs_f64());
        let mut cost = 0.0;
        let mut correct = 0;
        for (input, label) in test_input.iter().zip(&test_labels) {
            let out = network.forward(input.clone()).0;
            if out.iter().enumerate().max_by(|(_, x), (_, y)| {x.partial_cmp(y).unwrap()}).unwrap().0 == *label {
                correct += 1;
            }
            cost += CrossEntropy::cost(&out, label);
        }
        println!("> Cost: {:.3}\n> Test accuracy: {:.1}", cost / test_input.len() as f32, correct as f32 / test_input.len() as f32 * 100.0);
        println!();
    }
    network.save("./network_dense.bin").expect("couldn't save");
}
