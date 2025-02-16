use convoluted::cost::{CostFunction, CrossEntropy};
use convoluted::layer::{dense::DenseLayer, sigmoid::SigmoidLayer, LayerChain};
use convoluted::Network;
use rand::{rng, seq::SliceRandom};

fn main() {
    let mut network = Network::<[f32; 28*28], _, CrossEntropy, [f32; 10], _>::new(
        LayerChain::new(DenseLayer::<{ 28*28 }, 100>::random())
            .push(SigmoidLayer::new())
            .push(DenseLayer::<100, 10>::random())
            .push(SigmoidLayer::new())
    );
    let (input, labels) = mnist::get_mnist_train();
    let mut data: Vec<_> = input.into_iter().zip(labels).collect();
    let (test_input, test_labels) = mnist::get_mnist_test();
    let mut rng = rng();
    for x in 0..20 {
        println!("shuffling");
        data.shuffle(&mut rng);
        println!("epoch {} start", x+1);
        for chunk in data.chunks(10) {
            let (input, labels) = chunk.iter().copied().unzip();
            network.learn_batch(input, labels, 1.0);
        }
        println!("epoch {} complete", x+1);
        let mut cost = 0.0;
        let mut correct = 0;
        for (input, label) in test_input.iter().zip(&test_labels) {
            let out = network.forward(*input).0;
            if out.iter().enumerate().max_by(|(_, x), (_, y)| {x.partial_cmp(y).unwrap()}).unwrap().0 == *label {
                correct += 1;
            }
            cost += CrossEntropy::cost(&out, label);
        }
        println!("cost: {}, correct: {}", cost / test_input.len() as f32, correct as f32 / test_input.len() as f32 * 100.0);
    }
    network.save("./network.bin").expect("couldn't save");
}
