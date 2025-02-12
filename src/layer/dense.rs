
use rand::{rng, Rng};

use super::Layer;

#[derive(Clone, Copy)]
pub struct DenseLayer<const I: usize, const O: usize> {
    weights: [[f32; I]; O],
    biases: [f32; O],

    input: [f32; I],
    weight_gradients: [[f32; I]; O],
    bias_gradients: [f32; O],
}
impl<const I: usize, const O: usize> Layer<[f32; I]> for DenseLayer<I, O> {
    type Output = [f32; O];

    fn forward(&mut self, input: [f32; I]) -> Self::Output {
        self.input = input;
        let mut output = self.biases;
        for (i, node) in output.iter_mut().enumerate() {
            for (j, input) in self.input.iter().enumerate() {
                *node += self.weights[i][j] * input;
            }
        }
        output
    }

    fn backward(&mut self, forward: Self::Output) -> [f32; I] {
        for (i, bias) in self.bias_gradients.iter_mut().enumerate() {
            *bias += forward[i];
        }
        let mut output = [0.0; I];
        #[allow(clippy::needless_range_loop)]
        for i in 0..I {
            for o in 0..O {
                self.weight_gradients[o][i] += self.input[i] * forward[o];
                output[i] += self.weights[o][i] * forward[o];
            }
        }
        output
    }

    fn apply_gradients(&mut self, multiplier: f32) {
        for (bias, gradient) in self.biases.iter_mut().zip(&self.bias_gradients) {
            *bias += *gradient * multiplier;
        }
        for (node, gradients) in self.weights.iter_mut().zip(&self.weight_gradients) {
            for (weight, gradient) in node.iter_mut().zip(gradients) {
                *weight += *gradient * multiplier;
            }
        }
    }

    fn clear_gradients(&mut self) {
        self.bias_gradients = [0.0; O];
        self.weight_gradients = [[0.0; I]; O];
    }

}

impl<const I: usize, const O: usize> DenseLayer<I, O> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn random() -> Self {
        let mut rng = rng();
        let mut biases = [0.0; O];
        for bias in &mut biases {
            *bias = rng.random::<f32>() * 2.0 - 1.0;
        }
        let mut weights = [[0.0; I]; O];
        for node in &mut weights {
            for weight in node {
                *weight = rng.random::<f32>() * 2.0 - 1.0;
            }
        }
        Self {
            weights,
            biases,
            input: [0.0; I],
            weight_gradients: [[0.0; I]; O],
            bias_gradients: [0.0; O],
        }
    }
}

impl<const I: usize, const O: usize> Default for DenseLayer<I, O> {
    fn default() -> Self {
        Self {
            weights: [[0.0; I]; O],
            biases: [0.0; O],
            input: [0.0; I],
            weight_gradients: [[0.0; I]; O],
            bias_gradients: [0.0; O],
        }
    }
}