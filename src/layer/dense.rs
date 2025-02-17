use rand::{rng, Rng};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::array::{Array1D, Array2D};

use super::Layer;

#[cfg(feature = "serde")]
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct DenseLayer<const I: usize, const O: usize> {
    weights: Array2D<I, O>,
    biases: Array1D<O>,
}

#[cfg(not(feature = "serde"))]
#[derive(Clone, Default)]
pub struct DenseLayer<const I: usize, const O: usize> {
    weights: Array2D<I, O>,
    biases: Array1D<O>,
}
impl<const I: usize, const O: usize> Layer<Array1D<I>> for DenseLayer<I, O> {
    type Output = Array1D<O>;
    type ForwardData = Array1D<I>;
    type Gradients = (Array2D<I, O>, Array1D<O>);

    fn forward(&self, input: Array1D<I>) -> (Self::Output, Self::ForwardData) {
        let forward_data = input.clone();
        let mut output = self.biases.clone();
        for (i, node) in output.iter_mut().enumerate() {
            for (j, input) in input.iter().enumerate() {
                *node += self.weights[i][j] * input;
            }
        }
        (output, forward_data)
    }

    fn backward(&self, forward: Self::Output, forward_data: Self::ForwardData) -> (Array1D<I>, Self::Gradients) {
        let mut gradients = Self::Gradients::default();
        for (i, bias) in gradients.1.iter_mut().enumerate() {
            *bias += forward[i];
        }
        let mut output = Array1D::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..I {
            for o in 0..O {
                gradients.0[o][i] += forward_data[i] * forward[o];
                output[i] += self.weights[o][i] * forward[o];
            }
        }
        (output, gradients)
    }

    fn apply_gradients(&mut self, gradients: Self::Gradients, multiplier: f32) {
        for (bias, gradient) in self.biases.iter_mut().zip(gradients.1.as_ref()) {
            *bias += *gradient * multiplier;
        }
        for (node, gradients) in self.weights.iter_mut().zip(gradients.0.as_ref()) {
            for (weight, gradient) in node.iter_mut().zip(gradients) {
                *weight += *gradient * multiplier;
            }
        }
    }
}

impl<const I: usize, const O: usize> DenseLayer<I, O> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn random() -> Self {
        let mut rng = rng();
        let mut biases = Array1D::new();
        for bias in biases.iter_mut() {
            *bias = rng.random::<f32>() * 2.0 - 1.0;
        }
        let mut weights = Array2D::new();
        for node in weights.iter_mut() {
            for weight in node {
                *weight = rng.random::<f32>() * 2.0 - 1.0;
            }
        }
        Self {
            weights,
            biases,
        }
    }
}
