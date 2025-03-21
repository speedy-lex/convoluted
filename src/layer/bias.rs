use crate::array::Array2D;

use super::Layer;

use rand::{rng, Rng};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub struct BiasLayer<const X: usize, const Y: usize> {
    biases: Array2D<X, Y>,
}
impl<const X: usize, const Y: usize> BiasLayer<X, Y> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn random() -> Self {
        let mut rng = rng();
        let mut biases = Array2D::new();
        for x in 0..X {
            for y in 0..Y {
                biases.array[y][x] = rng.random::<f32>() * 2.0 - 1.0;
            }
        }
        Self { biases }
    }
}
impl<const X: usize, const Y: usize> Layer<Array2D<X, Y>> for BiasLayer<X, Y> {
    type Output = Array2D<X, Y>;

    type ForwardData = ();

    type Gradients = Array2D<X, Y>;

    fn forward(&self, mut input: Array2D<X, Y>) -> (Self::Output, Self::ForwardData) {
        input += self.biases.clone();
        (input, ())
    }

    fn backward(&self, forward: Self::Output, _forward_data: Self::ForwardData) -> (Array2D<X, Y>, Self::Gradients) {
        (forward.clone(), forward)
    }

    fn apply_gradients(&mut self, mut gradients: Self::Gradients, multiplier: f32) {
        gradients *= multiplier;
        self.biases += gradients
    }
}