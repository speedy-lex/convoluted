use crate::array::Array1D;

use super::Layer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct ReluLayer<const I: usize>;

#[cfg(not(feature = "serde"))]
#[derive(Clone, Copy, Default)]
pub struct ReluLayer<const I: usize>;

impl<const I: usize> ReluLayer<I> {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        (x >= 0.0) as u32 as f32
    }

    pub fn new() -> Self {
        Self
    }
}

impl<const I: usize> Layer<Array1D<I>> for ReluLayer<I> {
    type Output = Array1D<I>;
    type ForwardData = Array1D<I>;
    type Gradients = ();

    fn forward(&self, mut input: Array1D<I>) -> (Self::Output, Self::ForwardData) {
        let forward_data = input.clone();
        for x in input.iter_mut() {
            *x = Self::activate(*x);
        }
        (input, forward_data)
    }

    fn backward(&self, mut forward: Self::Output, forward_data: Self::ForwardData) -> (Array1D<I>, Self::Gradients) {
        for (forward, input) in forward.iter_mut().zip(forward_data.iter()) {
            *forward *= Self::derivate(*input);
        }
        (forward, ())
    }

    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}