use crate::array::{Array1D, Array2D};

use super::Layer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct ReluLayer;

#[cfg(not(feature = "serde"))]
#[derive(Clone, Copy, Default)]
pub struct ReluLayer;

impl ReluLayer {
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

impl<const I: usize> Layer<Array1D<I>> for ReluLayer {
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
impl<const X: usize, const Y: usize> Layer<Array2D<X, Y>> for ReluLayer {
    type Output = Array2D<X, Y>;
    type ForwardData = Array2D<X, Y>;
    type Gradients = ();

    fn forward(&self, mut input: Array2D<X, Y>) -> (Self::Output, Self::ForwardData) {
        let forward_data = input.clone();
        for x in input.iter_mut() {
            for y in x {
                *y = Self::activate(*y);
            }
        }
        (input, forward_data)
    }

    fn backward(&self, mut forward: Self::Output, forward_data: Self::ForwardData) -> (Array2D<X, Y>, Self::Gradients) {
        for (forward, input) in forward.iter_mut().zip(forward_data.iter()) {
            for (forward, input) in forward.iter_mut().zip(input.iter()) {
                *forward *= Self::derivate(*input);
            }
        }
        (forward, ())
    }

    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}