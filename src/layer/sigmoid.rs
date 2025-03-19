use crate::array::{Array1D, Array2D};

use super::Layer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde")]
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct SigmoidLayer;

#[cfg(not(feature = "serde"))]
#[derive(Clone, Copy, Default)]
pub struct SigmoidLayer;

impl SigmoidLayer {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        let activated = Self::activate(x);
        activated * (1.0 - activated)
    }

    pub fn new() -> Self {
        Self
    }
}

impl<const I: usize> Layer<Array1D<I>> for SigmoidLayer {
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
impl<const X: usize, const Y: usize> Layer<Array2D<X, Y>> for SigmoidLayer {
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