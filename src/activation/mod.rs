use crate::{array::{Array1D, Array2D}, layer::Layer};

pub mod relu;
pub mod leaky_relu;
pub mod sigmoid;

pub trait Activation {
    fn activate(x: f32) -> f32;
    fn derivate(x: f32) -> f32;
}

impl<T: Activation, const N: usize> Layer<Array1D<N>> for T {
    type Output = Array1D<N>;

    type ForwardData = Array1D<N>;

    type Gradients = ();

    fn forward(&self, mut input: Array1D<N>) -> (Self::Output, Self::ForwardData) {
        let forward_data = input.clone();
        for x in input.iter_mut() {
            *x = Self::activate(*x);
        }
        (input, forward_data)
    }

    fn backward(&self, mut forward: Self::Output, forward_data: Self::ForwardData) -> (Array1D<N>, Self::Gradients) {
        for (forward, input) in forward.iter_mut().zip(forward_data.iter()) {
            *forward *= Self::derivate(*input);
        }
        (forward, ())
    }
    
    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}
impl<T: Activation, const X: usize, const Y: usize> Layer<Array2D<X, Y>> for T {
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