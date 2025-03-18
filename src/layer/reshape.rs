use typenum::{Const, ToUInt, U};
use std::ops::Mul;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::array::{Array1D, Array2D};

use super::Layer;

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape<const N: usize, const X: usize, const Y: usize>
where
    Const<N>: ToUInt,
    Const<X>: ToUInt,
    Const<Y>: ToUInt,
    U<X>: Mul<U<Y>, Output = U<N>>, {}

impl<const N: usize, const X: usize, const Y: usize> Layer<Array1D<N>> for Shape<N, X, Y>
where
    Const<N>: ToUInt,
    Const<X>: ToUInt,
    Const<Y>: ToUInt,
    U<X>: Mul<U<Y>, Output = U<N>>, {
    type Output = Array2D<X, Y>;

    type ForwardData = ();

    type Gradients = ();

    fn forward(&self, input: Array1D<N>) -> (Self::Output, Self::ForwardData) {
        (Array2D::from(unsafe { std::mem::transmute::<Box<[f32; N]>, Box<[[f32; X]; Y]>>(input.array) }), ())
    }

    fn backward(&self, forward: Self::Output, _forward_data: Self::ForwardData) -> (Array1D<N>, Self::Gradients) {
        (Array1D::from(unsafe { std::mem::transmute::<Box<[[f32; X]; Y]>, Box<[f32; N]>>(forward.array) }), ())
    }

    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Flatten<const N: usize, const X: usize, const Y: usize>
where
    Const<N>: ToUInt,
    Const<X>: ToUInt,
    Const<Y>: ToUInt,
    U<X>: Mul<U<Y>, Output = U<N>>, {}

impl<const N: usize, const X: usize, const Y: usize> Layer<Array2D<X, Y>> for Flatten<N, X, Y>
where
    Const<N>: ToUInt,
    Const<X>: ToUInt,
    Const<Y>: ToUInt,
    U<X>: Mul<U<Y>, Output = U<N>>, {
    type Output = Array1D<N>;

    type ForwardData = ();

    type Gradients = ();

    fn forward(&self, input: Array2D<X, Y>) -> (Self::Output, Self::ForwardData) {
        (Array1D::from(unsafe { std::mem::transmute::<Box<[[f32; X]; Y]>, Box<[f32; N]>>(input.array) }), ())
    }

    fn backward(&self, forward: Self::Output, _forward_data: Self::ForwardData) -> (Array2D<X, Y>, Self::Gradients) {
        (Array2D::from(unsafe { std::mem::transmute::<Box<[f32; N]>, Box<[[f32; X]; Y]>>(forward.array) }), ())
    }

    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}