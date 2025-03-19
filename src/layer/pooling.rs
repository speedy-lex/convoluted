use typenum::{Cmp, Const, Less, ToUInt, U};
use std::ops::{Mul, Rem};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::array::Array2D;

use super::Layer;

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Default)]
pub struct MaxPooling<const N: usize, const A: usize, const B: usize>
where
    Const<N>: ToUInt,
    U<N>: Cmp<U<65536>, Output = Less> {}

impl<const X: usize, const Y: usize, const N: usize, const A: usize, const B: usize> Layer<Array2D<X, Y>> for MaxPooling<N, A, B>
where
    Const<N>: ToUInt,
    Const<X>: ToUInt,
    Const<Y>: ToUInt,
    Const<A>: ToUInt,
    Const<B>: ToUInt,

    U<N>: Cmp<U<65536>, Output = Less>,
    U<X>: Rem<U<N>, Output = U<0>>,
    U<Y>: Rem<U<N>, Output = U<0>>,
    // A = X/N
    // B = Y/N
    U<A>: Mul<U<N>, Output = U<X>>,
    U<B>: Mul<U<N>, Output = U<Y>>, {
    type Output = Array2D<A, B>;

    type ForwardData = Array2D<A, B>;

    type Gradients = ();

    fn forward(&self, input: Array2D<X, Y>) -> (Self::Output, Self::ForwardData) {
        let mut out = Array2D::new();
        let mut forward_data = Array2D::new(); // TODO: Array2d<T>

        for chunk_x in 0..A {
            for chunk_y in 0..B {
                let (mut max_x, mut max_y) = (N, N); // set to an invalid value to catch errors (should be overwritten)
                let mut max = f32::NEG_INFINITY;

                for x in 0..N {
                    for y in 0..N {
                        let x_index = chunk_x * N + x;
                        let y_index = chunk_y * N + y;
                        let val = input.array[y_index][x_index];

                        if val > max {
                            max = val;
                            max_x = x;
                            max_y = y;
                        }
                    }
                }

                forward_data.array[chunk_y][chunk_x] = f32::from_bits(((max_y << 16) | max_x) as u32);
                out.array[chunk_y][chunk_x] = max;
            }
        }
        (out, forward_data)
    }

    fn backward(&self, forward: Self::Output, forward_data: Self::ForwardData) -> (Array2D<X, Y>, Self::Gradients) {
        let mut out = Array2D::new();
        for chunk_x in 0..A {
            for chunk_y in 0..B {
                let packed = forward_data.array[chunk_y][chunk_x].to_bits();
                let x = packed & 0xffff;
                let y = (packed >> 16) & 0xffff;

                let x_index = chunk_x * N + x as usize;
                let y_index = chunk_y * N + y as usize;

                out.array[y_index][x_index] = forward[chunk_y][chunk_x];
            }
        }
        (out, ())
    }

    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}
