use rand::{rng, Rng};


use crate::array::Array2D;

use super::Layer;

pub const fn is_odd(n: usize) -> usize {
    if n % 2 == 0 {
        panic!("even convolution sizes are not allowed")
    }
    0
}

#[derive(Clone, Default)]
pub struct Convolution<const N: usize>
where
    [(); is_odd(N)]:, {
    kernel: Array2D<N, N>,
    rotated_kernel: Array2D<N, N>
}

impl<const N: usize> Convolution<N>
where
    [(); is_odd(N)]:, {
    pub fn new() -> Convolution<N> {
        Convolution::default()
    }
    pub fn random() -> Convolution<N> {
        let mut rng = rng();
        let mut array = Array2D::new();
        for x in 0..N {
            for y in 0..N {
                array.array[y][x] = rng.random::<f32>() * 2.0 - 1.0;
            }
        }

        let mut x = Self { kernel: array, rotated_kernel: Array2D::new() };
        x.update_rotated_kernel();
        x
    }
    pub fn update_rotated_kernel(&mut self) {
        for x in 0..N {
            for y in 0..N {
                self.rotated_kernel.array[y][x] = self.kernel.array[N - 1 - y][N - 1 - x];
            }
        }
    }
    fn convolve<const X: usize, const Y: usize>(array: &Array2D<X, Y>, kernel: &Array2D<N, N>) -> Array2D<X, Y> {
        let mut new = Array2D::new();
        let kernel_offset = (N - 1)/2; // offset to move kernel center to pixel

        for new_x in 0..X {
            for new_y in 0..Y {
                let mut value = 0.0;
                for kernel_x in 0..N {
                    for kernel_y in 0..N {
                        value += kernel.array[kernel_y][kernel_x]
                            * try_sample(
                                array,
                                // this can only fail if you have arrays with ridiculous sizes which no one can fit in memory so it's ok
                                (new_x + kernel_x).wrapping_sub(kernel_offset),
                                (new_y + kernel_y).wrapping_sub(kernel_offset),
                            ).unwrap_or_default();
                    }
                }
                new.array[new_y][new_x] = value;
            }
        }
        new
    }
}

impl<const X: usize, const Y: usize, const N: usize> Layer<Array2D<X, Y>> for Convolution<N>
where
    [(); is_odd(N)]: {
    type Output = Array2D<X, Y>;

    type ForwardData = Array2D<X, Y>;

    type Gradients = Array2D<N, N>;

    fn forward(&self, input: Array2D<X, Y>) -> (Self::Output, Self::ForwardData) {
        (Self::convolve(&input, &self.kernel), input)
    }

    fn backward(&self, forward: Self::Output, forward_data: Self::ForwardData) -> (Array2D<X, Y>, Self::Gradients) {
        (Self::convolve(&forward, &self.rotated_kernel), Array2D::new())
    }

    fn apply_gradients(&mut self, mut gradients: Self::Gradients, multiplier: f32) {
        gradients *= multiplier;
        self.kernel += gradients;
        self.update_rotated_kernel();
    }
}

fn try_sample<const X: usize, const Y: usize>(array: &Array2D<X, Y>, index_x: usize, index_y: usize) -> Option<f32> {
    if index_x >= X || index_y >= Y {
        return None;
    }
    Some(array.array[index_y][index_x])
}