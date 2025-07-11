use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod convolution;
pub mod pooling;
pub mod bias;
pub mod dense;
pub mod reshape;

pub trait Layer<I> {
    type Output;
    type ForwardData;
    type Gradients;

    fn forward(&self, input: I) -> (Self::Output, Self::ForwardData);
    fn backward(&self, forward: Self::Output, forward_data: Self::ForwardData) -> (I, Self::Gradients);
    fn apply_gradients(&mut self, gradients: Self::Gradients, multiplier: f32);
}
impl<I> Layer<I> for () {
    type Output = I;
    type ForwardData = ();
    type Gradients = ();
    
    #[inline]
    fn forward(&self, input: I) -> (I, ()) {
        (input, ())
    }

    #[inline]
    fn backward(&self, forward: Self::Output, _forward_data: Self::ForwardData) -> (I, Self::Gradients) {
        (forward, ())
    }
    
    #[inline]
    fn apply_gradients(&mut self, _gradients: Self::Gradients, _multiplier: f32) {}
}

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, Default)]
pub struct LayerChain<S, N, I> {
    pub step: S,
    pub next: N,
    pub _marker: PhantomData<I>,
}

impl<S, N, I> LayerChain<S, N, I> where S: Layer<I>, N: Layer<S::Output> {
    pub fn push<T: Layer<N::Output>>(self, next: T) -> LayerChain<LayerChain<S, N, I>, T, I> {
        LayerChain {
            step: self,
            next,
            _marker: PhantomData::<I>,
        }
    }
    pub fn new(step: S, next: N) -> Self {
        Self {
            step,
            next,
            _marker: PhantomData,
        }
    }
}

impl<S, N, I> Layer<I> for LayerChain<S, N, I>
where
    S: Layer<I>,
    N: Layer<S::Output>,
{
    type Output = N::Output;
    type ForwardData = (S::ForwardData, N::ForwardData);
    type Gradients = (S::Gradients, N::Gradients);

    #[inline]
    fn forward(&self, input: I) -> (Self::Output, Self::ForwardData) {
        let intermediate = self.step.forward(input);
        let output = self.next.forward(intermediate.0);
        (output.0, (intermediate.1, output.1))
    }
    #[inline]
    fn backward(&self, forward: Self::Output, forward_data: Self::ForwardData) -> (I, Self::Gradients) {
        let intermediate = self.next.backward(forward, forward_data.1);
        let output = self.step.backward(intermediate.0, forward_data.0);
        (output.0, (output.1, intermediate.1))
    }
    
    #[inline]
    fn apply_gradients(&mut self, gradients: Self::Gradients, multiplier: f32) {
        self.step.apply_gradients(gradients.0, multiplier);
        self.next.apply_gradients(gradients.1, multiplier);
    }
}

#[macro_export]
macro_rules! layer_chain {
    ($a:expr) => {
        $a
    };
    ($a:expr, $b:expr) => {
        LayerChain::new($a, $b)
    };
    ($a:expr, $($tail:expr),+) => {
        LayerChain::new($a, layer_chain!($($tail),+))
    };
}