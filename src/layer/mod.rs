use std::marker::PhantomData;

pub mod dense;
pub mod relu;
pub mod sigmoid;

pub trait Layer<I> {
    type Output;

    fn forward(&mut self, input: I) -> Self::Output;
    fn backward(&mut self, forward: Self::Output) -> I;
    fn apply_gradients(&mut self, multiplier: f32);
    fn clear_gradients(&mut self);
}
impl<I> Layer<I> for () {
    type Output = I;

    fn forward(&mut self, input: I) -> Self::Output {
        input
    }

    fn backward(&mut self, forward: Self::Output) -> I {
        forward
    }
    
    fn apply_gradients(&mut self, _multiplier: f32) {}
    
    fn clear_gradients(&mut self) {}
}

pub struct LayerChain<S, N, I> {
    pub step: S,
    pub next: N,
    pub _marker: PhantomData<I>,
}

impl<S, I> LayerChain<S, (), I> {
    pub fn new(step: S) -> Self {
        Self {
            step,
            next: (),
            _marker: PhantomData,
        }
    }
}
impl<S, N, I> LayerChain<S, N, I> where S: Layer<I>, N: Layer<S::Output> {
    pub fn push<T: Layer<N::Output>>(self, next: T) -> LayerChain<LayerChain<S, N, I>, T, I> {
        LayerChain {
            step: self,
            next,
            _marker: PhantomData::<I>,
        }
    }
}

impl<S, N, I> Layer<I> for LayerChain<S, N, I>
where
    S: Layer<I>,
    N: Layer<S::Output>,
{
    type Output = N::Output;

    #[inline]
    fn forward(&mut self, input: I) -> Self::Output {
        let intermediate = self.step.forward(input);
        self.next.forward(intermediate)
    }
    #[inline]
    fn backward(&mut self, forward: Self::Output) -> I {
        let intermediate = self.next.backward(forward);
        self.step.backward(intermediate)
    }
    
    #[inline]
    fn apply_gradients(&mut self, multiplier: f32) {
        self.step.apply_gradients(multiplier);
        self.next.apply_gradients(multiplier);
    }
    
    #[inline]
    fn clear_gradients(&mut self) {
        self.step.clear_gradients();
        self.next.clear_gradients();
    }
}
