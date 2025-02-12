use std::marker::PhantomData;

use crate::{cost::CostFunction, layer::{Layer, LayerChain}};

pub struct Network<I, L: Layer<I>, C: CostFunction<P, E>, P, E> {
    layer: L,
    _input_marker: PhantomData<I>,
    _cost_marker: PhantomData<C>,
    _predicted_marker: PhantomData<P>,
    _label_marker: PhantomData<E>,
}
impl<I, C: CostFunction<<() as Layer<I>>::Output, E>, E> Network<I, (), C, <() as Layer<I>>::Output, E> {
    pub fn new() -> Self {
        Self {
            layer: (),
            _input_marker: PhantomData,
            _cost_marker: PhantomData,
            _predicted_marker: PhantomData,
            _label_marker: PhantomData,
        }
    }
}
impl<I, L: Layer<I>, C: CostFunction<P, E>, P, E> Network<I, L, C, P, E> {
    pub fn add<N: Layer<L::Output, Output = std::vec::Vec<f32>>>(self, next: N) -> Network<I, LayerChain<L, N, I>, C, P, E> {
        Network {
            layer: LayerChain {
                step: self.layer,
                next,
                _marker: PhantomData,
            },
            _input_marker: PhantomData,
            _cost_marker: PhantomData,
            _predicted_marker: PhantomData,
            _label_marker: PhantomData,
        }
    }
}
impl<I, L: Layer<I, Output = Vec<f32>>, C: CostFunction<L::Output, E>, E> Network<I, L, C, L::Output, E> {
    pub fn forward(&mut self, input: I) -> L::Output {
        self.layer.forward(input)
    }
    fn backwards(&mut self, output: &L::Output, expected: &E) {
        self.layer.backward(C::derivative(output, expected));
    }
    fn collect_gradients(&mut self, input: I, expected: E) {
        let forward = self.forward(input);
        self.backwards(&forward, &expected);
    }
    pub fn learn_batch(&mut self, input: Vec<I>, expected: Vec<E>, learn_rate: f32) {
        let batch_size = input.len();
        for (input, expected) in input.into_iter().zip(expected.into_iter()) {
            self.collect_gradients(input, expected);
        }
        self.layer.apply_gradients(learn_rate / batch_size as f32);
        self.layer.clear_gradients();
    }
}