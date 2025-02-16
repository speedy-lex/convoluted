use std::marker::PhantomData;

#[cfg(feature = "bincode")]
use std::{path::Path, fs::write, fs::read};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{cost::CostFunction, layer::Layer};

pub mod cost;
pub mod layer;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Network<I, L: Layer<I>, C: CostFunction<P, E>, P, E> {
    layer: L,
    _input_marker: PhantomData<I>,
    _cost_marker: PhantomData<C>,
    _predicted_marker: PhantomData<P>,
    _label_marker: PhantomData<E>,
}
impl<I, C: CostFunction<L::Output, E>, E, L: Layer<I>> Network<I, L, C, L::Output, E> {
    pub fn new(layer: L) -> Self {
        Self {
            layer,
            _input_marker: PhantomData,
            _cost_marker: PhantomData,
            _predicted_marker: PhantomData,
            _label_marker: PhantomData,
        }
    }
}
impl<I, L: Layer<I, Output = [f32; N]>, C: CostFunction<L::Output, E>, E, const N: usize> Network<I, L, C, L::Output, E> {
    pub fn forward(&mut self, input: I) -> (L::Output, L::ForwardData) {
        self.layer.forward(input)
    }
    fn backwards(&mut self, output: &L::Output, expected: &E, forward_data: L::ForwardData) -> (I, L::Gradients) {
        self.layer.backward(C::derivative(output, expected), forward_data)
    }
    fn get_gradients(&mut self, input: I, expected: E) -> L::Gradients {
        let forward = self.forward(input);
        self.backwards(&forward.0, &expected, forward.1).1
    }
    pub fn learn_batch(&mut self, input: Vec<I>, expected: Vec<E>, learn_rate: f32) {
        let batch_size = input.len();
        if batch_size == 0 {
            return;
        }
        let mut gradients = Vec::with_capacity(input.len());
        for (input, expected) in input.into_iter().zip(expected.into_iter()) {
            gradients.push(self.get_gradients(input, expected));
        }
        for gradient in gradients {
            self.layer.apply_gradients(gradient, -learn_rate / batch_size as f32);
        }
    }

}

#[cfg(feature = "bincode")]
impl<I, C: CostFunction<L::Output, E>, E, L: Layer<I>> Network<I, L, C, L::Output, E> 
where Self: Serialize + for<'a> Deserialize<'a> {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let x = bincode::serialize(self)?;
        write(path, &x)?;
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let x = read(path)?;
        Ok(bincode::deserialize(&x)?)
    }
}