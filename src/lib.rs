use std::marker::PhantomData;

#[cfg(feature = "bincode")]
use std::{path::Path, fs::write, fs::read};
use array::Array1D;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{cost::CostFunction, layer::Layer};

pub mod cost;
pub mod layer;
pub mod activation;
pub mod array;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct Network<I, L: Layer<I>, C: CostFunction<P, E>, P, E> {
    pub layer: L,
    _input_marker: PhantomData<I>,
    _cost_marker: PhantomData<C>,
    _predicted_marker: PhantomData<P>,
    _label_marker: PhantomData<E>,
}

impl<I, L: Layer<I>, C: CostFunction<P, E>, P, E> Network<I, L, C, P, E> {
    pub fn into_layer(self) -> L {
        self.layer
    }
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
impl<I, L: Layer<I, Output = Array1D<N>>, C: CostFunction<L::Output, E>, E, const N: usize> Network<I, L, C, L::Output, E> {
    pub fn forward(&self, input: I) -> (L::Output, L::ForwardData) {
        self.layer.forward(input)
    }
    fn backwards(&self, output: &L::Output, expected: &E, forward_data: L::ForwardData) -> (I, L::Gradients) {
        self.layer.backward(C::derivative(output, expected), forward_data)
    }
    fn get_gradients(&self, input: I, expected: E) -> L::Gradients {
        let forward = self.forward(input);
        self.backwards(&forward.0, &expected, forward.1).1
    }
    pub fn learn_batch(&mut self, data: Vec<(I, E)>, learn_rate: f32) {
        let batch_size = data.len();
        if batch_size == 0 {
            return;
        }
        let mut gradients = Vec::with_capacity(data.len());
        for (input, expected) in data {
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