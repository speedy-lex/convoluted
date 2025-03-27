use super::Activation;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Sigmoid {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        let activated = Self::activate(x);
        activated * (1.0 - activated)
    }
}