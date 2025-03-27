use super::Activation;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Relu {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        x.max(0.0)
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        (x >= 0.0) as u32 as f32
    }
}