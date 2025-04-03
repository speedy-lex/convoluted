use super::Activation;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeakyRelu;

impl LeakyRelu {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for LeakyRelu {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        if x < 0.0 {
            x * 0.01
        } else {
            x
        }
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        (x >= 0.0) as u32 as f32 * 0.99 + 0.01
    }
}