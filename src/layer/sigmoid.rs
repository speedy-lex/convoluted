use super::Layer;

#[derive(Clone, Copy)]
pub struct SigmoidLayer<const I: usize> {
    input: [f32; I]
}

impl<const I: usize> SigmoidLayer<I> {
    #[inline(always)]
    fn activate(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline(always)]
    fn derivate(x: f32) -> f32 {
        let activated = Self::activate(x);
        activated * (1.0 - activated)
    }

    pub fn new() -> Self {
        Self::default()
    }
}
impl<const I: usize> Default for SigmoidLayer<I> {
    fn default() -> Self {
        Self { input: [0.0; I] }
    }
}

impl<const I: usize> Layer<[f32; I]> for SigmoidLayer<I> {
    type Output = [f32; I];

    fn forward(&mut self, mut input: [f32; I]) -> Self::Output {
        self.input = input;
        for x in input.iter_mut() {
            *x = Self::activate(*x);
        }
        input
    }

    fn backward(&mut self, mut forward: Self::Output) -> [f32; I] {
        for (forward, input) in forward.iter_mut().zip(&self.input) {
            *forward *= Self::derivate(*input);
        }
        forward
    }

    fn apply_gradients(&mut self, _multiplier: f32) {}

    fn clear_gradients(&mut self) {}
}