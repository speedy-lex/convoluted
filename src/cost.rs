use crate::array::Array1D;

pub trait CostFunction<P, E> {
    fn cost(predicted: &P, expected: &E) -> f32;
    fn derivative(predicted: &P, expected: &E) -> P;
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Mse;
impl<const I: usize> CostFunction<Array1D<I>, Array1D<I>> for Mse {
    fn cost(predicted: &Array1D<I>, expected: &Array1D<I>) -> f32 {
        let mut result = 0.0;
        for (p, e) in predicted.iter().zip(expected.iter()) {
            result += (*p - *e).powi(2)
        }
        result / I as f32
    }

    fn derivative(predicted: &Array1D<I>, expected: &Array1D<I>) -> Array1D<I> {
        let mut result = Array1D::new();
        for (r, (p, e)) in result.iter_mut().zip(predicted.iter().zip(expected.iter())) {
            *r = 2.0 * (*p - *e);
        }
        result
    }
}
impl<const I: usize> CostFunction<Array1D<I>, usize> for Mse {
    fn cost(predicted: &Array1D<I>, expected: &usize) -> f32 {
        let mut result = 0.0;
        for (i, p) in predicted.iter().enumerate() {
            result += (*p - ((i == *expected) as u32 as f32)).powi(2)
        }
        result / I as f32
    }

    fn derivative(predicted: &Array1D<I>, expected: &usize) -> Array1D<I> {
        let mut result = Array1D::new();
        for (r, (i, p)) in result.iter_mut().zip(predicted.iter().enumerate()) {
            *r = 2.0 * (*p - ((i == *expected) as u32 as f32));
        }
        result
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CrossEntropy;
impl CrossEntropy {
    fn softmax<const I: usize>(values: &Array1D<I>) -> Array1D<I> {
        let mut result = Array1D::new();
        let mut total = 0.0;
        for value in values.iter() {
            total += value.exp();
        }
        for (r, value) in result.iter_mut().zip(values.iter()) {
            *r = value.exp() / total;
        }
        result
    }
}
impl<const I: usize> CostFunction<Array1D<I>, usize> for CrossEntropy {
    fn cost(predicted: &Array1D<I>, expected: &usize) -> f32 {
        -Self::softmax(predicted)[*expected].ln()
    }

    fn derivative(predicted: &Array1D<I>, expected: &usize) -> Array1D<I> {
        debug_assert!(I > *expected);
        let mut softmax = Self::softmax(predicted);
        softmax[*expected] -= 1.0;
        softmax
    }
}