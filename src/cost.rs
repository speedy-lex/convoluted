pub trait CostFunction<P, E> {
    fn cost(predicted: &P, expected: &E) -> f32;
    fn derivative(predicted: &P, expected: &E) -> P;
}

pub struct Mse;
impl<const I: usize> CostFunction<[f32; I], [f32; I]> for Mse {
    fn cost(predicted: &[f32; I], expected: &[f32; I]) -> f32 {
        let mut result = 0.0;
        for (p, e) in predicted.iter().zip(expected) {
            result += (*p - *e).powi(2)
        }
        result
    }

    fn derivative(predicted: &[f32; I], expected: &[f32; I]) -> [f32; I] {
        let mut result = [0.0; I];
        for (r, (p, e)) in result.iter_mut().zip(predicted.iter().zip(expected)) {
            *r = 2.0 * (*p - *e);
        }
        result
    }
}
impl<const I: usize> CostFunction<[f32; I], usize> for Mse {
    fn cost(predicted: &[f32; I], expected: &usize) -> f32 {
        let mut result = 0.0;
        for (i, p) in predicted.iter().enumerate() {
            result += (*p - ((i == *expected) as u32 as f32)).powi(2)
        }
        result
    }

    fn derivative(predicted: &[f32; I], expected: &usize) -> [f32; I] {
        let mut result = [0.0; I];
        for (r, (i, p)) in result.iter_mut().zip(predicted.iter().enumerate()) {
            *r = 2.0 * (*p - ((i == *expected) as u32 as f32));
        }
        result
    }
}

pub struct CrossEntropy;
impl CrossEntropy {
    fn softmax<const I: usize>(values: &[f32; I]) -> [f32; I] {
        let mut result = [0.0; I];
        let mut total = 0.0;
        for value in values {
            total += value.exp();
        }
        for (r, value) in result.iter_mut().zip(values) {
            *r = value.exp() / total;
        }
        result
    }
}
impl<const I: usize> CostFunction<[f32; I], usize> for CrossEntropy {
    fn cost(predicted: &[f32; I], expected: &usize) -> f32 {
        -Self::softmax(predicted)[*expected].ln()
    }

    fn derivative(predicted: &[f32; I], expected: &usize) -> [f32; I] {
        debug_assert!(I > *expected);
        let mut softmax = Self::softmax(predicted);
        softmax[*expected] -= 1.0;
        softmax
    }
}