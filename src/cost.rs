pub trait CostFunction<P, E> {
    fn cost(predicted: &P, expected: &E) -> f32;
    fn derivative(predicted: &P, expected: &E) -> Vec<f32>;
}

pub struct Mse;
impl CostFunction<Vec<f32>, Vec<f32>> for Mse {
    fn cost(predicted: &Vec<f32>, expected: &Vec<f32>) -> f32 {
        let mut result = 0.0;
        for (p, e) in predicted.iter().zip(expected) {
            result += (*p - *e).powi(2)
        }
        result
    }

    fn derivative(predicted: &Vec<f32>, expected: &Vec<f32>) -> Vec<f32> {
        let mut result = Vec::with_capacity(predicted.len());
        for (p, e) in predicted.iter().zip(expected) {
            result.push(2.0 * (*p - *e));
        }
        result
    }
}

pub struct CrossEntropy;
impl CrossEntropy {
    fn softmax(values: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(values.len());
        let mut total = 0.0;
        for value in values {
            total += value.exp();
        }
        for value in values {
            result.push(value.exp() / total);
        }
        result
    }
}
impl CostFunction<Vec<f32>, usize> for CrossEntropy {
    fn cost(predicted: &Vec<f32>, expected: &usize) -> f32 {
        -Self::softmax(predicted)[*expected].ln()
    }

    fn derivative(predicted: &Vec<f32>, expected: &usize) -> Vec<f32> {
        debug_assert!(predicted.len() > *expected);
        let mut softmax = Self::softmax(predicted);
        softmax[*expected] -= 1.0;
        softmax
    }
}