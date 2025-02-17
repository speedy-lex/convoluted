use std::{borrow::Borrow, ops::{AddAssign, Deref, DerefMut, MulAssign}};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct Array1D<const N: usize> {
    pub array: Box<[f32; N]>
}
impl<const N: usize> Default for Array1D<N> {
    fn default() -> Self {
        let array: Box<std::mem::MaybeUninit<[f32; N]>> = Box::new_uninit();
        for offset in 0..N {
            unsafe { std::ptr::write(array.as_ptr().cast::<f32>().cast_mut().add(offset), 0.0) };
        }
        let array = unsafe { array.assume_init() };
        Self { array }
    }
}
impl<const N: usize> Array1D<N> {
    pub fn new() -> Self {
        Self::default()
    }
}
impl<const N: usize> AsRef<[f32; N]> for Array1D<N> {
    fn as_ref(&self) -> &[f32; N] {
        &self.array
    }
}
impl<const N: usize> Borrow<[f32; N]> for Array1D<N> {
    fn borrow(&self) -> &[f32; N] {
        &self.array
    }
}
impl<const N: usize> Deref for Array1D<N> {
    type Target = [f32; N];

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}
impl<const N: usize> DerefMut for Array1D<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}
impl<const N: usize> AddAssign<Array1D<N>> for Array1D<N> {
    fn add_assign(&mut self, rhs: Array1D<N>) {
        for (x, y) in self.array.iter_mut().zip(rhs.array.iter()) {
            *x += *y;
        }
    }
}
impl<const N: usize> AddAssign<f32> for Array1D<N> {
    fn add_assign(&mut self, rhs: f32) {
        for x in self.array.iter_mut() {
            *x += rhs;
        }
    }
}
impl<const N: usize> MulAssign<Array1D<N>> for Array1D<N> {
    fn mul_assign(&mut self, rhs: Array1D<N>) {
        for (x, y) in self.array.iter_mut().zip(rhs.array.iter()) {
            *x *= *y;
        }
    }
}
impl<const N: usize> MulAssign<f32> for Array1D<N> {
    fn mul_assign(&mut self, rhs: f32) {
        for x in self.array.iter_mut() {
            *x *= rhs;
        }
    }
}
impl<const N: usize> From<&[f32]> for Array1D<N> {
    fn from(value: &[f32]) -> Self {
        assert_eq!(value.len(), N);
        let mut new = Self::default();
        new.as_mut_slice().copy_from_slice(value);
        new
    }
}
impl<const N: usize> From<&Array1D<N>> for Vec<f32> {
    fn from(value: &Array1D<N>) -> Vec<f32> {
        let mut new = Vec::with_capacity(N);
        for x in value.as_ref() {
            new.push(*x);
        }
        new
    }
}
#[cfg(feature = "serde")]
impl<const N: usize> Serialize for Array1D<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        Vec::<f32>::from(self).serialize(serializer)
    }
}
#[cfg(feature = "serde")]
impl<'de, const N: usize> Deserialize<'de> for Array1D<N>  {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {
        Vec::<f32>::deserialize(deserializer).map(|x| Self::from(x.as_slice()))
    }
}

#[derive(Clone)]
pub struct Array2D<const X: usize, const Y: usize> {
    pub array: Box<[[f32; X]; Y]>
}
impl<const X: usize, const Y: usize> Default for Array2D<X, Y> {
    fn default() -> Self {
        let array: Box<std::mem::MaybeUninit<[[f32; X]; Y]>> = Box::new_uninit();
        for y in 0..Y {
            for x in 0..X {
                unsafe { std::ptr::write(array.as_ptr().cast::<f32>().cast_mut().add(x + y * X), 0.0) };
            }
        }
        let array = unsafe { array.assume_init() };
        Self { array }
    }
}
impl<const X: usize, const Y: usize> Array2D<X, Y> {
    pub fn new() -> Self {
        Self::default()
    }
}
impl<const X: usize, const Y: usize> AsRef<[[f32; X]; Y]> for Array2D<X, Y> {
    fn as_ref(&self) -> &[[f32; X]; Y] {
        &self.array
    }
}
impl<const X: usize, const Y: usize> Borrow<[[f32; X]; Y]> for Array2D<X, Y> {
    fn borrow(&self) -> &[[f32; X]; Y] {
        &self.array
    }
}
impl<const X: usize, const Y: usize> Deref for Array2D<X, Y> {
    type Target = [[f32; X]; Y];

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}
impl<const X: usize, const Y: usize> DerefMut for Array2D<X, Y> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}
impl<const X: usize, const Y: usize> AddAssign<Array2D<X, Y>> for Array2D<X, Y> {
    fn add_assign(&mut self, rhs: Array2D<X, Y>) {
        for (row_x, row_y) in self.array.iter_mut().zip(rhs.array.iter()) {
            for (x, y) in row_x.iter_mut().zip(row_y) {
                *x += *y;
            }
        }
    }
}
impl<const X: usize, const Y: usize> AddAssign<f32> for Array2D<X, Y> {
    fn add_assign(&mut self, rhs: f32) {
        for row in self.array.iter_mut() {
            for element in row {
                *element += rhs;
            }
        }
    }
}
impl<const X: usize, const Y: usize> MulAssign<Array2D<X, Y>> for Array2D<X, Y> {
    fn mul_assign(&mut self, rhs: Array2D<X, Y>) {
        for (row_x, row_y) in self.array.iter_mut().zip(rhs.array.iter()) {
            for (x, y) in row_x.iter_mut().zip(row_y) {
                *x *= *y;
            }
        }
    }
}
impl<const X: usize, const Y: usize> MulAssign<f32> for Array2D<X, Y> {
    fn mul_assign(&mut self, rhs: f32) {
        for row in self.array.iter_mut() {
            for element in row {
                *element *= rhs;
            }
        }
    }
}
impl<const X: usize, const Y: usize> From<&Vec<Vec<f32>>> for Array2D<X, Y> {
    fn from(value: &Vec<std::vec::Vec<f32>>) -> Self {
        assert_eq!(Y, value.len());
        for row in value {
            assert_eq!(row.len(), X);
        }
        let mut new = Self::default();
        for (row_x, row_y) in new.iter_mut().zip(value) {
            for (x, y) in row_x.iter_mut().zip(row_y) {
                *x = *y;
            }
        }
        new
    }
}
impl<const X: usize, const Y: usize> From<&Array2D<X, Y>> for Vec<Vec<f32>> {
    fn from(value: &Array2D<X, Y>) -> Self {
        let mut new = Vec::with_capacity(Y);
        for row in value.iter() {
            new.push(Vec::with_capacity(X));
            // Safety: we just pushed a vector
            let last = unsafe { new.last_mut().unwrap_unchecked() };
            for element in row {
                last.push(*element);
            }
        }
        new
    }
}
#[cfg(feature = "serde")]
impl<const X: usize, const Y: usize> Serialize for Array2D<X, Y> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        Vec::<Vec<f32>>::from(self).serialize(serializer)
    }
}
#[cfg(feature = "serde")]
impl<'de, const X: usize, const Y: usize> Deserialize<'de> for Array2D<X, Y>  {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de> {
        Vec::<Vec<f32>>::deserialize(deserializer).map(|x| Self::from(&x))
    }
}

#[test]
fn huge_array_test() {
    let array: Array1D<2000000> = Array1D::new();
    for x in array.as_ref() {
        assert_eq!(*x, 0.0);
    }
}