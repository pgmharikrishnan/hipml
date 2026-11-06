//! A plain host tensor: row-major, f32, shape only.
//!
//! Deliberately dumb. Layout is the contract with the kernels, so it stays
//! explicit: a `[m, k]` tensor is `m` rows of `k` contiguous values.

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Build from a shape and data, panicking when the sizes disagree.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {shape:?} needs {n} values, got {}",
            data.len()
        );
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn filled(shape: Vec<usize>, v: f32) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![v; n],
        }
    }

    /// Build from a closure over the flat index.
    pub fn from_fn(shape: Vec<usize>, mut f: impl FnMut(usize) -> f32) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: (0..n).map(&mut f).collect(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Row `i` of a 2-D tensor.
    pub fn row(&self, i: usize) -> &[f32] {
        assert_eq!(self.shape.len(), 2, "row() needs a 2-D tensor");
        let k = self.shape[1];
        &self.data[i * k..(i + 1) * k]
    }
}
