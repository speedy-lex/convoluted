use std::fs;

use convoluted::array::Array1D;
use rand::Rng;

pub fn get_mnist_train() -> (Vec<Array1D<{ 28*28 }>>, Vec<usize>) {
    let mut data = Vec::with_capacity(60_000);
    let mut labels = Vec::with_capacity(60_000);
    let mut n = 0;
    let mut src = rand::rng();
    for x in fs::read_to_string("src\\mnist_train.csv").unwrap().split('\n') {
        labels.push(x[0..1].parse::<usize>().unwrap());
        data.push(Array1D::new());
        for (i, pixel) in x.split(',').skip(1).enumerate() {
            data.last_mut().unwrap()[i] = (pixel.parse::<u8>().unwrap() as f32)/255.0;
        }
        *data.last_mut().unwrap() = transform(data.last().unwrap(), src.random::<f32>()/2.0-0.25, (src.random::<f32>()-0.5)*0.3+1.05, (src.random::<f32>()*6.0-3.0, src.random::<f32>()*6.0-3.0));
        n+=1;
        if n % 20000==0 {
            println!("{}% loading mnist", n/600)
        }
    }
    println!("loaded mnist!");
    (data, labels)
}
pub fn get_mnist_test() -> (Vec<Array1D<{ 28*28 }>>, Vec<usize>) {
    let mut data = Vec::with_capacity(10_000);
    let mut labels = Vec::with_capacity(10_000);
    let mut n = 0;
    for x in fs::read_to_string("src\\mnist_test.csv").unwrap().split('\n') {
        labels.push(x[0..1].parse::<usize>().unwrap());
        data.push(Array1D::new());
        for (i, pixel) in x.split(',').skip(1).enumerate() {
            data.last_mut().unwrap()[i] = (pixel.parse::<u8>().unwrap() as f32)/255.0;
        }
        n+=1;
        if n % 5000==0 {
            println!("{}% loading mnist test", n/100)
        }
    }
    println!("loaded mnist test!");
    (data, labels)
}

pub fn transform(img: &Array1D<{ 28*28 }>, rot: f32, scale: f32, translation: (f32, f32)) -> Array1D<{ 28*28 }> {
    let (sin, cos) = rot.sin_cos();
    let mut i = Array1D::new();
    for x in 0..(28_usize.pow(2)) {
        let (mut u, mut v) = ((x%28) as f32 - 14.0, (x/28) as f32 - 14.0);
        (u, v) = (u*cos-v*sin, u*sin+v*cos);
        (u, v) = (u/scale, v/scale);
        (u, v) = (u-translation.0+14.0, v-translation.1+14.0);
        let (u_frac, v_frac) = (u%1.0, v%1.0);
        let (u, v) = (u as usize, v as usize);
        let uv_lin = u+v*28;
        let (pix00, pix10, pix01, pix11) = (sample(img, uv_lin), sample(img, uv_lin+1), sample(img, uv_lin+28), sample(img, uv_lin+29));
        let (pix0, pix1) = (pix00*(1.0-v_frac)+pix01*v_frac, pix10*(1.0-v_frac)+pix11*v_frac);
        let pix = pix0*(1.0-u_frac)+pix1*u_frac;
        i[x]=pix;
    }
    i
}
fn sample(img: &Array1D<{ 28*28 }>, idx: usize) -> f32 {
    if idx<img.len() {
        img[idx]
    } else {
        0.0
    }
}