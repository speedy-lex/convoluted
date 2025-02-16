use std::fs;

pub fn get_mnist_train() -> (Vec<[f32; 28*28]>, Vec<usize>) {
    let mut data = Vec::with_capacity(60_000);
    let mut labels = Vec::with_capacity(60_000);
    let mut n = 0;
    for x in fs::read_to_string("examples\\mnist\\mnist_train.csv").unwrap().split('\n') {
        labels.push(x[0..1].parse::<usize>().unwrap());
        data.push([0.0; 28*28]);
        for (i, pixel) in x.split(',').skip(1).enumerate() {
            data.last_mut().unwrap()[i] = (pixel.parse::<u8>().unwrap() as f32)/255.0;
        }
        n+=1;
        if n % 20000==0 {
            println!("{}% loading mnist", n/600)
        }
    }
    println!("loaded mnist!");
    (data, labels)
}
pub fn get_mnist_test() -> (Vec<[f32; 28*28]>, Vec<usize>) {
    let mut data = Vec::with_capacity(10_000);
    let mut labels = Vec::with_capacity(10_000);
    let mut n = 0;
    for x in fs::read_to_string("examples\\mnist\\mnist_test.csv").unwrap().split('\n') {
        labels.push(x[0..1].parse::<usize>().unwrap());
        data.push([0.0; 28*28]);
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