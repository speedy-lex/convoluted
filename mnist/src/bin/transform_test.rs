use raylib::prelude::*;

fn main() {
    let train = mnist::get_mnist_train().0;

    let (mut rl, rt) = init()
    .size(28*20, 28*20)
    .build();

    let mut idx = 0;
    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_ENTER) {
            idx += 1;
        }
        let mut d = rl.begin_drawing(&rt);
        d.clear_background(Color::new(16, 16, 16, 255));

        for y in 0..28 {
            for x in 0..28 {
                let col = (train[idx][x + 28*y] * 256.0).floor() as u8;
                d.draw_rectangle((x*20) as i32, (y*20) as i32, 20, 20, Color::new(col, col, col, 255));
            }
        }
    }
}