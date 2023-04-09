## bytebeat-rs

**WARNING** ⚠️ Lower your volume when play the video below❗️

https://user-images.githubusercontent.com/35621141/230710706-76a2a4a8-a4e9-4819-b755-802103b76ba4.mp4

### usage

installation:

`cargo install bytebeat`

then in your terminal:

```sh
bytebeat "(t>>7)*(t>>9)|t>>6"
```

### basic idea on bytebeat

you have a `t` that keeps increamenting from 0, 1, 2, to u32::MAX.

in the source code:
```
let result = ((t));
t += 1;
```
where `((t))` will be replaced by the rules you define:

for example `(t>>7)*(t>>9)|t>>6` 

the output will be a float: `(result % 255) as f32 / 255.0 * 2.0 - 1.0`

what kind of tune it will make depends on the your rules:

### contribution

pull request is welcomed!