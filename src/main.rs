use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn create_temp_cargo_toml(temp_dir: &Path) {
    let temp_cargo_toml_path = temp_dir.join("Cargo.toml");
    let mut temp_cargo_toml_file = fs::File::create(&temp_cargo_toml_path).expect("Unable to create temporary Cargo.toml file.");
    
    let cargo_toml_content = r#"[package]
name = "temp_package"
version = "0.1.0"
edition = "2021"

[dependencies]
cpal = "0.15.2"
anyhow = "*"

[profile.release]
opt-level = 0
codegen-units = 16
lto = false
incremental = true
"#;

    temp_cargo_toml_file.write_all(cargo_toml_content.as_bytes()).expect("Unable to write to temporary Cargo.toml file.");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: dynamic_compile_tool <source_code>");
        return;
    }

    let code = TEMPLATE.replace("((t))", &args[1]);

    let temp_dir = tempfile::tempdir().unwrap();
    println!("Temporary directory: {:?}", temp_dir.path());
    let temp_src_path = temp_dir.path().join("src").join("main.rs");
    fs::create_dir(temp_dir.path().join("src")).expect("Unable to create src directory in temporary directory.");

    create_temp_cargo_toml(temp_dir.path());

    let mut temp_src_file = fs::File::create(&temp_src_path).expect("Unable to create temporary source file.");
    temp_src_file.write_all(code.as_bytes()).expect("Unable to write code to temporary source file.");

    let _status = Command::new("cargo")
        .arg("run")
        .arg("--release")
        .current_dir(temp_dir.path())
        .status()
        .expect("Failed to cargo run.");

}

const TEMPLATE: &str = r#"use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SizedSample,
};

use anyhow;

fn main() -> anyhow::Result<()> {

    let host = cpal::default_host();

    let device = host.default_output_device()
    .expect("failed to find output device");
    println!("Output device: {}", device.name().unwrap());
    
    let config = device.default_output_config().unwrap();

    match config.sample_format() {
        cpal::SampleFormat::I8 => run::<i8>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        // cpal::SampleFormat::I24 => run::<I24>(&device, &config.into()),
        cpal::SampleFormat::I32 => run::<i32>(&device, &config.into()),
        // cpal::SampleFormat::I48 => run::<I48>(&device, &config.into()),
        cpal::SampleFormat::I64 => run::<i64>(&device, &config.into()),
        cpal::SampleFormat::U8 => run::<u8>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
        // cpal::SampleFormat::U24 => run::<U24>(&device, &config.into()),
        cpal::SampleFormat::U32 => run::<u32>(&device, &config.into()),
        // cpal::SampleFormat::U48 => run::<U48>(&device, &config.into()),
        cpal::SampleFormat::U64 => run::<u64>(&device, &config.into()),
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::F64 => run::<f64>(&device, &config.into()),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
}


pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    // let sample_rate = config.sample_rate.0 as u32;
    let channels = config.channels as usize;
    
    let mut t = 0_u32;

    let mut next_value = move || {
        let result = ((t));
        t += 1;
        (result % 255) as f32 / 255.0 * 2.0 - 1.0
    };

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_data(data, channels, &mut next_value)
        },
        err_fn,
        None,
    )?;
    stream.play()?;
    loop {}
}

fn write_data<T>(output: &mut [T], channels: usize, next_sample: &mut dyn FnMut() -> f32)
where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let value: T = T::from_sample(next_sample());
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}"#;