use bytebeat::eval;
use anyhow;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SizedSample,
};
use clap;

use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicU32};

const CODE1: &str = "(t>>7)*(t>>9)|t>>2";
const CODE: &str = CODE1;
const SR: u32 = 44100;

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();

    let device = host.default_output_device()
    .expect("failed to find output device");
    println!("Output device: {}", device.name()?);

    let config = device.default_output_config().unwrap();
    println!("Default output config: {:?}", config);

    let t = Arc::new(AtomicU32::new(0));

    match config.sample_format() {
        cpal::SampleFormat::I8 => run::<i8>(&device, &config.into(), t),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), t),
        // cpal::SampleFormat::I24 => run::<I24>(&device, &config.into()),
        cpal::SampleFormat::I32 => run::<i32>(&device, &config.into(), t),
        // cpal::SampleFormat::I48 => run::<I48>(&device, &config.into()),
        cpal::SampleFormat::I64 => run::<i64>(&device, &config.into(), t),
        cpal::SampleFormat::U8 => run::<u8>(&device, &config.into(), t),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), t),
        // cpal::SampleFormat::U24 => run::<U24>(&device, &config.into()),
        cpal::SampleFormat::U32 => run::<u32>(&device, &config.into(), t),
        // cpal::SampleFormat::U48 => run::<U48>(&device, &config.into()),
        cpal::SampleFormat::U64 => run::<u64>(&device, &config.into(), t),
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), t),
        cpal::SampleFormat::F64 => run::<f64>(&device, &config.into(), t),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
}

pub fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig, arct: Arc<AtomicU32>) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate.0 as u32;
    let channels = config.channels as usize;

    // Produce a sinusoid of maximum amplitude.
    // let mut sample_clock = 0f32;
    let mut next_value = move || {
        let t = arct.load(Ordering::SeqCst);
        let code = CODE.replace("t", &format!("{}", t));
        arct.store(t + (sample_rate/SR) as u32, Ordering::SeqCst);
        let result = eval(&code).unwrap_or(0);
        // println!("{}", (result % 256) as f32 / 255.0 * 2.0 - 1.0);
        (result % 256) as f32 / 255.0 * 2.0 - 1.0
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

    // std::thread::sleep(std::time::Duration::from_millis(1000));

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
}