use anyhow;
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SizedSample,
};
use std::str::FromStr;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// bytebeat cli tool written in Rust
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// bytebeat code
    #[arg(index = 1)]
    code: String,

    /// set sr
    #[arg(short, long, default_value_t = 44100)]
    sr: u32,

    /// The audio device to use
    #[arg(short, long, default_value_t = String::from("default"))]
    device: String,

    /// Use the JACK host
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "netbsd"
        ),
        feature = "jack"
    ))]
    #[arg(short, long)]
    #[allow(dead_code)]
    jack: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let code = args.code;
    let device = args.device;
    let sr = args.sr;

    // Conditionally compile with jack if the feature is specified.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "netbsd"
        ),
        feature = "jack"
    ))]
    let host = if args.jack {
        cpal::host_from_id(cpal::available_hosts()
            .into_iter()
            .find(|id| *id == cpal::HostId::Jack)
            .expect(
                "make sure --features jack is specified. only works on OSes where jack is available",
            )).expect("jack host unavailable")
    } else {
        cpal::default_host()
    };

    #[cfg(any(
        not(any(
            target_os = "linux",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "netbsd"
        )),
        not(feature = "jack")
    ))]
    let host = cpal::default_host();

    let device = if device == "default" {
        host.default_output_device()
    } else {
        host.output_devices()?
            .find(|x| x.name().map(|y| y == device).unwrap_or(false))
    }
    .expect("failed to find output device");
    // println!("Output device: {}", device.name()?);
    let config = device.default_output_config().unwrap();

    let t = Arc::new(AtomicU32::new(0));

    match config.sample_format() {
        cpal::SampleFormat::I8 => run::<i8>(&device, &config.into(), t, code, sr),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), t, code, sr),
        // cpal::SampleFormat::I24 => run::<I24>(&device, &config.into()),
        cpal::SampleFormat::I32 => run::<i32>(&device, &config.into(), t, code, sr),
        // cpal::SampleFormat::I48 => run::<I48>(&device, &config.into()),
        cpal::SampleFormat::I64 => run::<i64>(&device, &config.into(), t, code, sr),
        cpal::SampleFormat::U8 => run::<u8>(&device, &config.into(), t, code, sr),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), t, code, sr),
        // cpal::SampleFormat::U24 => run::<U24>(&device, &config.into()),
        cpal::SampleFormat::U32 => run::<u32>(&device, &config.into(), t, code, sr),
        // cpal::SampleFormat::U48 => run::<U48>(&device, &config.into()),
        cpal::SampleFormat::U64 => run::<u64>(&device, &config.into(), t, code, sr),
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), t, code, sr),
        cpal::SampleFormat::F64 => run::<f64>(&device, &config.into(), t, code, sr),
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    }
}

pub fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    arct: Arc<AtomicU32>,
    code: String,
    sr: u32,
) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate.0 as u32;
    let channels = config.channels as usize;
    // let mut next_value = move || {
    //     let t = arct.load(Ordering::SeqCst);
    //     let code = code.replace("t", &format!("{}", t));
    //     arct.store(t + (sample_rate / sr) as u32, Ordering::SeqCst);
    //     let result = eval(&code).unwrap_or(0);

    //     (result % 256) as f32 / 255.0 * 2.0 - 1.0
    // };

    let counter = Arc::new(AtomicU32::new(0));
    let mut next_value = move || {
        let count = counter.fetch_add(1, Ordering::SeqCst);
        let t = if count % (sample_rate / sr) as u32 == 0 {
            arct.fetch_add(1, Ordering::SeqCst)
        } else {
            arct.load(Ordering::SeqCst)
        };
        let code = code.replace("t", &format!("{}", t));
        // arct.store(t + (sample_rate / sr) as u32, Ordering::SeqCst);
        let result = eval(&code).unwrap_or(0);
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

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Number(u32),
    Operator(char),
    Paren(char),
}

pub fn tokenize(expr: &str) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut num_buffer = String::new();
    let mut prev_char = '\0';

    for ch in expr.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_digit(10) {
            num_buffer.push(ch);
        } else {
            if !num_buffer.is_empty() {
                if let Ok(num) = u32::from_str(&num_buffer) {
                    tokens.push(Token::Number(num));
                }
                num_buffer.clear();
            }
            if ch == '(' || ch == ')' {
                tokens.push(Token::Paren(ch));
            } else if ch == '<' || ch == '>' {
                if prev_char == ch {
                    tokens.pop();
                    tokens.push(Token::Operator(if ch == '<' { '<' } else { '>' }));
                } else {
                    tokens.push(Token::Operator(ch));
                }
            } else {
                tokens.push(Token::Operator(ch));
            }
            prev_char = ch;
        }
    }

    if !num_buffer.is_empty() {
        if let Ok(num) = u32::from_str(&num_buffer) {
            tokens.push(Token::Number(num));
        }
    }

    tokens
}

pub fn precedence(op: &char) -> u8 {
    match op {
        '+' | '-' => 1,
        '*' | '/' => 2,
        '&' | '|' => 3,
        '^' => 4,
        '<' | '>' => 5,
        _ => 0,
    }
}

pub fn infix_to_postfix(tokens: &[Token]) -> Vec<Token> {
    let mut output: Vec<Token> = Vec::new();
    let mut stack: Vec<Token> = Vec::new();

    for token in tokens {
        match token {
            Token::Number(_) => output.push(token.clone()),
            Token::Operator(op) => {
                while let Some(Token::Operator(top_op)) = stack.last() {
                    if precedence(op) <= precedence(top_op) {
                        output.push(stack.pop().unwrap());
                    } else {
                        break;
                    }
                }
                stack.push(token.clone());
            }
            Token::Paren('(') => stack.push(token.clone()),
            Token::Paren(')') => {
                while let Some(top) = stack.pop() {
                    if top == Token::Paren('(') {
                        break;
                    }
                    output.push(top);
                }
            }
            _ => (),
        }
    }

    while let Some(top) = stack.pop() {
        output.push(top);
    }

    output
}

pub fn eval_postfix(tokens: &[Token]) -> Option<u32> {
    let mut stack: Vec<u32> = Vec::new();

    for token in tokens {
        match token {
            Token::Number(n) => stack.push(*n),
            Token::Operator(op) => {
                let rhs = stack.pop()?;
                let lhs = stack.pop()?;
                let result = match op {
                    '+' => lhs.wrapping_add(rhs),
                    '-' => lhs.wrapping_sub(rhs),
                    '*' => lhs.wrapping_mul(rhs),
                    '/' => {
                        if rhs == 0 {
                            return None;
                        }
                        lhs.wrapping_div(rhs)
                    }
                    '&' => lhs & rhs,
                    '|' => lhs | rhs,
                    '^' => lhs ^ rhs,
                    '<' => lhs << rhs,
                    '>' => lhs >> rhs,
                    _ => return None,
                };
                stack.push(result);
            }
            _ => return None,
        }
    }

    if stack.len() == 1 {
        Some(stack[0])
    } else {
        None
    }
}

pub fn eval(expr: &str) -> Option<u32> {
    let tokens = tokenize(expr);
    let postfix_tokens = infix_to_postfix(&tokens);
    eval_postfix(&postfix_tokens)
}
