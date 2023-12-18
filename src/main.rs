use anyhow;
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample, SizedSample,
};
use std::{
    str::FromStr,
    sync::atomic::{AtomicPtr, AtomicUsize},
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Chart, Dataset, GraphType, Paragraph},
    widgets::{Block, Borders},
    Frame, Terminal,
};

use ratatui::style::Stylize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

const RB_SIZE: usize = 200;

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
    // setup terminal
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut samples_l = [0.0; RB_SIZE];
    let samples_index = Arc::new(AtomicUsize::new(0));
    let samples_index_clone = Arc::clone(&samples_index);

    let samples_l_ptr = Arc::new(AtomicPtr::<f32>::new(samples_l.as_mut_ptr()));
    let samples_l_ptr_clone = Arc::clone(&samples_l_ptr);

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
    let info: String = format!("{:?} {:?}", device.name()?.clone(), config.clone());

    let t = Arc::new(AtomicU32::new(0));

    let audio_thread = thread::spawn(move || {
        match config.sample_format() {
            cpal::SampleFormat::I8 => run::<i8>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            cpal::SampleFormat::I16 => run::<i16>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            // cpal::SampleFormat::I24 => run::<I24>(&device, &config.into()),
            cpal::SampleFormat::I32 => run::<i32>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            // cpal::SampleFormat::I48 => run::<I48>(&device, &config.into()),
            cpal::SampleFormat::I64 => run::<i64>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            cpal::SampleFormat::U8 => run::<u8>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            cpal::SampleFormat::U16 => run::<u16>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            // cpal::SampleFormat::U24 => run::<U24>(&device, &config.into()),
            cpal::SampleFormat::U32 => run::<u32>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            // cpal::SampleFormat::U48 => run::<U48>(&device, &config.into()),
            cpal::SampleFormat::U64 => run::<u64>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            cpal::SampleFormat::F32 => run::<f32>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            cpal::SampleFormat::F64 => run::<f64>(
                &device,
                &config.into(),
                t,
                code,
                sr,
                samples_l_ptr_clone,
                samples_index_clone,
            ),
            sample_format => panic!("Unsupported sample format '{sample_format}'"),
        }
    });

    let tick_rate = Duration::from_millis(16);

    let res = run_app(&mut terminal, tick_rate, samples_l_ptr, samples_index, info);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    let _ = audio_thread.join().unwrap();

    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    tick_rate: Duration,
    samples_l_ptr: Arc<AtomicPtr<f32>>,
    sampels_index: Arc<AtomicUsize>,
    info: String,
) -> std::io::Result<()> {
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| {
            ui(
                f,
                &samples_l_ptr,
                // &samples_r_ptr,
                &sampels_index,
                &info,
            )
        })?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if let KeyCode::Esc = key.code {
                    return Ok(());
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // app.on_tick();
            last_tick = Instant::now();
        }
    }
}

fn ui(
    f: &mut Frame,
    samples_l: &Arc<AtomicPtr<f32>>, // block step length
    frame_index: &Arc<AtomicUsize>,
    info: &str,
) {
    let mut data = [0.0; RB_SIZE];
    // let mut data2 = [0.0; RB_SIZE];
    let ptr = samples_l.load(Ordering::Acquire);
    // let ptr2 = samples_r.load(Ordering::Acquire);

    let mut idx = frame_index.load(Ordering::Acquire);

    for i in 0..RB_SIZE {
        data[RB_SIZE - 1 - i] = unsafe { ptr.add(idx).read() };
        // data2[RB_SIZE - 1 - i] = unsafe { ptr2.add(idx).read() };
        if idx == 0 {
            idx = RB_SIZE - 1; // read from the tail
        } else {
            idx -= 1;
        }
    }

    let left: Vec<(f64, f64)> = data
        .into_iter()
        .enumerate()
        .map(|(x, y)| (x as f64, y as f64))
        .collect();

    let size = f.size();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(10), Constraint::Percentage(90)].as_ref())
        .split(size);

    f.render_widget(
        Paragraph::new("press esc to exit tui".red().on_white().bold()),
        chunks[0],
    );

    let x_labels = vec![Span::styled(
        format!("[0, 200]"),
        Style::default().add_modifier(Modifier::BOLD),
    )];
    let datasets = vec![Dataset::default()
        .name("audio data")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Blue))
        .data(&left)];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .title(Span::styled(
                    info.replace("SupportedStreamConfig", ""),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::NONE),
        )
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(x_labels)
                .bounds([0., 200.]),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().fg(Color::Gray))
                .labels(vec![
                    Span::styled("-1", Style::default().add_modifier(Modifier::BOLD)),
                    Span::raw("0"),
                    Span::styled("1", Style::default().add_modifier(Modifier::BOLD)),
                ])
                .bounds([-1., 1.]),
        );
    f.render_widget(chart, chunks[1]);
}

pub fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    arct: Arc<AtomicU32>,
    code: String,
    sr: u32,
    samples_l_ptr_clone: Arc<AtomicPtr<f32>>,
    samples_index_clone: Arc<AtomicUsize>,
) -> Result<(), anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let sample_rate = config.sample_rate.0 as u32;
    let channels = config.channels as usize;

    // let samples_l_ptr_clone = options.0;
    // let samples_index_clone = options.1;
    // let _capacity = options.6;

    let counter = Arc::new(AtomicU32::new(0));
    let mut next_value = move || {
        let count = counter.fetch_add(1, Ordering::SeqCst);
        let t = if count % (sample_rate / sr) as u32 == 0 {
            arct.fetch_add(1, Ordering::SeqCst)
        } else {
            arct.load(Ordering::SeqCst)
        };
        let code = code.replace("t", &format!("{}", t));
        let result = eval(&code).unwrap_or(0);
        (result % 256) as f32 / 255.0 * 2.0 - 1.0
    };

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let samples_left_ptr = samples_l_ptr_clone.load(Ordering::SeqCst);
            write_data(
                data,
                channels,
                &mut next_value,
                samples_left_ptr,
                &samples_index_clone,
            );
        },
        err_fn,
        None,
    )?;
    stream.play()?;

    // std::thread::sleep(std::time::Duration::from_millis(1000));

    loop {}
}

fn write_data<T>(
    output: &mut [T],
    channels: usize,
    next_sample: &mut dyn FnMut() -> f32,
    samples_l: *mut f32,
    samples_index: &AtomicUsize,
) where
    T: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let samples_i = samples_index.load(Ordering::SeqCst);
        let s = next_sample();
        let value: T = T::from_sample(s);
        for sample in frame.iter_mut() {
            *sample = value;
        }

        unsafe {
            samples_l.add(samples_i).write(s);
        }
        samples_index.store((samples_i + 1) % 200, Ordering::SeqCst);
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
