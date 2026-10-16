//! The parametric equalizer: the curve a user edits and the settings keep, the response that
//! curve promises, and the presets. `engine` runs the filters and `shared` carries a curve from
//! its one writer to every output.

mod design;

use std::f64::consts::SQRT_2;
use std::num::NonZero;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use anyhow::{Result, ensure};
use rodio::Source;
use rodio::source::SeekError;

use crate::audio::{Format, Frames};
use design::{Biquad, Shape, phi};

/// How many filters a curve holds at most, which is also how many the audio thread reserves.
pub const MAX_FILTERS: usize = 20;
/// The frequencies a filter accepts, in hertz.
pub const FREQUENCY_RANGE: RangeInclusive<f64> = 10.0..=24_000.0;
/// The gains a peak or shelf accepts, in decibels.
pub const GAIN_RANGE: RangeInclusive<f64> = -24.0..=24.0;
/// The Q a filter accepts. The top is Equalizer APO's default for a notch, with room over it.
pub const Q_RANGE: RangeInclusive<f64> = 0.05..=50.0;
/// The bandwidth in octaves a filter accepts.
pub const OCTAVES_RANGE: RangeInclusive<f64> = 0.01..=6.0;
/// The preamp a user may set by hand, in decibels.
pub const PREAMP_RANGE: RangeInclusive<f64> = -30.0..=12.0;

/// The rates the automatic preamp is read at, so a curve that is safe at one playing rate is
/// safe at every other. The same filter reads higher near the Nyquist frequency of a higher rate.
const RATES: [u32; 6] = [44_100, 48_000, 88_200, 96_000, 176_400, 192_000];
/// The grid the automatic preamp reads the response on, from 20 Hz up in one percent steps,
/// plus every filter's centre, without which a narrow peak between grid points is under-read.
const GRID_START: f64 = 20.;
const GRID_RATIO: f64 = 1.01;
const GRID_TOP: f64 = 24_000.;
/// The centre frequencies of the graphic equalizer the presets and the sliders are drawn on,
/// one per octave, and the Q that makes adjacent bands meet at their half gain points.
const BAND_FREQUENCIES: [f64; 10] = [
    31.25, 62.5, 125., 250., 500., 1_000., 2_000., 4_000., 8_000., 16_000.,
];
const BAND_Q: f64 = SQRT_2;

/// What a filter does to the band around its frequency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Peak,
    LowShelf,
    HighShelf,
    LowPass,
    HighPass,
    Notch,
    BandPass,
    AllPass,
}

impl Kind {
    pub const ALL: [Self; 8] = [
        Self::Peak,
        Self::LowShelf,
        Self::HighShelf,
        Self::LowPass,
        Self::HighPass,
        Self::Notch,
        Self::BandPass,
        Self::AllPass,
    ];

    /// The name the settings file spells the kind with.
    pub fn id(self) -> &'static str {
        match self {
            Self::Peak => "peak",
            Self::LowShelf => "low-shelf",
            Self::HighShelf => "high-shelf",
            Self::LowPass => "low-pass",
            Self::HighPass => "high-pass",
            Self::Notch => "notch",
            Self::BandPass => "band-pass",
            Self::AllPass => "all-pass",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.id() == id)
    }

    /// Whether the kind boosts or cuts. The others only shape, and their gain is kept at zero.
    pub fn has_gain(self) -> bool {
        matches!(self, Self::Peak | Self::LowShelf | Self::HighShelf)
    }

    fn is_shelf(self) -> bool {
        matches!(self, Self::LowShelf | Self::HighShelf)
    }
}

/// How wide a filter is. Octaves are kept as written, because Equalizer APO converts them to a Q
/// at the playing rate, so converting at import would be wrong at every other rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Q(f64),
    Octaves(f64),
}

impl Width {
    pub fn value(self) -> f64 {
        match self {
            Self::Q(value) | Self::Octaves(value) => value,
        }
    }
}

/// One filter of a curve. Only `new` makes one, so a filter that exists is in range and the audio
/// thread can trust it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Filter {
    kind: Kind,
    frequency: f64,
    gain: f64,
    width: Width,
    enabled: bool,
}

impl Filter {
    /// A filter in range, or why it is not. A shelf takes a Q, never octaves, since APO ignores
    /// octaves on shelves and so do the files Sonora imports. A kind without a gain keeps zero.
    pub fn new(kind: Kind, frequency: f64, gain: f64, width: Width, enabled: bool) -> Result<Self> {
        ensure!(
            FREQUENCY_RANGE.contains(&frequency),
            "frequency {frequency} Hz is outside {FREQUENCY_RANGE:?}"
        );
        ensure!(
            GAIN_RANGE.contains(&gain),
            "gain {gain} dB is outside {GAIN_RANGE:?}"
        );
        match width {
            Width::Q(q) => ensure!(Q_RANGE.contains(&q), "Q {q} is outside {Q_RANGE:?}"),
            Width::Octaves(octaves) => {
                ensure!(
                    !kind.is_shelf(),
                    "a shelf takes a Q, not a width in octaves"
                );
                ensure!(
                    OCTAVES_RANGE.contains(&octaves),
                    "width {octaves} octaves is outside {OCTAVES_RANGE:?}"
                );
            }
        }
        Ok(Self {
            kind,
            frequency,
            gain: match kind.has_gain() {
                true => gain,
                false => 0.,
            },
            width,
            enabled,
        })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn gain(&self) -> f64 {
        self.gain
    }

    pub fn width(&self) -> Width {
        self.width
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn with_gain(self, gain: f64) -> Result<Self> {
        Self::new(self.kind, self.frequency, gain, self.width, self.enabled)
    }
}

/// How the level is lowered to make room for a curve's boosts: by the curve's own highest
/// boost, or by what the user or an imported file says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preamp {
    Auto,
    Manual(f64),
}

/// A preamp and up to `MAX_FILTERS` filters, applied in order. One curve plays on every channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    preamp: Preamp,
    filters: Vec<Filter>,
}

impl Curve {
    pub fn new(preamp: Preamp, filters: Vec<Filter>) -> Result<Self> {
        ensure!(
            filters.len() <= MAX_FILTERS,
            "a curve holds at most {MAX_FILTERS} filters, not {}",
            filters.len()
        );
        if let Preamp::Manual(db) = preamp {
            ensure!(
                PREAMP_RANGE.contains(&db),
                "preamp {db} dB is outside {PREAMP_RANGE:?}"
            );
        }
        Ok(Self { preamp, filters })
    }

    /// The graphic equalizer's curve: one octave wide peak per band at `gains` decibels, lowest
    /// band first, with the preamp automatic. Gains beyond the ten bands are dropped, missing
    /// ones are flat, and each is clamped into range.
    pub fn bands(gains: &[f64]) -> Self {
        let filters = BAND_FREQUENCIES
            .iter()
            .enumerate()
            .map(|(band, frequency)| {
                let gain = gains.get(band).copied().unwrap_or(0.);
                let gain = match gain.is_finite() {
                    true => gain.clamp(*GAIN_RANGE.start(), *GAIN_RANGE.end()),
                    false => 0.,
                };
                Filter::new(Kind::Peak, *frequency, gain, Width::Q(BAND_Q), true)
                    .expect("a band is in range")
            })
            .collect();
        Self {
            preamp: Preamp::Auto,
            filters,
        }
    }

    pub fn preamp(&self) -> Preamp {
        self.preamp
    }

    pub fn filters(&self) -> &[Filter] {
        &self.filters
    }

    /// The filters to edit in place. A curve has no invariant beyond its length, and a filter
    /// stays valid because only `Filter::new` makes one.
    pub fn filters_mut(&mut self) -> &mut [Filter] {
        &mut self.filters
    }

    /// The preamp in decibels the engine applies. An automatic one is lowered by the curve's
    /// highest boost at any rate it may play at, so no steady tone leaves above full scale.
    pub fn preamp_db(&self) -> f64 {
        match self.preamp {
            Preamp::Manual(db) => db,
            Preamp::Auto => {
                let peak = RATES
                    .into_iter()
                    .map(|rate| self.response(rate).peak())
                    .fold(0., f64::max);
                (0. - peak).min(0.)
            }
        }
    }

    /// What the enabled filters do when playing at `rate`, without the preamp. This is the
    /// response the audio thread reproduces, so a graph drawn from it shows what is heard.
    pub fn response(&self, rate: u32) -> Response {
        let rate = f64::from(rate);
        let biquads = self
            .filters
            .iter()
            .filter(|filter| filter.enabled)
            .map(|filter| Biquad::of(&Shape::of(filter), rate))
            .collect();
        let centres = self
            .filters
            .iter()
            .filter(|filter| filter.enabled)
            .map(|filter| filter.frequency)
            .collect();
        Response {
            biquads,
            centres,
            rate,
        }
    }
}

/// A curve's steady-state magnitude at one rate, ready to read at any frequency.
#[derive(Clone, Debug)]
pub struct Response {
    biquads: Vec<Biquad>,
    centres: Vec<f64>,
    rate: f64,
}

impl Response {
    /// The gain at `frequency` in decibels.
    pub fn db(&self, frequency: f64) -> f64 {
        let phi = phi(frequency, self.rate);
        self.biquads.iter().map(|biquad| biquad.db(phi)).sum()
    }

    /// The highest gain over the band the rate carries, read on the grid and at each centre.
    fn peak(&self) -> f64 {
        let top = (self.rate / 2.).min(GRID_TOP);
        let grid = std::iter::successors(Some(GRID_START), |f| Some(f * GRID_RATIO))
            .take_while(|f| *f <= top);
        let centres = self.centres.iter().copied().filter(|f| *f <= top);
        grid.chain(centres)
            .map(|f| self.db(f))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// A named curve a user can pick instead of shaping each band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Flat,
    BassBoost,
    BassReducer,
    TrebleBoost,
    Vocal,
    Rock,
    Pop,
    Jazz,
    Classical,
    Electronic,
    Acoustic,
    Loudness,
}

impl Preset {
    pub const ALL: [Self; 12] = [
        Self::Flat,
        Self::BassBoost,
        Self::BassReducer,
        Self::TrebleBoost,
        Self::Vocal,
        Self::Rock,
        Self::Pop,
        Self::Jazz,
        Self::Classical,
        Self::Electronic,
        Self::Acoustic,
        Self::Loudness,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::BassBoost => "bass-boost",
            Self::BassReducer => "bass-reducer",
            Self::TrebleBoost => "treble-boost",
            Self::Vocal => "vocal",
            Self::Rock => "rock",
            Self::Pop => "pop",
            Self::Jazz => "jazz",
            Self::Classical => "classical",
            Self::Electronic => "electronic",
            Self::Acoustic => "acoustic",
            Self::Loudness => "loudness",
        }
    }

    /// The preset as a curve of ten octave wide peaks, the same curve the graphic equalizer
    /// played, now with an automatic preamp under it.
    pub fn curve(self) -> Curve {
        Curve::bands(&self.band_gains())
    }

    pub fn gains(self) -> Gains {
        self.band_gains().map(|gain| gain as f32)
    }

    /// The preset these gains spell, if they match one to within a tenth of a decibel.
    pub fn matching(gains: &Gains) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| {
            preset
                .gains()
                .iter()
                .zip(gains)
                .all(|(want, have)| (want - have).abs() < 0.05)
        })
    }

    fn band_gains(self) -> [f64; 10] {
        match self {
            Self::Flat => [0.; 10],
            Self::BassBoost => [6., 5., 4., 2.5, 1., 0., 0., 0., 0., 0.],
            Self::BassReducer => [-6., -5., -4., -2.5, -1., 0., 0., 0., 0., 0.],
            Self::TrebleBoost => [0., 0., 0., 0., 0., 1., 2.5, 4., 5., 6.],
            Self::Vocal => [-2., -3., -3., 1., 4., 4., 3., 1.5, 0., -1.5],
            Self::Rock => [5., 4., 3., 1., -0.5, -1., 0.5, 2.5, 3.5, 4.5],
            Self::Pop => [-1.5, -1., 0., 2., 4., 4., 2., 0., -1., -1.5],
            Self::Jazz => [4., 3., 1.5, 2., -1.5, -1.5, 0., 1.5, 3., 4.],
            Self::Classical => [4.5, 3.5, 3., 2.5, -1.5, -1.5, 0., 2.5, 3.5, 4.],
            Self::Electronic => [4.5, 4., 1., 0., -2., 2., 1., 1.5, 4., 5.],
            Self::Acoustic => [5., 4.5, 3.5, 1., 2., 1.5, 3.5, 4., 3.5, 2.],
            Self::Loudness => [6., 4., 0., 0., -2., 0., -1., -5., 5., 1.],
        }
    }
}

/// How many bands the equalizer has, one per octave from 31 Hz to 16 kHz.
pub const BANDS: usize = 10;
/// The centre frequency of each band, in hertz.
pub const FREQUENCIES: [f32; BANDS] = [
    31.25, 62.5, 125., 250., 500., 1_000., 2_000., 4_000., 8_000., 16_000.,
];
/// The lowest gain a band accepts, in decibels.
pub const MIN_GAIN: f32 = -12.;
/// The highest gain a band accepts, in decibels.
pub const MAX_GAIN: f32 = 12.;

/// Per band gains, in decibels, from the lowest band to the highest.
pub type Gains = [f32; BANDS];

/// One octave wide bands. Q relates the centre frequency to the bandwidth at the half gain
/// points, and the square root of two is what makes adjacent octave bands meet there.
const GRAPHIC_Q: f32 = std::f32::consts::SQRT_2;
/// How long a gain change takes to settle, so a dragged slider never clicks.
const RAMP: Duration = Duration::from_millis(40);
/// A band this close to the Nyquist frequency is left flat rather than aliased.
const NYQUIST_MARGIN: f32 = 0.95;
/// How many channels' delay lines are reserved up front, so a format change on the audio thread
/// allocates nothing up to this many.
const CHANNELS: usize = 8;

/// Clamps every band into the accepted range and turns anything that is not a number flat.
pub fn clamped(gains: &Gains) -> Gains {
    gains.map(|gain| match gain.is_finite() {
        true => gain.clamp(MIN_GAIN, MAX_GAIN),
        false => 0.,
    })
}

struct Shared {
    enabled: AtomicBool,
    gains: [AtomicU32; BANDS],
    /// Bumped on every write so an output reads the bands only when they moved.
    generation: AtomicU32,
}

/// The equalizer settings as the audio thread sees them: lock free, read once per frame.
/// Cloning shares the same values, so every output an engine opens follows one setting.
#[derive(Clone)]
pub struct Equalizer(Arc<Shared>);

impl Equalizer {
    pub fn new(enabled: bool, gains: &Gains) -> Self {
        let gains = clamped(gains);
        Self(Arc::new(Shared {
            enabled: AtomicBool::new(enabled),
            gains: gains.map(|gain| AtomicU32::new(gain.to_bits())),
            generation: AtomicU32::new(0),
        }))
    }

    pub fn enabled(&self) -> bool {
        self.0.enabled.load(Ordering::Relaxed)
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.0.enabled.store(enabled, Ordering::Relaxed);
        self.0.generation.fetch_add(1, Ordering::Release);
    }

    pub fn gains(&self) -> Gains {
        std::array::from_fn(|band| f32::from_bits(self.0.gains[band].load(Ordering::Relaxed)))
    }

    pub fn set_gains(&self, gains: &Gains) {
        for (slot, gain) in self.0.gains.iter().zip(clamped(gains)) {
            slot.store(gain.to_bits(), Ordering::Relaxed);
        }
        self.0.generation.fetch_add(1, Ordering::Release);
    }

    /// The gains the output should apply right now: flat while disabled.
    fn effective(&self) -> Gains {
        match self.enabled() {
            true => self.gains(),
            false => [0.; BANDS],
        }
    }

    fn generation(&self) -> u32 {
        self.0.generation.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for Equalizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Equalizer")
            .field("enabled", &self.enabled())
            .field("gains", &self.gains())
            .finish()
    }
}

impl Default for Equalizer {
    fn default() -> Self {
        Self::new(false, &[0.; BANDS])
    }
}

/// One peaking biquad in transposed direct form II. `a0` is normalised away.
#[derive(Clone, Copy, Default)]
struct Peaking {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Peaking {
    /// The RBJ cookbook peaking filter, or a pass through when the band sits above what the
    /// rate can carry.
    fn band(frequency: f32, gain_db: f32, rate: f32) -> Self {
        if gain_db == 0. || frequency >= rate / 2. * NYQUIST_MARGIN {
            return Self::identity();
        }
        let a = 10f32.powf(gain_db / 40.);
        let w0 = std::f32::consts::TAU * frequency / rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2. * GRAPHIC_Q);

        let a0 = 1. + alpha / a;
        Self {
            b0: (1. + alpha * a) / a0,
            b1: (-2. * cos) / a0,
            b2: (1. - alpha * a) / a0,
            a1: (-2. * cos) / a0,
            a2: (1. - alpha / a) / a0,
        }
    }

    fn identity() -> Self {
        Self {
            b0: 1.,
            ..Self::default()
        }
    }
}

/// The two delay elements one biquad keeps for one channel.
#[derive(Clone, Copy, Default)]
struct Delay {
    z1: f32,
    z2: f32,
}

impl Delay {
    #[inline]
    fn run(&mut self, coefficients: &Peaking, x: f32) -> f32 {
        let y = coefficients.b0 * x + self.z1;
        self.z1 = coefficients.b1 * x - coefficients.a1 * y + self.z2;
        self.z2 = coefficients.b2 * x - coefficients.a2 * y;
        y
    }
}

/// Runs the equalizer over an interleaved source. The gains it applies glide toward the
/// handle's values over `RAMP`, and the whole chain is skipped while every band is flat, so
/// a disabled equalizer costs nothing and changes nothing.
pub struct Equalized<I> {
    input: I,
    equalizer: Equalizer,
    seen: u32,

    current: Gains,
    target: Gains,
    step: Gains,
    frames_left: u32,
    ramp_frames: u32,

    coefficients: [Peaking; BANDS],
    /// Delay lines indexed by band, then channel.
    delays: Vec<[Delay; BANDS]>,
    /// Whether any coefficient is doing work, so a flat chain is skipped whole.
    active: bool,

    frames: Frames,
}

impl<I: Source> Equalized<I> {
    pub(crate) fn new(input: I, equalizer: Equalizer, format: Format) -> Self {
        let current = equalizer.effective();
        let seen = equalizer.generation();
        Self {
            input,
            equalizer,
            seen,
            current,
            target: current,
            step: [0.; BANDS],
            frames_left: 0,
            ramp_frames: 1,
            coefficients: [Peaking::identity(); BANDS],
            delays: Vec::with_capacity(CHANNELS),
            active: false,
            frames: Frames::new(format),
        }
    }

    /// Takes up a new rate and channel count. The delay lines reset, since what they hold
    /// belongs to another format, and every coefficient is recomputed.
    fn resync(&mut self) {
        let channels = usize::from(self.frames.channels());
        let rate = self.frames.rate();
        self.ramp_frames = (RAMP.as_secs_f64() * rate as f64).round().max(1.) as u32;
        self.frames_left = self.frames_left.min(self.ramp_frames);
        self.delays.clear();
        self.delays.resize(channels, [Delay::default(); BANDS]);
        self.recompute();
    }

    fn recompute(&mut self) {
        let rate = self.frames.rate() as f32;
        for (band, coefficients) in self.coefficients.iter_mut().enumerate() {
            *coefficients = Peaking::band(FREQUENCIES[band], self.current[band], rate);
        }
        self.active = self.current.iter().any(|gain| *gain != 0.);
    }

    /// Once per frame: picks up a new target and moves one step toward it.
    fn advance(&mut self) {
        let generation = self.equalizer.generation();
        if generation != self.seen {
            self.seen = generation;
            let target = self.equalizer.effective();
            if target != self.target {
                self.target = target;
                self.frames_left = self.ramp_frames;
                for ((step, goal), now) in self.step.iter_mut().zip(target).zip(self.current) {
                    *step = (goal - now) / self.ramp_frames as f32;
                }
            }
        }

        if self.frames_left == 0 {
            return;
        }
        self.frames_left -= 1;
        match self.frames_left {
            0 => self.current = self.target,
            _ => {
                for (now, step) in self.current.iter_mut().zip(self.step) {
                    *now += step;
                }
            }
        }
        self.recompute();
    }
}

impl<I: Source> Iterator for Equalized<I> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let (sample, place) = self.frames.pull(&mut self.input)?;
        let Some(place) = place else {
            return Some(sample);
        };

        if place.changed {
            self.resync();
        }
        if place.channel == 0 {
            self.advance();
        }

        let output = match self.active {
            true => {
                let delays = &mut self.delays[usize::from(place.channel)];
                self.coefficients
                    .iter()
                    .zip(delays.iter_mut())
                    .fold(sample, |x, (coefficients, delay)| {
                        delay.run(coefficients, x)
                    })
            }
            false => sample,
        };

        Some(output)
    }
}

impl<I: Source> Source for Equalized<I> {
    fn current_span_len(&self) -> Option<usize> {
        self.input.current_span_len()
    }

    fn channels(&self) -> NonZero<u16> {
        self.input.channels()
    }

    fn sample_rate(&self) -> NonZero<u32> {
        self.input.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.input.total_duration()
    }

    fn try_seek(&mut self, position: Duration) -> Result<(), SeekError> {
        self.input.try_seek(position)?;
        for delays in &mut self.delays {
            *delays = [Delay::default(); BANDS];
        }
        Ok(())
    }
}
